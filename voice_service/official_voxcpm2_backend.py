"""Optional official VoxCPM2 backend with per-turn prompt-cache continuation.

The official package is imported only by :meth:`start`, so importing this
module remains safe in the Nano/Pipecat environment.  The official streaming
generator is synchronous; all model work therefore runs in a worker thread and
never blocks the asyncio event loop.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
import threading
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class _LeadingTrimConfig:
    enabled: bool
    dbfs: float
    confirm_ms: int
    preroll_ms: int
    max_scan_ms: int

    @classmethod
    def from_env(cls) -> "_LeadingTrimConfig":
        enabled = _bool_env("VOXCPM2_LEADING_TRIM_ENABLED")
        if not enabled:
            return cls(
                enabled=False,
                dbfs=-60.0,
                confirm_ms=20,
                preroll_ms=40,
                max_scan_ms=320,
            )
        config = cls(
            enabled=True,
            dbfs=float(os.getenv("VOXCPM2_LEADING_TRIM_DBFS", "-60.0")),
            confirm_ms=int(os.getenv("VOXCPM2_LEADING_TRIM_CONFIRM_MS", "20")),
            preroll_ms=int(os.getenv("VOXCPM2_LEADING_TRIM_PREROLL_MS", "40")),
            max_scan_ms=int(os.getenv("VOXCPM2_LEADING_TRIM_MAX_SCAN_MS", "320")),
        )
        if enabled:
            if not -120.0 <= config.dbfs <= 0.0:
                raise ValueError("VOXCPM2_LEADING_TRIM_DBFS must be between -120 and 0")
            if config.confirm_ms <= 0:
                raise ValueError("VOXCPM2_LEADING_TRIM_CONFIRM_MS must be positive")
            if config.preroll_ms < 0:
                raise ValueError("VOXCPM2_LEADING_TRIM_PREROLL_MS must be non-negative")
            if config.max_scan_ms < config.confirm_ms:
                raise ValueError(
                    "VOXCPM2_LEADING_TRIM_MAX_SCAN_MS must cover the confirmation window"
                )
        return config


class _LeadingPCMFilter:
    """The same opt-in leading-silence policy used by the Nano backend."""

    def __init__(self, sample_rate: int, config: _LeadingTrimConfig) -> None:
        import numpy as np

        self._np = np
        self._sample_rate = sample_rate
        self._config = config
        self._pending: list[tuple[Any, bytes]] = []
        self._scan_samples = np.empty(0, dtype=np.float32)
        self._pass_through = not config.enabled
        self._frame_samples = max(1, round(sample_rate * 0.010))
        self._confirm_frames = max(1, (config.confirm_ms + 9) // 10)
        self._preroll_samples = round(sample_rate * config.preroll_ms / 1000)
        self._max_scan_samples = round(sample_rate * config.max_scan_ms / 1000)
        self._threshold = 10.0 ** (config.dbfs / 20.0)

    def push(self, waveform: Any) -> list[bytes]:
        np = self._np
        samples = self._to_samples(waveform)
        if samples.size == 0:
            return []
        pcm_bytes = self._to_pcm16(samples)
        if self._pass_through:
            return [pcm_bytes]

        self._pending.append((samples, pcm_bytes))
        self._scan_samples = np.concatenate((self._scan_samples, samples))
        analysis = self._scan_samples[: self._max_scan_samples]
        frame_count = analysis.size // self._frame_samples
        onset_sample: int | None = None
        if frame_count >= self._confirm_frames:
            frames = analysis[: frame_count * self._frame_samples].reshape(
                frame_count,
                self._frame_samples,
            )
            rms = np.sqrt(np.mean(np.square(frames), axis=1, dtype=np.float64))
            active = rms >= self._threshold
            for index in range(frame_count - self._confirm_frames + 1):
                if bool(np.all(active[index : index + self._confirm_frames])):
                    onset_sample = index * self._frame_samples
                    break

        if onset_sample is not None:
            return self._release(max(0, onset_sample - self._preroll_samples))
        if self._scan_samples.size >= self._max_scan_samples:
            return self._release(0)
        return []

    def finish(self) -> list[bytes]:
        if self._pass_through:
            return []
        return self._release(0)

    def _to_samples(self, waveform: Any) -> Any:
        value = waveform
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        to_numpy = getattr(value, "numpy", None)
        if callable(to_numpy):
            value = to_numpy()
        return self._np.asarray(value, dtype=self._np.float32).reshape(-1)

    def _to_pcm16(self, samples: Any) -> bytes:
        return (
            self._np.clip(samples, -1.0, 1.0) * 32767.0
        ).astype("<i2").tobytes()

    def _release(self, samples_to_drop: int) -> list[bytes]:
        chunks: list[bytes] = []
        remaining_drop = samples_to_drop
        for pending_samples, pending_pcm in self._pending:
            if remaining_drop >= pending_samples.size:
                remaining_drop -= pending_samples.size
                continue
            if remaining_drop:
                chunks.append(pending_pcm[remaining_drop * 2 :])
                remaining_drop = 0
            else:
                chunks.append(pending_pcm)
        self._pending.clear()
        self._pass_through = True
        return chunks


@dataclass
class _ContextState:
    cache: Mapping[str, Any]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    invalidated: threading.Event = field(default_factory=threading.Event)
    active_stops: set[threading.Event] = field(default_factory=set)


@dataclass(frozen=True)
class _WorkerMessage:
    kind: str
    value: Any = None


class OfficialPromptCacheBackend:
    """Official VoxCPM2 streaming backend with context-scoped prompt caches.

    A context is created lazily by its first sentence.  A sentence updates the
    context only after its generator and cache merge both finish successfully.
    Releasing a context tombstones its identifier, so a late request cannot
    recreate or commit stale state.
    """

    def __init__(
        self,
        *,
        model_path: str,
        reference_wav: str,
        source_path: str = "",
        device: str = "cuda:0",
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        streaming_prefix_len: int = 4,
        seed: int = 42,
        optimize: bool = True,
        local_files_only: bool = True,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._model_path = model_path
        self._reference_wav = Path(reference_wav)
        self._source_path = source_path.strip()
        self._inserted_source_path: str | None = None
        self._device = device
        self._inference_timesteps = inference_timesteps
        self._cfg_value = cfg_value
        self._streaming_prefix_len = streaming_prefix_len
        self._seed = seed
        self._optimize = optimize
        self._local_files_only = local_files_only
        self._model_factory = model_factory
        self._trim_config = _LeadingTrimConfig.from_env()

        self._model: Any | None = None
        self._base_cache: Mapping[str, Any] | None = None
        self.sample_rate = 0
        self._accepting = False
        self._lifecycle_lock = asyncio.Lock()
        self._contexts_lock = asyncio.Lock()
        self._contexts: dict[str, _ContextState] = {}
        self._released_contexts: OrderedDict[str, None] = OrderedDict()
        self._released_context_limit = int(
            os.getenv("VOXCPM2_CONTEXT_TOMBSTONE_LIMIT", "4096")
        )
        if self._released_context_limit <= 0:
            raise ValueError("VOXCPM2_CONTEXT_TOMBSTONE_LIMIT must be positive")
        self._active_tasks: set[asyncio.Task[None]] = set()
        self._model_lock = threading.Lock()

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._model is not None:
                return
            if not self._reference_wav.is_file():
                raise FileNotFoundError(
                    f"VoxCPM2 prompt wav not found: {self._reference_wav}"
                )

            if self._source_path:
                source_path = str(Path(self._source_path).resolve())
                if source_path not in sys.path:
                    sys.path.insert(0, source_path)
                    self._inserted_source_path = source_path

            factory = self._model_factory
            try:
                if factory is None:
                    from voxcpm import VoxCPM

                    factory = VoxCPM.from_pretrained

                def load() -> tuple[Any, Mapping[str, Any], int]:
                    model = factory(
                        self._model_path,
                        load_denoiser=False,
                        local_files_only=self._local_files_only,
                        optimize=self._optimize,
                        device=self._device,
                    )
                    try:
                        tts_model = model.tts_model
                        base_cache = tts_model.build_prompt_cache(
                            reference_wav_path=str(self._reference_wav)
                        )
                        sample_rate = int(tts_model.sample_rate)
                        if sample_rate <= 0:
                            raise RuntimeError(
                                f"invalid VoxCPM2 output sample rate: {sample_rate}"
                            )
                        if not isinstance(base_cache, Mapping):
                            raise RuntimeError("VoxCPM2 returned an invalid prompt cache")
                        return model, base_cache, sample_rate
                    except BaseException:
                        self._close_model_sync(model)
                        raise

                model, base_cache, sample_rate = await asyncio.to_thread(load)
            except BaseException:
                self._remove_inserted_source_path()
                raise
            async with self._contexts_lock:
                self._contexts.clear()
                self._released_contexts.clear()
            self._model = model
            self._base_cache = base_cache
            self.sample_rate = sample_rate
            self._accepting = True

    async def generate_pcm16(
        self,
        text: str,
        context_id: str,
    ) -> AsyncIterator[bytes]:
        if not text.strip():
            return
        if not context_id:
            raise ValueError("context_id must be non-empty")

        state = await self._get_or_create_context(context_id)
        if state is None:
            return

        async with state.lock:
            if not await self._context_is_current(context_id, state):
                return
            model = self._model
            if model is None:
                raise RuntimeError("VoxCPM2 backend is not started")

            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[_WorkerMessage] = asyncio.Queue()
            stop_requested = threading.Event()
            state.active_stops.add(stop_requested)
            worker_task = asyncio.create_task(
                asyncio.to_thread(
                    self._run_generation,
                    loop,
                    queue,
                    model,
                    state,
                    text,
                    stop_requested,
                )
            )
            self._active_tasks.add(worker_task)
            completed_normally = False
            try:
                while True:
                    message = await queue.get()
                    if message.kind == "pcm":
                        if not stop_requested.is_set() and not state.invalidated.is_set():
                            yield message.value
                    elif message.kind == "done":
                        async with self._contexts_lock:
                            if (
                                self._contexts.get(context_id) is state
                                and context_id not in self._released_contexts
                                and not state.invalidated.is_set()
                                and not stop_requested.is_set()
                            ):
                                state.cache = message.value
                                completed_normally = True
                        return
                    elif message.kind == "cancelled":
                        return
                    elif message.kind == "error":
                        raise message.value
                    else:
                        raise RuntimeError(f"unknown VoxCPM2 worker message: {message.kind}")
            finally:
                if not completed_normally:
                    stop_requested.set()
                cancelled_during_wait = False
                try:
                    while not worker_task.done():
                        try:
                            await asyncio.shield(worker_task)
                        except asyncio.CancelledError:
                            cancelled_during_wait = True
                            stop_requested.set()
                    worker_task.result()
                finally:
                    state.active_stops.discard(stop_requested)
                    self._active_tasks.discard(worker_task)
                if cancelled_during_wait:
                    raise asyncio.CancelledError

    async def release_context(
        self,
        context_id: str,
        *,
        interrupted: bool = False,
    ) -> None:
        del interrupted  # Both paths invalidate state; the reason is diagnostic only.
        if not context_id:
            return
        async with self._contexts_lock:
            self._released_contexts[context_id] = None
            self._released_contexts.move_to_end(context_id)
            while len(self._released_contexts) > self._released_context_limit:
                self._released_contexts.popitem(last=False)
            state = self._contexts.pop(context_id, None)
            if state is not None:
                state.invalidated.set()
                for stop_requested in tuple(state.active_stops):
                    stop_requested.set()

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            self._accepting = False
            async with self._contexts_lock:
                states = tuple(self._contexts.values())
                self._contexts.clear()
                for state in states:
                    state.invalidated.set()
                    for stop_requested in tuple(state.active_stops):
                        stop_requested.set()

            active_tasks = tuple(self._active_tasks)
            if active_tasks:
                await asyncio.gather(
                    *(asyncio.shield(task) for task in active_tasks),
                    return_exceptions=True,
                )
            self._active_tasks.clear()

            model = self._model
            self._model = None
            self._base_cache = None
            self.sample_rate = 0
            self._released_contexts.clear()
            try:
                if model is not None:
                    await self._close_model(model)
            finally:
                self._remove_inserted_source_path()

    async def _get_or_create_context(self, context_id: str) -> _ContextState | None:
        async with self._contexts_lock:
            if self._model is None or self._base_cache is None or not self._accepting:
                raise RuntimeError("VoxCPM2 backend is not started")
            if context_id in self._released_contexts:
                return None
            state = self._contexts.get(context_id)
            if state is None:
                state = _ContextState(cache=dict(self._base_cache))
                self._contexts[context_id] = state
            return state

    async def _context_is_current(
        self,
        context_id: str,
        state: _ContextState,
    ) -> bool:
        async with self._contexts_lock:
            return (
                self._accepting
                and self._contexts.get(context_id) is state
                and context_id not in self._released_contexts
                and not state.invalidated.is_set()
            )

    def _run_generation(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[_WorkerMessage],
        model: Any,
        state: _ContextState,
        text: str,
        stop_requested: threading.Event,
    ) -> None:
        generator: Any | None = None
        acquired = False
        try:
            while not self._cancelled(state, stop_requested):
                acquired = self._model_lock.acquire(timeout=0.05)
                if acquired:
                    break
            if not acquired or self._cancelled(state, stop_requested):
                self._post(loop, queue, _WorkerMessage("cancelled"))
                return

            tts_model = model.tts_model
            trimmer = _LeadingPCMFilter(self.sample_rate, self._trim_config)
            generated_features: list[Any] = []
            waveform_seen = False
            generator = tts_model.generate_with_prompt_cache_streaming(
                target_text=text,
                prompt_cache=state.cache,
                inference_timesteps=self._inference_timesteps,
                cfg_value=self._cfg_value,
                streaming_prefix_len=self._streaming_prefix_len,
                seed=self._seed,
            )
            for waveform, _text_tokens, feature_sequence in generator:
                if self._cancelled(state, stop_requested):
                    self._post(loop, queue, _WorkerMessage("cancelled"))
                    return
                chunks = trimmer.push(waveform)
                if chunks:
                    waveform_seen = True
                    for chunk in chunks:
                        self._post(loop, queue, _WorkerMessage("pcm", chunk))
                if not feature_sequence:
                    raise RuntimeError("official VoxCPM2 returned no generated feature")
                generated_features.append(self._detach_cpu(feature_sequence[-1]))

            if self._cancelled(state, stop_requested):
                self._post(loop, queue, _WorkerMessage("cancelled"))
                return
            tail = trimmer.finish()
            if tail:
                waveform_seen = True
                for chunk in tail:
                    self._post(loop, queue, _WorkerMessage("pcm", chunk))
            if not waveform_seen or not generated_features:
                raise RuntimeError("official VoxCPM2 returned an incomplete stream")

            new_audio_feature = self._concatenate_features(generated_features)
            merged_cache = tts_model.merge_prompt_cache(
                state.cache,
                text,
                new_audio_feature,
            )
            if self._cancelled(state, stop_requested):
                self._post(loop, queue, _WorkerMessage("cancelled"))
                return
            self._post(loop, queue, _WorkerMessage("done", merged_cache))
        except BaseException as error:
            self._post(loop, queue, _WorkerMessage("error", error))
        finally:
            try:
                close = getattr(generator, "close", None)
                if callable(close):
                    close()
            finally:
                if acquired:
                    self._model_lock.release()

    @staticmethod
    def _cancelled(state: _ContextState, stop_requested: threading.Event) -> bool:
        return state.invalidated.is_set() or stop_requested.is_set()

    @staticmethod
    def _post(
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[_WorkerMessage],
        message: _WorkerMessage,
    ) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, message)
        except RuntimeError:
            pass

    @staticmethod
    def _detach_cpu(value: Any) -> Any:
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        return value

    @staticmethod
    def _concatenate_features(features: list[Any]) -> Any:
        first = features[0]
        if first.__class__.__module__.split(".", 1)[0] == "torch":
            import torch

            return torch.cat(features, dim=1).squeeze(0)

        import numpy as np

        return np.concatenate(
            [np.asarray(feature) for feature in features],
            axis=1,
        ).squeeze(0)

    @classmethod
    async def _close_model(cls, model: Any) -> None:
        closer = getattr(model, "close", None)
        if not callable(closer):
            return
        result = await asyncio.to_thread(closer)
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _close_model_sync(model: Any) -> None:
        closer = getattr(model, "close", None)
        if callable(closer):
            result = closer()
            if inspect.isawaitable(result):
                asyncio.run(result)

    def _remove_inserted_source_path(self) -> None:
        if self._inserted_source_path is None:
            return
        try:
            sys.path.remove(self._inserted_source_path)
        except ValueError:
            pass
        self._inserted_source_path = None
