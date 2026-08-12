"""Local WebSocket bridge for VoxCPM2 backends.

The GPU model lives in a separate Python 3.11 environment.  The realtime
Pipecat process only sees mono PCM16 chunks over localhost, keeping the two
dependency stacks isolated.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Callable, Protocol


logger = logging.getLogger(__name__)


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


class PCMBackend(Protocol):
    sample_rate: int

    async def start(self) -> None: ...

    async def generate_pcm16(
        self,
        text: str,
        context_id: str = "",
    ) -> AsyncIterator[bytes]: ...

    async def release_context(self, context_id: str) -> None: ...

    async def stop(self) -> None: ...


class NanoVoxCPM2Backend:
    """Thin adapter around AsyncVoxCPM2ServerPool.

    Prompt audio is encoded exactly once during startup: ``add_prompt`` when an
    accurate transcript exists, otherwise ``encode_latents`` for ref-only
    cloning. A cancelled consumer closes Nano's async generator; Nano then
    calls ``cancel(seq_id)`` in its own ``finally`` block.
    """

    def __init__(
        self,
        *,
        model_path: str,
        prompt_wav: str,
        prompt_text: str,
        devices: list[int],
        inference_timesteps: int = 10,
        gpu_memory_utilization: float = 0.90,
        max_num_batched_tokens: int = 8192,
        max_num_seqs: int = 16,
        pool_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._model_path = model_path
        self._prompt_wav = Path(prompt_wav)
        self._prompt_text = prompt_text.strip()
        self._devices = devices
        self._inference_timesteps = inference_timesteps
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_num_batched_tokens = max_num_batched_tokens
        self._max_num_seqs = max_num_seqs
        self._pool_factory = pool_factory
        self._pool: Any | None = None
        self._prompt_id: str | None = None
        self._ref_audio_latents: bytes | None = None
        self.sample_rate = 0
        self._leading_trim_enabled = _bool_env("VOXCPM2_LEADING_TRIM_ENABLED")
        self._leading_trim_dbfs = -60.0
        self._leading_trim_confirm_ms = 20
        self._leading_trim_preroll_ms = 40
        self._leading_trim_max_scan_ms = 320
        if self._leading_trim_enabled:
            self._leading_trim_dbfs = float(
                os.getenv("VOXCPM2_LEADING_TRIM_DBFS", "-60.0")
            )
            self._leading_trim_confirm_ms = int(
                os.getenv("VOXCPM2_LEADING_TRIM_CONFIRM_MS", "20")
            )
            self._leading_trim_preroll_ms = int(
                os.getenv("VOXCPM2_LEADING_TRIM_PREROLL_MS", "40")
            )
            self._leading_trim_max_scan_ms = int(
                os.getenv("VOXCPM2_LEADING_TRIM_MAX_SCAN_MS", "320")
            )
            if not -120.0 <= self._leading_trim_dbfs <= 0.0:
                raise ValueError("VOXCPM2_LEADING_TRIM_DBFS must be between -120 and 0")
            if self._leading_trim_confirm_ms <= 0:
                raise ValueError("VOXCPM2_LEADING_TRIM_CONFIRM_MS must be positive")
            if self._leading_trim_preroll_ms < 0:
                raise ValueError("VOXCPM2_LEADING_TRIM_PREROLL_MS must be non-negative")
            if self._leading_trim_max_scan_ms < self._leading_trim_confirm_ms:
                raise ValueError(
                    "VOXCPM2_LEADING_TRIM_MAX_SCAN_MS must cover the confirmation window"
                )

    async def start(self) -> None:
        if self._pool is not None:
            return
        if not self._prompt_wav.is_file():
            raise FileNotFoundError(f"VoxCPM2 prompt wav not found: {self._prompt_wav}")

        pool_factory = self._pool_factory
        if pool_factory is None:
            from nanovllm_voxcpm import VoxCPM

            # The public factory selects the VoxCPM2 async server pool from the
            # checkpoint architecture while running inside this event loop.
            pool_factory = VoxCPM.from_pretrained
        pool = pool_factory(
            model=self._model_path,
            devices=self._devices,
            inference_timesteps=self._inference_timesteps,
            gpu_memory_utilization=self._gpu_memory_utilization,
            max_num_batched_tokens=self._max_num_batched_tokens,
            max_num_seqs=self._max_num_seqs,
        )
        try:
            await pool.wait_for_ready()
            model_info = await pool.get_model_info()
            sample_rate = int(model_info["output_sample_rate"])
            if sample_rate <= 0:
                raise RuntimeError(
                    f"invalid VoxCPM2 output sample rate: {sample_rate}"
                )

            wav_bytes = self._prompt_wav.read_bytes()
            wav_format = self._prompt_wav.suffix.lstrip(".") or "wav"
            if self._prompt_text:
                prompt_id = await pool.add_prompt(
                    wav_bytes,
                    wav_format,
                    self._prompt_text,
                )
                ref_audio_latents = None
            else:
                # Reference-only cloning remains usable when the user has not yet
                # supplied a transcript. This is encoded once, then reused.
                prompt_id = None
                ref_audio_latents = await pool.encode_latents(wav_bytes, wav_format)
        except BaseException:
            # Pool construction can spawn Nano workers immediately. Never leave
            # those workers (and their CUDA allocations) behind when any later
            # readiness/model/prompt operation fails or startup is cancelled.
            try:
                await pool.stop()
            except BaseException:
                logger.exception("VoxCPM2 pool cleanup failed during startup")
            raise
        self._pool = pool
        self._prompt_id = prompt_id
        self._ref_audio_latents = ref_audio_latents
        self.sample_rate = sample_rate

    async def generate_pcm16(
        self,
        text: str,
        context_id: str = "",
    ) -> AsyncIterator[bytes]:
        del context_id
        pool = self._pool
        prompt_id = self._prompt_id
        ref_audio_latents = self._ref_audio_latents
        if pool is None or (prompt_id is None and ref_audio_latents is None):
            raise RuntimeError("VoxCPM2 backend is not started")

        import numpy as np

        generation_kwargs: dict[str, Any] = {"target_text": text}
        if prompt_id is not None:
            generation_kwargs["prompt_id"] = prompt_id
        else:
            generation_kwargs["ref_audio_latents"] = ref_audio_latents
        if self._leading_trim_enabled and self.sample_rate <= 0:
            raise RuntimeError("cannot trim leading silence without a valid sample rate")

        pending: list[tuple[Any, bytes]] = []
        scan_samples = np.empty(0, dtype=np.float32)
        pass_through = not self._leading_trim_enabled
        frame_samples = max(1, round(self.sample_rate * 0.010)) if not pass_through else 1
        confirm_frames = max(
            1,
            (self._leading_trim_confirm_ms + 9) // 10,
        )
        preroll_samples = round(self.sample_rate * self._leading_trim_preroll_ms / 1000)
        max_scan_samples = round(
            self.sample_rate * self._leading_trim_max_scan_ms / 1000
        )
        threshold = 10.0 ** (self._leading_trim_dbfs / 20.0)

        async for waveform in pool.generate(**generation_kwargs):
            samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
            if samples.size == 0:
                continue
            pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
            pcm_bytes = pcm16.tobytes()
            if pass_through:
                yield pcm_bytes
                continue

            pending.append((samples, pcm_bytes))
            scan_samples = np.concatenate((scan_samples, samples))
            analysis = scan_samples[:max_scan_samples]
            frame_count = analysis.size // frame_samples
            onset_sample: int | None = None
            if frame_count >= confirm_frames:
                frames = analysis[: frame_count * frame_samples].reshape(
                    frame_count,
                    frame_samples,
                )
                rms = np.sqrt(np.mean(np.square(frames), axis=1, dtype=np.float64))
                active = rms >= threshold
                for index in range(frame_count - confirm_frames + 1):
                    if bool(np.all(active[index : index + confirm_frames])):
                        onset_sample = index * frame_samples
                        break

            if onset_sample is not None:
                samples_to_drop = max(0, onset_sample - preroll_samples)
                if samples_to_drop:
                    logger.info(
                        "trimmed %.1f ms of generated leading silence",
                        samples_to_drop * 1000.0 / self.sample_rate,
                    )
                remaining_drop = samples_to_drop
                for pending_samples, pending_pcm in pending:
                    if remaining_drop >= pending_samples.size:
                        remaining_drop -= pending_samples.size
                        continue
                    if remaining_drop:
                        yield pending_pcm[remaining_drop * 2 :]
                        remaining_drop = 0
                    else:
                        yield pending_pcm
                pending.clear()
                pass_through = True
            elif scan_samples.size >= max_scan_samples:
                # No confident onset: preserve the original stream byte-for-byte.
                for _, pending_pcm in pending:
                    yield pending_pcm
                pending.clear()
                pass_through = True

        # Short or silence-only generations never crossed the scan limit. Keep
        # their original chunks and samples rather than risking a clipped onset.
        for _, pending_pcm in pending:
            yield pending_pcm

    async def release_context(self, context_id: str) -> None:
        del context_id

    async def stop(self) -> None:
        pool = self._pool
        self._pool = None
        self._prompt_id = None
        self._ref_audio_latents = None
        if pool is not None:
            await pool.stop()


class VoxCPM2WebSocketBridge:
    """One synthesis request per connection, with in-band cancellation."""

    def __init__(self, backend: PCMBackend) -> None:
        self._backend = backend

    async def handle_connection(self, websocket: Any) -> None:
        request_id = ""
        context_id = ""
        generation_task: asyncio.Task[int] | None = None
        control_task: asyncio.Task[None] | None = None
        cancelled = asyncio.Event()
        try:
            raw_request = await websocket.recv()
            request = self._parse_request(raw_request)
            request_type = request.get("type")
            if request_type == "health":
                # Candidate network-backed PCM engines can expose an async
                # liveness check. Native VoxCPM2 backends intentionally keep
                # the existing zero-overhead health behavior.
                health_check = getattr(self._backend, "health_check", None)
                if callable(health_check):
                    await health_check()
                await websocket.send(
                    json.dumps(
                        {
                            "type": "health",
                            "status": "ok",
                            "sample_rate": self._backend.sample_rate,
                        }
                    )
                )
                return
            if request_type == "release_context":
                context_id = str(request.get("context_id", "")).strip()
                if not context_id:
                    raise ValueError("release_context requires non-empty context_id")
                await self._backend.release_context(context_id)
                await websocket.send(
                    json.dumps(
                        {
                            "type": "released",
                            "context_id": context_id,
                        }
                    )
                )
                return
            if request_type != "synthesize":
                raise ValueError(
                    "first message must be health, release_context, or synthesize"
                )

            request_id = str(request.get("request_id", "")).strip()
            context_id = str(request.get("context_id", "")).strip() or request_id
            text = str(request.get("text", "")).strip()
            if not request_id or not text:
                raise ValueError("synthesize requires non-empty request_id and text")

            await websocket.send(
                json.dumps(
                    {
                        "type": "start",
                        "request_id": request_id,
                        "context_id": context_id,
                        "sample_rate": self._backend.sample_rate,
                        "channels": 1,
                        "sample_width": 2,
                        "audio_format": "pcm_s16le",
                    }
                )
            )
            generation_task = asyncio.create_task(
                self._stream_audio(websocket, request_id, context_id, text),
                name=f"voxcpm2-generate-{request_id}",
            )
            control_task = asyncio.create_task(
                self._watch_control(websocket, request_id, cancelled),
                name=f"voxcpm2-control-{request_id}",
            )
            done, _ = await asyncio.wait(
                (generation_task, control_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if control_task in done:
                generation_task.cancel()
                await asyncio.gather(generation_task, return_exceptions=True)
                status = "cancelled" if cancelled.is_set() else "disconnected"
                await self._try_send_json(
                    websocket,
                    {
                        "type": "done",
                        "request_id": request_id,
                        "context_id": context_id,
                        "status": status,
                    },
                )
                try:
                    await self._backend.release_context(context_id)
                except Exception:
                    logger.exception(
                        "failed to release cancelled VoxCPM2 context %s",
                        context_id,
                    )
            else:
                await generation_task
                await self._try_send_json(
                    websocket,
                    {
                        "type": "done",
                        "request_id": request_id,
                        "context_id": context_id,
                        "status": "completed",
                    },
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._try_send_json(
                websocket,
                {
                    "type": "error",
                    "request_id": request_id,
                    "context_id": context_id,
                    "error": str(exc),
                },
            )
        finally:
            for task in (generation_task, control_task):
                if task is not None and not task.done():
                    task.cancel()
            pending = [
                task
                for task in (generation_task, control_task)
                if task is not None
            ]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def _stream_audio(
        self,
        websocket: Any,
        request_id: str,
        context_id: str,
        text: str,
    ) -> int:
        pcm_bytes = 0
        async for pcm16 in self._backend.generate_pcm16(text, context_id):
            if not isinstance(pcm16, bytes) or not pcm16 or len(pcm16) % 2:
                raise RuntimeError("backend returned invalid PCM16 chunk")
            await websocket.send(pcm16)
            pcm_bytes += len(pcm16)
        if pcm_bytes == 0:
            raise RuntimeError("backend completed without PCM audio")
        return pcm_bytes

    @staticmethod
    async def _watch_control(
        websocket: Any,
        request_id: str,
        cancelled: asyncio.Event,
    ) -> None:
        while True:
            message = VoxCPM2WebSocketBridge._parse_request(await websocket.recv())
            if (
                message.get("type") == "cancel"
                and str(message.get("request_id", "")) == request_id
            ):
                cancelled.set()
                return

    @staticmethod
    def _parse_request(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, str):
            raise ValueError("control messages must be JSON text")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("control message must be a JSON object")
        return value

    @staticmethod
    async def _try_send_json(websocket: Any, message: dict[str, Any]) -> None:
        try:
            await websocket.send(json.dumps(message))
        except Exception:
            pass


async def _warm_backend(backend: PCMBackend, text: str) -> tuple[int, int]:
    """Run one complete synthesis before the bridge can report healthy."""

    chunks = 0
    pcm_bytes = 0
    context_id = "__warmup__"
    try:
        async for pcm16 in backend.generate_pcm16(text, context_id):
            if not isinstance(pcm16, bytes) or not pcm16 or len(pcm16) % 2:
                raise RuntimeError("VoxCPM2 warmup returned invalid PCM16 data")
            chunks += 1
            pcm_bytes += len(pcm16)
    finally:
        await backend.release_context(context_id)
    if pcm_bytes == 0:
        raise RuntimeError("VoxCPM2 warmup returned no audio")
    return chunks, pcm_bytes


def _prompt_text_from_env() -> str:
    inline = os.getenv("VOXCPM2_PROMPT_TEXT", "").strip()
    path = os.getenv("VOXCPM2_PROMPT_TEXT_FILE", "").strip()
    if inline and path:
        raise ValueError("set only one of VOXCPM2_PROMPT_TEXT or VOXCPM2_PROMPT_TEXT_FILE")
    if path:
        return Path(path).read_text(encoding="utf-8").strip()
    return inline


def _devices_from_env() -> list[int]:
    raw = os.getenv("VOXCPM2_DEVICES", "0")
    devices = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not devices:
        raise ValueError("VOXCPM2_DEVICES must contain at least one device index")
    return devices


def _backend_from_env() -> PCMBackend:
    backend_name = os.getenv("VOXCPM2_BACKEND", "nano").strip().lower()
    if backend_name == "nano":
        return NanoVoxCPM2Backend(
            model_path=os.environ["VOXCPM2_MODEL_PATH"],
            prompt_wav=os.environ["VOXCPM2_PROMPT_WAV"],
            prompt_text=_prompt_text_from_env(),
            devices=_devices_from_env(),
            inference_timesteps=int(os.getenv("VOXCPM2_INFERENCE_TIMESTEPS", "10")),
            gpu_memory_utilization=float(
                os.getenv("VOXCPM2_GPU_MEMORY_UTILIZATION", "0.90")
            ),
            max_num_batched_tokens=int(
                os.getenv("VOXCPM2_MAX_BATCHED_TOKENS", "8192")
            ),
            max_num_seqs=int(os.getenv("VOXCPM2_MAX_NUM_SEQS", "16")),
        )
    if backend_name == "official_prompt_cache":
        from voice_service.official_voxcpm2_backend import (
            OfficialPromptCacheBackend,
        )

        return OfficialPromptCacheBackend(
            model_path=os.environ["VOXCPM2_MODEL_PATH"],
            reference_wav=os.environ["VOXCPM2_PROMPT_WAV"],
            source_path=os.getenv("VOXCPM2_OFFICIAL_SOURCE", "").strip(),
            device=os.getenv("VOXCPM2_OFFICIAL_DEVICE", "cuda").strip(),
            optimize=_bool_env("VOXCPM2_OFFICIAL_OPTIMIZE", True),
            inference_timesteps=int(os.getenv("VOXCPM2_INFERENCE_TIMESTEPS", "10")),
            cfg_value=float(os.getenv("VOXCPM2_CFG_VALUE", "2.0")),
            seed=int(os.getenv("VOXCPM2_SEED", "42")),
        )
    raise ValueError(
        "VOXCPM2_BACKEND must be nano or official_prompt_cache, "
        f"got {backend_name!r}"
    )


async def _run_server(
    host: str,
    port: int,
    *,
    _backend: PCMBackend | None = None,
    _serve_factory: Callable[..., Any] | None = None,
    _shutdown_requested: asyncio.Event | None = None,
) -> None:
    if _serve_factory is None:
        from websockets.asyncio.server import serve

        _serve_factory = serve

    backend = _backend or _backend_from_env()
    loop = asyncio.get_running_loop()
    shutdown_requested = _shutdown_requested or asyncio.Event()
    installed_signals: list[signal.Signals] = []

    def request_shutdown(received_signal: signal.Signals) -> None:
        logger.info("received %s; draining VoxCPM2 bridge", received_signal.name)
        shutdown_requested.set()

    for received_signal in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(
                received_signal,
                request_shutdown,
                received_signal,
            )
            installed_signals.append(received_signal)
        except (NotImplementedError, RuntimeError):
            # Unix production hosts support loop signal handlers. This fallback
            # keeps import/unit tests portable on Windows.
            pass

    start_task = asyncio.create_task(backend.start(), name="voxcpm2-backend-start")
    shutdown_task = asyncio.create_task(
        shutdown_requested.wait(),
        name="voxcpm2-shutdown-wait",
    )
    try:
        done, _ = await asyncio.wait(
            (start_task, shutdown_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if shutdown_task in done and not start_task.done():
            start_task.cancel()
            await asyncio.gather(start_task, return_exceptions=True)
            return
        await start_task

        warmup_text = os.getenv("VOXCPM2_WARMUP_TEXT", "").strip()
        if warmup_text:
            warmup_timeout = float(
                os.getenv("VOXCPM2_WARMUP_TIMEOUT_SEC", "120.0")
            )
            if warmup_timeout <= 0:
                raise ValueError("VOXCPM2_WARMUP_TIMEOUT_SEC must be positive")
            warmup_started = loop.time()
            async with asyncio.timeout(warmup_timeout):
                chunks, pcm_bytes = await _warm_backend(backend, warmup_text)
            logger.info(
                "VoxCPM2 warmup complete: elapsed_ms=%.1f chunks=%d pcm_bytes=%d",
                (loop.time() - warmup_started) * 1000,
                chunks,
                pcm_bytes,
            )

        bridge = VoxCPM2WebSocketBridge(backend)
        async with _serve_factory(
            bridge.handle_connection,
            host,
            port,
            max_size=2 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        ):
            await shutdown_requested.wait()
    finally:
        for task in (start_task, shutdown_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(start_task, shutdown_task, return_exceptions=True)
        await backend.stop()
        for received_signal in installed_signals:
            loop.remove_signal_handler(received_signal)


def main() -> None:
    logging.basicConfig(
        level=os.getenv("VOXCPM2_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Local VoxCPM2 PCM bridge")
    parser.add_argument("--host", default=os.getenv("VOXCPM2_BRIDGE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("VOXCPM2_BRIDGE_PORT", "8770")))
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("VoxCPM2 bridge must bind to localhost")
    asyncio.run(_run_server(args.host, args.port))


if __name__ == "__main__":
    main()
