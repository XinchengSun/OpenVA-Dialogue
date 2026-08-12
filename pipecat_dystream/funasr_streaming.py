"""Stateful Paraformer streaming ASR for Pipecat 1.6.

Pipecat's built-in FunASR service is segmented: it waits for a complete VAD
segment and creates a fresh cache for every inference.  This service feeds raw
16 kHz mono PCM to ``paraformer-zh-streaming`` as fixed-size chunks and keeps
the FunASR cache for the duration of one user utterance.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any, Protocol

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601


SAMPLE_RATE = 16_000
_FUNASR_CHUNK_UNIT_SAMPLES = 960  # 60 ms at 16 kHz.


class StreamingProvider(Protocol):
    """Synchronous inference boundary used by the service and its tests."""

    def generate(
        self,
        audio: np.ndarray,
        *,
        cache: dict[str, Any],
        is_final: bool,
        chunk_size: tuple[int, int, int],
        encoder_chunk_look_back: int,
        decoder_chunk_look_back: int,
    ) -> str: ...


class FunASRAutoModelProvider:
    """Thin adapter around FunASR ``AutoModel``.

    The import is intentionally lazy so unit tests with a fake provider do not
    import FunASR or download model weights.
    """

    def __init__(
        self,
        *,
        model: str = "paraformer-zh-streaming",
        device: str = "cpu",
        hub: str = "ms",
    ):
        from funasr import AutoModel

        logger.info(f"Loading FunASR streaming model {model} on {device} via hub={hub}")
        self._model = AutoModel(
            model=model,
            device=device,
            hub=hub,
            disable_update=True,
        )

    def generate(
        self,
        audio: np.ndarray,
        *,
        cache: dict[str, Any],
        is_final: bool,
        chunk_size: tuple[int, int, int],
        encoder_chunk_look_back: int,
        decoder_chunk_look_back: int,
    ) -> str:
        result = self._model.generate(
            input=audio,
            cache=cache,
            is_final=is_final,
            chunk_size=list(chunk_size),
            encoder_chunk_look_back=encoder_chunk_look_back,
            decoder_chunk_look_back=decoder_chunk_look_back,
        )
        if not result:
            return ""
        return str(result[0].get("text", ""))


class ParaformerStreamingSTTService(STTService):
    """Incremental FunASR Paraformer service for one realtime conversation."""

    def __init__(
        self,
        *,
        provider: StreamingProvider | None = None,
        model: str = "paraformer-zh-streaming",
        device: str = "cpu",
        hub: str = "ms",
        chunk_size: tuple[int, int, int] = (0, 10, 5),
        encoder_chunk_look_back: int = 4,
        decoder_chunk_look_back: int = 1,
        pre_roll_secs: float = 0.30,
        language: Language = Language.ZH,
        audio_passthrough: bool = True,
        **kwargs,
    ):
        if len(chunk_size) != 3 or chunk_size[1] <= 0:
            raise ValueError("chunk_size must contain three values with a positive middle value")
        if encoder_chunk_look_back < 0 or decoder_chunk_look_back < 0:
            raise ValueError("chunk look-back values must be non-negative")
        if pre_roll_secs < 0:
            raise ValueError("pre_roll_secs must be non-negative")
        if not audio_passthrough:
            raise ValueError(
                "audio_passthrough must remain enabled because the downstream "
                "user aggregator owns VAD and emits the finalization boundary"
            )

        super().__init__(
            audio_passthrough=True,
            sample_rate=SAMPLE_RATE,
            settings=STTSettings(model=model, language=language),
            **kwargs,
        )
        self._provider = provider or FunASRAutoModelProvider(
            model=model,
            device=device,
            hub=hub,
        )
        self._chunk_size = tuple(chunk_size)
        self._chunk_bytes = chunk_size[1] * _FUNASR_CHUNK_UNIT_SAMPLES * 2
        self._encoder_chunk_look_back = encoder_chunk_look_back
        self._decoder_chunk_look_back = decoder_chunk_look_back
        self._pre_roll_secs = float(pre_roll_secs)
        self._pre_roll_bytes = round(self._pre_roll_secs * SAMPLE_RATE) * 2
        self._frame_language = language
        self._pre_roll_buffer = bytearray()
        self._audio_buffer = bytearray()
        self._cache: dict[str, Any] = {}
        self._transcript_parts: list[str] = []
        self._utterance_has_audio = False
        self._vad_active = False
        self._generation = 0

    @property
    def pre_roll_secs(self) -> float:
        return self._pre_roll_secs

    def can_generate_metrics(self) -> bool:
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, InputAudioRawFrame):
            error = self._validate_audio_frame(frame)
            if error is not None:
                await self.push_frame(ErrorFrame(error), FrameDirection.DOWNSTREAM)
                if self._audio_passthrough:
                    await self.push_frame(frame, direction)
                return

        if (
            isinstance(frame, VADUserStartedSpeakingFrame)
            and direction is FrameDirection.UPSTREAM
        ):
            await self._activate_utterance()
        elif isinstance(frame, InterruptionFrame):
            # Pipecat emits this after VAD confirms a barge-in. By then the first
            # audio of the new user turn is already in this service. Preserve it;
            # resetting here would consistently clip the first syllable.
            pass
        elif (
            isinstance(frame, VADUserStoppedSpeakingFrame)
            and direction is FrameDirection.UPSTREAM
        ):
            # Let Pipecat start its speech-end latency clock before the final
            # transcript is pushed; the finalized frame then stops that clock.
            await super().process_frame(frame, direction)
            await self.process_generator(self._finalize_utterance())
            return
        elif isinstance(frame, UserStoppedSpeakingFrame) and self._vad_active:
            await self.process_generator(self._finalize_utterance())
        elif isinstance(frame, EndFrame):
            if self._vad_active:
                await self.process_generator(self._finalize_utterance())
            else:
                self._reset_utterance()
        elif isinstance(frame, CancelFrame):
            self._reset_utterance()

        await super().process_frame(frame, direction)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Consume raw PCM and emit cumulative interim transcripts per full chunk."""
        if not self._vad_active:
            self._append_pre_roll(audio)
            return

        self._audio_buffer.extend(audio)
        self._utterance_has_audio = True

        while len(self._audio_buffer) >= self._chunk_bytes:
            chunk = bytes(self._audio_buffer[: self._chunk_bytes])
            del self._audio_buffer[: self._chunk_bytes]
            text, error = await self._infer(chunk, is_final=False)
            if error is not None:
                yield error
                return
            if text is None:  # An InterruptionFrame invalidated this inference.
                return
            if text:
                self._transcript_parts.append(text)
                yield InterimTranscriptionFrame(
                    "".join(self._transcript_parts),
                    self._user_id,
                    time_now_iso8601(),
                    self._frame_language,
                )

    async def _finalize_utterance(self) -> AsyncGenerator[Frame | None, None]:
        if not self._vad_active:
            return
        if not self._utterance_has_audio:
            self._reset_utterance()
            return

        # An empty final chunk is intentional when the utterance ended exactly
        # on a chunk boundary; FunASR still needs is_final=True to flush state.
        tail = bytes(self._audio_buffer)
        self._audio_buffer.clear()
        text, error = await self._infer(tail, is_final=True)
        if error is not None:
            yield error
            return
        if text is None:
            return
        if text:
            self._transcript_parts.append(text)

        transcript = "".join(self._transcript_parts).strip()
        user_id = self._user_id
        self._reset_utterance()
        if transcript:
            yield TranscriptionFrame(
                transcript,
                user_id,
                time_now_iso8601(),
                self._frame_language,
                finalized=True,
            )

    async def _activate_utterance(self) -> None:
        if self._vad_active:
            return

        pre_roll = bytes(self._pre_roll_buffer)
        self._generation += 1
        self._audio_buffer.clear()
        self._cache = {}
        self._transcript_parts.clear()
        self._utterance_has_audio = False
        self._pre_roll_buffer.clear()
        self._vad_active = True
        if pre_roll:
            await self.process_generator(self.run_stt(pre_roll))

    def _append_pre_roll(self, audio: bytes) -> None:
        if not self._pre_roll_bytes:
            self._pre_roll_buffer.clear()
            return
        self._pre_roll_buffer.extend(audio)
        overflow = len(self._pre_roll_buffer) - self._pre_roll_bytes
        if overflow > 0:
            del self._pre_roll_buffer[:overflow]

    async def _infer(
        self, audio: bytes, *, is_final: bool
    ) -> tuple[str | None, ErrorFrame | None]:
        generation = self._generation
        cache = self._cache
        samples = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0

        await self.start_processing_metrics()
        try:
            text = await asyncio.to_thread(
                self._provider.generate,
                samples,
                cache=cache,
                is_final=is_final,
                chunk_size=self._chunk_size,
                encoder_chunk_look_back=self._encoder_chunk_look_back,
                decoder_chunk_look_back=self._decoder_chunk_look_back,
            )
        except Exception as exc:
            logger.exception(f"{self} FunASR streaming inference failed")
            if generation == self._generation:
                self._reset_utterance()
            return None, ErrorFrame(f"FunASR streaming transcription error: {exc}")
        finally:
            await self.stop_processing_metrics()

        if generation != self._generation:
            return None, None
        return "" if text is None else str(text), None

    def _validate_audio_frame(self, frame: InputAudioRawFrame) -> str | None:
        if frame.sample_rate != SAMPLE_RATE or frame.num_channels != 1:
            return (
                "Paraformer streaming requires 16 kHz mono PCM; "
                f"received {frame.sample_rate} Hz/{frame.num_channels} channels"
            )
        if len(frame.audio) % 2:
            return "Paraformer streaming requires complete signed 16-bit PCM samples"
        return None

    def _reset_utterance(self):
        # Replace, rather than clear, the cache. An in-flight provider thread may
        # still own the old dict; its generation is rejected when it returns.
        self._generation += 1
        self._pre_roll_buffer.clear()
        self._audio_buffer.clear()
        self._cache = {}
        self._transcript_parts.clear()
        self._utterance_has_audio = False
        self._vad_active = False
