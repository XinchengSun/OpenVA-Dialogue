"""Pipecat v1.6 speech services shared by the current DyStream MSE runtime."""

from __future__ import annotations

import asyncio
import os
import time

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from .bridge import DyStreamAvatarProcessor, DyStreamUserAudioTap


DEFAULT_SYSTEM_INSTRUCTION = (
    "\u4f60\u662f\u4e00\u4e2a\u53cb\u597d\u3001\u7b80\u6d01\u7684\u4e2d\u6587\u8bed\u97f3\u52a9\u624b\u3002"
    "\u65e0\u8bba\u7528\u6237\u4f7f\u7528\u4ec0\u4e48\u8bed\u8a00\uff0c\u4f60\u90fd\u53ea\u7528\u4e2d\u6587\u56de\u7b54\u3002"
    "\u56de\u7b54\u8981\u5b8c\u6574\u3001\u81ea\u7136\u3001\u53e3\u8bed\u5316\uff1a"
    "\u7b80\u5355\u95ee\u9898\u7528\u4e00\u5230\u4e24\u53e5\uff0c\u590d\u6742\u95ee\u9898\u7528\u4e8c\u5230\u56db\u53e5\uff0c"
    "\u9664\u975e\u7528\u6237\u660e\u786e\u8981\u6c42\u8be6\u7ec6\uff0c\u4e0d\u8981\u8d85\u8fc7\u56db\u53e5\u3002"
    "\u4e0d\u4f7f\u7528 Markdown\u3001\u5217\u8868\u3001\u5e8f\u53f7\u6216\u7279\u6b8a\u7b26\u53f7\u3002"
)


def _required_env(name: str, *fallbacks: str) -> str:
    for key in (name, *fallbacks):
        value = os.getenv(key, "").strip()
        if value:
            return value
    names = ", ".join((name, *fallbacks))
    raise RuntimeError(f"missing required environment variable: {names}")


def _optional_bool_env(name: str) -> bool | None:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(
        f"{name} must be one of: true, false, 1, 0, yes, no, on, off"
    )


def _enabled_env(name: str, default: bool) -> bool:
    value = os.getenv(name, str(int(default))).strip().lower()
    return value not in {"0", "false", "no", "off"}


def create_user_aggregator_params() -> LLMUserAggregatorParams:
    """Use a short VAD timeout instead of the English-biased smart-turn default."""
    # Silero already waits roughly 200 ms before emitting speech-stopped. A
    # 250 ms strategy timeout cut ordinary 450-500 ms thinking pauses into new
    # turns and immediately cancelled the request that had just started.
    speech_timeout = float(os.getenv("PIPECAT_USER_SPEECH_TIMEOUT_SEC", "0.45"))
    stop_timeout = float(os.getenv("PIPECAT_USER_TURN_STOP_TIMEOUT_SEC", "2.0"))
    if speech_timeout < 0:
        raise RuntimeError("PIPECAT_USER_SPEECH_TIMEOUT_SEC must be non-negative")
    if stop_timeout <= 0:
        raise RuntimeError("PIPECAT_USER_TURN_STOP_TIMEOUT_SEC must be positive")
    return LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(),
        user_turn_strategies=UserTurnStrategies(
            stop=[
                SpeechTimeoutUserTurnStopStrategy(
                    user_speech_timeout=speech_timeout,
                )
            ],
        ),
        user_turn_stop_timeout=stop_timeout,
    )


async def _warm_llm_connection(llm, model: str, llm_extra: dict):
    """Warm DNS/TLS/API routing without adding anything to chat context."""
    if not _enabled_env("PIPECAT_LLM_WARMUP", True):
        return True
    started_at = time.monotonic()
    try:
        warmup_client = llm._client.with_options(
            max_retries=0,
            # The first request can include a multi-second model cold start.
            # Absorb it while the two GPU workers are loading, not on turn one.
            timeout=float(os.getenv("PIPECAT_LLM_WARMUP_TIMEOUT_SEC", "15.0")),
        )
        await warmup_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "\u4f60\u597d"}],
            max_tokens=1,
            stream=False,
            temperature=0,
            **llm_extra,
        )
        logger.info(
            f"[PIPELINE WARMUP] llm_model={model} "
            f"llm_ms={(time.monotonic() - started_at) * 1000.0:.1f}"
        )
        return True
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            f"[PIPELINE WARMUP] llm_failed_ms="
            f"{(time.monotonic() - started_at) * 1000.0:.1f} "
            f"model={model} error={type(exc).__name__}"
        )
        return False


async def _warm_tts_connection(tts):
    """Wait for the persistent realtime TTS WebSocket to be ready."""
    if not _enabled_env("PIPECAT_TTS_WARMUP", True):
        return True
    wait_ready = getattr(tts, "wait_ready", None)
    if wait_ready is None:
        return True
    started_at = time.monotonic()
    try:
        await wait_ready(
            timeout=float(os.getenv("PIPECAT_TTS_CONNECT_TIMEOUT_SEC", "6.0"))
        )
        logger.info(
            f"[PIPELINE WARMUP] tts_ready_ms="
            f"{(time.monotonic() - started_at) * 1000.0:.1f}"
        )
        return True
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            f"[PIPELINE WARMUP] tts_failed_ms="
            f"{(time.monotonic() - started_at) * 1000.0:.1f} error={type(exc).__name__}"
        )
        return False


async def _warm_services(llm, models: tuple[str, ...], llm_extra: dict, tts):
    results = await asyncio.gather(
        *(
            _warm_llm_connection(llm, model, llm_extra)
            for model in models
        ),
        _warm_tts_connection(tts),
    )
    llm_results = tuple(bool(result) for result in results[:-1])
    return {
        "llm_ready": bool(llm_results) and llm_results[0],
        "llm_models_ready": llm_results,
        "tts_warmup_ready": bool(results[-1]),
    }


def create_services():
    """Create local ASR, domestic API LLM, and persistent realtime TTS."""
    # Imports are intentionally lazy: bridge unit tests do not need provider
    # model packages or model downloads.
    from .funasr_compat import RawPCMFunASRSTTService
    from .qwen3_tts import Qwen3RealtimeTTSService
    from .resilient_llm import RecoveringOpenAILLMService

    llm_api_key = _required_env("PIPECAT_LLM_API_KEY", "OPENAI_API_KEY")
    llm_model = _required_env("PIPECAT_LLM_MODEL", "OPENAI_MODEL")
    llm_hedge_model = os.getenv(
        "PIPECAT_LLM_HEDGE_MODEL", "qwen-turbo"
    ).strip()
    llm_base_url = os.getenv("PIPECAT_LLM_BASE_URL", "").strip() or None
    llm_extra = {}
    enable_thinking = _optional_bool_env("PIPECAT_LLM_ENABLE_THINKING")
    if enable_thinking is not None:
        llm_extra["extra_body"] = {"enable_thinking": enable_thinking}

    stt = RawPCMFunASRSTTService(
        device=os.getenv("PIPECAT_ASR_DEVICE", "cpu"),
        settings=RawPCMFunASRSTTService.Settings(
            model=os.getenv("PIPECAT_ASR_MODEL", "iic/SenseVoiceSmall"),
            language=Language.ZH,
            use_itn=True,
        ),
    )

    llm = RecoveringOpenAILLMService(
        api_key=llm_api_key,
        base_url=llm_base_url,
        settings=RecoveringOpenAILLMService.Settings(
            model=llm_model,
            extra=llm_extra,
            system_instruction=os.getenv(
                "PIPECAT_SYSTEM_INSTRUCTION", DEFAULT_SYSTEM_INSTRUCTION
            ),
            temperature=float(os.getenv("PIPECAT_LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("PIPECAT_LLM_MAX_TOKENS", "256")),
        ),
        timeout_fallback_text=os.getenv(
            "PIPECAT_LLM_TIMEOUT_FALLBACK_TEXT",
            "刚才网络有点慢，请再说一遍。",
        ),
        history_max_messages=int(
            os.getenv("PIPECAT_LLM_HISTORY_MAX_MESSAGES", "12")
        ),
        hedge_model=llm_hedge_model,
        hedge_delay_sec=float(
            os.getenv("PIPECAT_LLM_HEDGE_DELAY_SEC", "1.0")
        ),
        hedge_deadline_sec=float(
            os.getenv("PIPECAT_LLM_TIMEOUT_SEC", "3.0")
        ),
    )
    # Voice turns are latency-sensitive: an SDK retry can silently double the
    # wait before the caller gets an error and is worse than failing this turn.
    llm._client = llm._client.with_options(
        max_retries=int(os.getenv("PIPECAT_LLM_MAX_RETRIES", "0")),
        timeout=float(os.getenv("PIPECAT_LLM_TIMEOUT_SEC", "3.0")),
    )
    llm._dystream_warmup_models = tuple(
        dict.fromkeys(
            model for model in (llm_model, llm_hedge_model) if model
        )
    )
    llm._dystream_warmup_extra = llm_extra

    tts = Qwen3RealtimeTTSService(
        api_key=_required_env(
            "PIPECAT_TTS_API_KEY",
            "PIPECAT_LLM_API_KEY",
            "OPENAI_API_KEY",
        ),
        model=os.getenv("PIPECAT_TTS_MODEL", "qwen3-tts-flash-realtime"),
        voice=os.getenv("PIPECAT_TTS_VOICE", "Cherry"),
        sample_rate=16_000,
        language_type=os.getenv("PIPECAT_TTS_LANGUAGE", "Chinese"),
    )
    return stt, llm, tts


async def run_bot(webrtc_connection, engine):
    """Run one full-duplex browser session against the shared DyStream engine."""
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16_000,
            audio_out_enabled=True,
            audio_out_sample_rate=16_000,
            audio_out_channels=1,
            audio_out_10ms_chunks=2,
            audio_out_auto_silence=True,
            video_out_enabled=True,
            # sync_with_audio images are released by Pipecat's audio queue;
            # cycling mode is required for that queue to reach the video sender.
            video_out_is_live=False,
            video_out_width=512,
            video_out_height=512,
            video_out_framerate=25,
        ),
    )

    stt, llm, tts = create_services()
    warmup_task = asyncio.create_task(
        _warm_services(
            llm,
            llm._dystream_warmup_models,
            llm._dystream_warmup_extra,
            tts,
        )
    )
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=create_user_aggregator_params(),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            DyStreamUserAudioTap(engine),
            stt,
            user_aggregator,
            llm,
            tts,
            DyStreamAvatarProcessor(
                engine,
                max_segments=int(os.getenv("PIPECAT_DYSTREAM_MAX_SEGMENTS", "8")),
            ),
            transport.output(),
            assistant_aggregator,
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16_000,
            audio_out_sample_rate=16_000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        logger.info("Pipecat WebRTC client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info("Pipecat WebRTC client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    try:
        await runner.run()
    finally:
        if not warmup_task.done():
            warmup_task.cancel()
        try:
            await warmup_task
        except asyncio.CancelledError:
            pass
