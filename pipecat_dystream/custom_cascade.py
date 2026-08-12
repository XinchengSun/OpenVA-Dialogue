"""Custom low-latency ASR -> LLM -> cloned-TTS components for MSE.

This module is imported only when ``PIPECAT_MSE_DIALOG_MODE=custom_cascade``.
Keeping the import lazy is intentional: the verified native-S2S rollback path
must not depend on FunASR, the local VoxCPM2 bridge, or custom credentials.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from .funasr_streaming import ParaformerStreamingSTTService
from .realtime_search import normalize_realtime_search_mode
from .resilient_llm import RecoveringOpenAILLMService
from .voxcpm2_tts import VoxCPM2LocalTTSService


DEFAULT_SYSTEM_INSTRUCTION = (
    "你是一个友好、自然的中文语音助手。无论用户使用什么语言，你都只用中文回答。"
    "用自然口语直接回答：简单问题一到两句，普通问题两到三句。"
    "除非用户明确要求详细，回答不要超过三句；不要复述问题，不要使用 Markdown、列表或序号。"
    "每次回答必须把最后一句说完整，不能为了缩短长度在半句话处截断。"
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
    raise RuntimeError(f"{name} must be true/false, 1/0, yes/no, or on/off")


def _json_object_env(name: str) -> dict[str, Any]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} must contain valid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must contain a JSON object")
    return value


def _llm_provider(model: str, base_url: str | None) -> str:
    configured = os.getenv("PIPECAT_LLM_PROVIDER", "").strip().lower()
    if configured:
        return configured
    normalized_model = model.strip().lower()
    normalized_url = (base_url or "").strip().lower()
    if normalized_model.startswith("deepseek-v4-") or "api.deepseek.com" in normalized_url:
        return "deepseek"
    if "dashscope.aliyuncs.com" in normalized_url:
        return "dashscope"
    return "openai_compatible"


def _llm_extra_body(model: str, base_url: str | None) -> dict[str, Any]:
    """Build provider-correct non-standard request fields.

    DashScope and DeepSeek use different thinking controls.  Keeping this
    translation at the provider boundary prevents a valid speed setting for
    one API from producing a 400 response or silently enabling reasoning on
    the other.
    """

    extra_body = _json_object_env("PIPECAT_LLM_EXTRA_BODY_JSON")
    enable_thinking = _optional_bool_env("PIPECAT_LLM_ENABLE_THINKING")
    if enable_thinking is None:
        return extra_body

    provider = _llm_provider(model, base_url)
    if provider == "deepseek":
        extra_body.pop("enable_thinking", None)
        extra_body["thinking"] = {
            "type": "enabled" if enable_thinking else "disabled"
        }
    elif provider == "dashscope":
        # Preserve the verified Qwen/DashScope behavior.
        extra_body.pop("thinking", None)
        extra_body["enable_thinking"] = enable_thinking
    else:
        raise RuntimeError(
            "PIPECAT_LLM_ENABLE_THINKING is provider-specific; unset it and "
            "use PIPECAT_LLM_EXTRA_BODY_JSON for this OpenAI-compatible provider"
        )
    return extra_body


def _realtime_search_mode(model: str, base_url: str | None) -> str:
    try:
        mode = normalize_realtime_search_mode(
            os.getenv("PIPECAT_REALTIME_SEARCH_MODE", "off")
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if mode != "off" and _llm_provider(model, base_url) != "dashscope":
        raise RuntimeError(
            "PIPECAT_REALTIME_SEARCH_MODE currently requires the DashScope provider"
        )
    return mode


def _llm_warmup_models(
    llm_model: str,
    realtime_search_mode: str,
    realtime_search_model: str,
) -> tuple[str, ...]:
    """Warm every model that can serve the first user turn, once."""

    models = [llm_model.strip()]
    search_model = realtime_search_model.strip()
    if (
        realtime_search_mode != "off"
        and search_model
        and search_model not in models
    ):
        models.append(search_model)
    return tuple(models)


def _chunk_size_env() -> tuple[int, int, int]:
    raw = os.getenv("PIPECAT_ASR_CHUNK_SIZE", "0,10,5")
    try:
        values = tuple(int(part.strip()) for part in raw.split(","))
    except ValueError as exc:
        raise RuntimeError("PIPECAT_ASR_CHUNK_SIZE must contain three integers") from exc
    if len(values) != 3 or values[1] <= 0:
        raise RuntimeError("PIPECAT_ASR_CHUNK_SIZE must look like 0,10,5")
    return values


def _vad_params() -> VADParams:
    try:
        params = VADParams(
            confidence=float(os.getenv("PIPECAT_BARGE_IN_CONFIDENCE", "0.65")),
            start_secs=float(os.getenv("PIPECAT_BARGE_IN_START_SEC", "0.12")),
            stop_secs=float(os.getenv("PIPECAT_BARGE_IN_STOP_SEC", "0.20")),
            min_volume=float(os.getenv("PIPECAT_BARGE_IN_MIN_VOLUME", "0.50")),
        )
    except ValueError as exc:
        raise RuntimeError("custom cascade VAD parameters must be numbers") from exc
    if not 0 <= params.confidence <= 1 or not 0 <= params.min_volume <= 1:
        raise RuntimeError("custom cascade VAD confidence and min volume must be 0..1")
    if params.start_secs < 0 or params.stop_secs < 0:
        raise RuntimeError("custom cascade VAD start/stop seconds must be non-negative")
    return params


def _create_user_aggregator_params(
    vad_params: VADParams | None = None,
) -> LLMUserAggregatorParams:
    """Let Silero own endpoints and Pipecat's default VAD start own barge-in.

    ``UserTurnStrategies.start`` is intentionally omitted: Pipecat 1.6 then
    installs ``VADUserTurnStartStrategy(enable_interruptions=True)``.  Do not
    add a second VAD-to-interruption processor downstream or one speech start
    will cancel the new turn twice.
    """

    speech_timeout = float(os.getenv("PIPECAT_USER_SPEECH_TIMEOUT_SEC", "0.45"))
    stop_timeout = float(os.getenv("PIPECAT_USER_TURN_STOP_TIMEOUT_SEC", "2.0"))
    if speech_timeout < 0 or stop_timeout <= 0:
        raise RuntimeError("custom cascade turn timeouts must be positive")
    return LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(params=vad_params or _vad_params()),
        user_turn_strategies=UserTurnStrategies(
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=speech_timeout)]
        ),
        user_turn_stop_timeout=stop_timeout,
    )


async def _warm_llm(llm: Any, model: str, extra: dict[str, Any]) -> None:
    """Warm DNS, TLS and provider routing without mutating chat history."""

    timeout = float(os.getenv("PIPECAT_LLM_WARMUP_TIMEOUT_SEC", "15.0"))
    started_at = time.monotonic()
    client = llm._client.with_options(max_retries=0, timeout=timeout)
    await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "你好"}],
        max_tokens=1,
        stream=False,
        temperature=0,
        **extra,
    )
    logger.info(
        "[CUSTOM CASCADE] LLM warmup ready "
        f"model={model} elapsed_ms={(time.monotonic() - started_at) * 1000.0:.1f}"
    )


@dataclass
class CustomCascadeComponents:
    stt: ParaformerStreamingSTTService
    llm: RecoveringOpenAILLMService
    tts: VoxCPM2LocalTTSService
    user_aggregator: Any
    assistant_aggregator: Any
    llm_model: str
    llm_warmup_models: tuple[str, ...]
    llm_extra: dict[str, Any]
    log: Callable[[str], None]
    _llm_ready: bool = False

    async def wait_ready(self, timeout: float) -> None:
        """Block MSE readiness until both local TTS and remote LLM are warm."""

        started_at = time.monotonic()
        await self.tts.wait_ready(timeout=timeout)
        tts_warmup_text = os.getenv("PIPECAT_TTS_WARMUP_TEXT", "你好。").strip()
        if tts_warmup_text:
            remaining = max(0.001, timeout - (time.monotonic() - started_at))
            await self.tts.warmup(tts_warmup_text, timeout=remaining)
        for model in self.llm_warmup_models:
            remaining = max(0.001, timeout - (time.monotonic() - started_at))
            await asyncio.wait_for(
                _warm_llm(self.llm, model, self.llm_extra),
                timeout=remaining,
            )
        self._llm_ready = True
        self.log("[CUSTOM CASCADE] ASR, LLM and VoxCPM2 bridge ready")

    @property
    def ready(self) -> bool:
        return self._llm_ready and bool(getattr(self.tts, "ready", False))

    def health_snapshot(self) -> dict[str, Any]:
        tts_health = getattr(self.tts, "health_snapshot", None)
        return {
            "ready": self.ready,
            "asr": {
                "ready": True,
                "model": os.getenv("PIPECAT_ASR_MODEL", "paraformer-zh-streaming"),
                "device": os.getenv("PIPECAT_ASR_DEVICE", "cpu"),
                "hub": os.getenv("PIPECAT_ASR_HUB", "ms"),
                "chunk_size": list(_chunk_size_env()),
                "pre_roll_secs": self.stt.pre_roll_secs,
            },
            "llm": {
                "ready": self._llm_ready,
                "model": self.llm_model,
                "provider": _llm_provider(
                    self.llm_model,
                    os.getenv("PIPECAT_LLM_BASE_URL", "").strip() or None,
                ),
                "thinking": _optional_bool_env("PIPECAT_LLM_ENABLE_THINKING"),
                "warmup_models": list(self.llm_warmup_models),
                "max_reply_sentences": self.llm.reply_max_sentences,
                "realtime_search": self.llm.realtime_search_snapshot(),
            },
            "tts": (
                tts_health()
                if callable(tts_health)
                else {"ready": bool(getattr(self.tts, "ready", False))}
            ),
        }


def create_custom_cascade_components(
    log: Callable[[str], None],
) -> CustomCascadeComponents:
    """Construct the isolated custom cascade without touching native S2S."""

    llm_api_key = _required_env("PIPECAT_LLM_API_KEY", "OPENAI_API_KEY")
    llm_model = _required_env("PIPECAT_LLM_MODEL", "OPENAI_MODEL")
    llm_base_url = os.getenv("PIPECAT_LLM_BASE_URL", "").strip() or None
    realtime_search_mode = _realtime_search_mode(llm_model, llm_base_url)
    realtime_search_model = os.getenv("PIPECAT_REALTIME_SEARCH_MODEL", "").strip()
    llm_warmup_models = _llm_warmup_models(
        llm_model,
        realtime_search_mode,
        realtime_search_model,
    )
    llm_extra: dict[str, Any] = {}
    extra_body = _llm_extra_body(llm_model, llm_base_url)
    if extra_body:
        llm_extra["extra_body"] = extra_body

    vad_params = _vad_params()
    try:
        requested_pre_roll = float(os.getenv("PIPECAT_ASR_PRE_ROLL_SEC", "0.30"))
    except ValueError as exc:
        raise RuntimeError("PIPECAT_ASR_PRE_ROLL_SEC must be a number") from exc
    if requested_pre_roll < 0:
        raise RuntimeError("PIPECAT_ASR_PRE_ROLL_SEC must be non-negative")

    stt = ParaformerStreamingSTTService(
        model=os.getenv("PIPECAT_ASR_MODEL", "paraformer-zh-streaming"),
        device=os.getenv("PIPECAT_ASR_DEVICE", "cpu"),
        hub=os.getenv("PIPECAT_ASR_HUB", "ms"),
        chunk_size=_chunk_size_env(),
        encoder_chunk_look_back=int(
            os.getenv("PIPECAT_ASR_ENCODER_CHUNK_LOOK_BACK", "4")
        ),
        decoder_chunk_look_back=int(
            os.getenv("PIPECAT_ASR_DECODER_CHUNK_LOOK_BACK", "1")
        ),
        pre_roll_secs=max(requested_pre_roll, vad_params.start_secs),
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
            "PIPECAT_LLM_TIMEOUT_FALLBACK_TEXT", "刚才网络有点慢，请再说一遍。"
        ),
        history_max_messages=int(os.getenv("PIPECAT_LLM_HISTORY_MAX_MESSAGES", "12")),
        reply_max_sentences=int(os.getenv("PIPECAT_LLM_MAX_SENTENCES", "3")),
        hedge_model=os.getenv("PIPECAT_LLM_HEDGE_MODEL", "").strip(),
        hedge_delay_sec=float(os.getenv("PIPECAT_LLM_HEDGE_DELAY_SEC", "0.75")),
        hedge_deadline_sec=float(os.getenv("PIPECAT_LLM_TIMEOUT_SEC", "8.0")),
        realtime_search_mode=realtime_search_mode,
        realtime_search_strategy=os.getenv(
            "PIPECAT_REALTIME_SEARCH_STRATEGY", "turbo"
        ),
        realtime_search_model=realtime_search_model,
    )
    llm._client = llm._client.with_options(
        max_retries=int(os.getenv("PIPECAT_LLM_MAX_RETRIES", "0")),
        timeout=float(os.getenv("PIPECAT_LLM_TIMEOUT_SEC", "8.0")),
    )

    tts = VoxCPM2LocalTTSService(
        uri=os.getenv("VOXCPM2_BRIDGE_URI", "ws://127.0.0.1:8770")
    )
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=_create_user_aggregator_params(vad_params),
    )
    return CustomCascadeComponents(
        stt=stt,
        llm=llm,
        tts=tts,
        user_aggregator=aggregators.user(),
        assistant_aggregator=aggregators.assistant(),
        llm_model=llm_model,
        llm_warmup_models=llm_warmup_models,
        llm_extra=llm_extra,
        log=log,
    )
