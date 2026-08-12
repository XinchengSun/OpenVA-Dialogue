"""Candidate OpenAI-compatible streaming speech bridge.

This module intentionally runs beside the production VoxCPM2 bridge.  It
adapts raw PCM streaming from engines such as SGLang-Omni Fish Speech S2 Pro
to the already verified local WebSocket contract used by Pipecat.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from voice_service.voxcpm2_server import VoxCPM2WebSocketBridge, _run_server


logger = logging.getLogger(__name__)


def _json_object_env(name: str) -> dict[str, Any]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _is_local_path(value: str) -> bool:
    if len(value) >= 3 and value[1] == ":" and value[2] in {"/", "\\"}:
        return True
    parsed = urlparse(value)
    return not parsed.scheme and not value.startswith("data:")


class OpenAISpeechPCMBackend:
    """Stream mono PCM16 from an OpenAI-compatible ``/audio/speech`` API."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        provider: str = "sglang",
        voice: str = "default",
        api_key: str = "",
        endpoint: str = "/v1/audio/speech",
        health_path: str = "/health",
        sample_rate: int = 44_100,
        reference_audio: str = "",
        reference_text: str = "",
        request_timeout_sec: float = 120.0,
        connect_timeout_sec: float = 10.0,
        extra_body: dict[str, Any] | None = None,
        client_factory: Callable[..., httpx.AsyncClient] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model.strip()
        self._provider = provider.strip().lower()
        self._voice = voice.strip() or "default"
        self._api_key = api_key.strip()
        self._endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self._health_path = (
            health_path if health_path.startswith("/") else f"/{health_path}"
        )
        self.sample_rate = int(sample_rate)
        self._reference_audio = reference_audio.strip()
        self._reference_text = reference_text.strip()
        self._request_timeout_sec = float(request_timeout_sec)
        self._connect_timeout_sec = float(connect_timeout_sec)
        self._extra_body = dict(extra_body or {})
        self._client_factory = client_factory or httpx.AsyncClient
        self._client: httpx.AsyncClient | None = None
        self._active_responses: dict[str, set[httpx.Response]] = defaultdict(set)

        if not self._base_url:
            raise ValueError("OpenAI speech base_url must not be empty")
        if not self._model:
            raise ValueError("OpenAI speech model must not be empty")
        if self._provider not in {"sglang", "vllm", "openai_compatible"}:
            raise ValueError(
                "OpenAI speech provider must be sglang, vllm, or openai_compatible"
            )
        if self.sample_rate <= 0:
            raise ValueError("OpenAI speech sample_rate must be positive")
        if self._request_timeout_sec <= 0 or self._connect_timeout_sec <= 0:
            raise ValueError("OpenAI speech timeouts must be positive")
        if bool(self._reference_audio) != bool(self._reference_text):
            raise ValueError(
                "reference_audio and its exact reference_text must be set together"
            )

    @property
    def model(self) -> str:
        return self._model

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    async def start(self) -> None:
        if self._client is not None:
            return
        if self._reference_audio and _is_local_path(self._reference_audio):
            reference_path = Path(self._reference_audio).expanduser()
            if not reference_path.is_file():
                raise FileNotFoundError(
                    f"OpenAI speech reference audio not found: {reference_path}"
                )

        headers = {"Accept": "audio/pcm"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        timeout = httpx.Timeout(
            self._request_timeout_sec,
            connect=self._connect_timeout_sec,
        )
        client = self._client_factory(headers=headers, timeout=timeout)
        self._client = client
        try:
            await self.health_check()
        except BaseException:
            self._client = None
            await client.aclose()
            raise
        logger.info(
            "OpenAI speech backend ready: model=%s sample_rate=%d endpoint=%s",
            self._model,
            self.sample_rate,
            self._url(self._endpoint),
        )

    async def health_check(self) -> None:
        client = self._client
        if client is None:
            raise RuntimeError("OpenAI speech backend is not started")
        response = await client.get(self._url(self._health_path))
        response.raise_for_status()

    def _payload(self, text: str) -> dict[str, Any]:
        payload = dict(self._extra_body)
        payload.update(
            {
                "model": self._model,
                "voice": self._voice,
                "input": text,
                "response_format": "pcm",
                "stream": True,
            }
        )
        if self._reference_audio:
            if self._provider == "vllm":
                reference_audio = self._reference_audio
                if _is_local_path(reference_audio):
                    # Raw vLLM HTTP requests do not apply the SDK's local-path
                    # conversion. The server must also be launched with an
                    # --allowed-local-media-path that contains this file.
                    reference_audio = str(
                        Path(reference_audio).expanduser().resolve().as_uri()
                    )
                payload["ref_audio"] = reference_audio
                payload["ref_text"] = self._reference_text
                payload["stream_format"] = "audio"
            else:
                payload["references"] = [
                    {
                        "audio_path": self._reference_audio,
                        "text": self._reference_text,
                    }
                ]
        elif self._provider == "vllm":
            payload["stream_format"] = "audio"
        return payload

    @staticmethod
    async def _error_excerpt(response: httpx.Response, limit: int = 2048) -> str:
        body = bytearray()
        async for chunk in response.aiter_bytes():
            if not chunk:
                continue
            remaining = limit - len(body)
            if remaining <= 0:
                break
            body.extend(chunk[:remaining])
            if len(body) >= limit:
                break
        return bytes(body).decode("utf-8", errors="replace")

    async def generate_pcm16(
        self,
        text: str,
        context_id: str = "",
    ) -> AsyncIterator[bytes]:
        client = self._client
        if client is None:
            raise RuntimeError("OpenAI speech backend is not started")
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("speech input must not be empty")

        started_at = time.perf_counter()
        first_pcm_at: float | None = None
        pending = b""
        async with client.stream(
            "POST",
            self._url(self._endpoint),
            json=self._payload(clean_text),
        ) as response:
            if response.is_error:
                detail = await self._error_excerpt(response)
                raise RuntimeError(
                    f"speech API returned HTTP {response.status_code}: {detail}"
                )
            content_type = response.headers.get("content-type", "").lower()
            media_type = content_type.partition(";")[0].strip()
            if media_type not in {
                "audio/pcm",
                "application/octet-stream",
                "binary/octet-stream",
            }:
                detail = await self._error_excerpt(response)
                raise RuntimeError(
                    f"speech API returned non-PCM content {content_type!r}: {detail}"
                )
            header_rate = int(response.headers.get("x-sample-rate", self.sample_rate))
            if header_rate != self.sample_rate:
                raise RuntimeError(
                    "speech API sample rate changed after bridge readiness: "
                    f"configured={self.sample_rate} response={header_rate}"
                )

            self._active_responses[context_id].add(response)
            try:
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    data = pending + chunk
                    aligned = len(data) - (len(data) % 2)
                    if aligned:
                        if first_pcm_at is None:
                            first_pcm_at = time.perf_counter()
                            logger.info(
                                "[OPENAI SPEECH] first_pcm_ms=%.1f model=%s",
                                (first_pcm_at - started_at) * 1000.0,
                                self._model,
                            )
                        yield data[:aligned]
                    pending = data[aligned:]
            finally:
                responses = self._active_responses.get(context_id)
                if responses is not None:
                    responses.discard(response)
                    if not responses:
                        self._active_responses.pop(context_id, None)

        if pending:
            raise RuntimeError("speech API ended with an incomplete PCM16 sample")
        if first_pcm_at is None:
            raise RuntimeError("speech API completed without PCM audio")

    async def release_context(self, context_id: str) -> None:
        responses = list(self._active_responses.pop(context_id, ()))
        if responses:
            await asyncio.gather(
                *(response.aclose() for response in responses),
                return_exceptions=True,
            )

    async def stop(self) -> None:
        responses = [
            response
            for active in self._active_responses.values()
            for response in active
        ]
        self._active_responses.clear()
        if responses:
            await asyncio.gather(
                *(response.aclose() for response in responses),
                return_exceptions=True,
            )
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()


def _backend_from_env() -> OpenAISpeechPCMBackend:
    return OpenAISpeechPCMBackend(
        base_url=os.environ["OPENAI_SPEECH_BASE_URL"],
        model=os.environ["OPENAI_SPEECH_MODEL"],
        provider=os.getenv("OPENAI_SPEECH_PROVIDER", "sglang"),
        voice=os.getenv("OPENAI_SPEECH_VOICE", "default"),
        api_key=os.getenv("OPENAI_SPEECH_API_KEY", ""),
        endpoint=os.getenv("OPENAI_SPEECH_ENDPOINT", "/v1/audio/speech"),
        health_path=os.getenv("OPENAI_SPEECH_HEALTH_PATH", "/health"),
        sample_rate=int(os.getenv("OPENAI_SPEECH_SAMPLE_RATE", "44100")),
        reference_audio=os.getenv("OPENAI_SPEECH_REFERENCE_AUDIO", ""),
        reference_text=os.getenv("OPENAI_SPEECH_REFERENCE_TEXT", ""),
        request_timeout_sec=float(
            os.getenv("OPENAI_SPEECH_REQUEST_TIMEOUT_SEC", "120.0")
        ),
        connect_timeout_sec=float(
            os.getenv("OPENAI_SPEECH_CONNECT_TIMEOUT_SEC", "10.0")
        ),
        extra_body=_json_object_env("OPENAI_SPEECH_EXTRA_BODY_JSON"),
    )


def main() -> None:
    logging.basicConfig(
        level=os.getenv("OPENAI_SPEECH_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Candidate OpenAI-compatible PCM bridge"
    )
    parser.add_argument(
        "--host", default=os.getenv("OPENAI_SPEECH_BRIDGE_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("OPENAI_SPEECH_BRIDGE_PORT", "8771")),
    )
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("OpenAI speech bridge must bind to localhost")

    backend = _backend_from_env()
    # Reuse the production-tested wire protocol and signal-safe runner.  The
    # candidate stays isolated by its module name, process, port and env file.
    candidate_warmup = os.getenv("OPENAI_SPEECH_WARMUP_TEXT", "").strip()
    if candidate_warmup and not os.getenv("VOXCPM2_WARMUP_TEXT", "").strip():
        os.environ["VOXCPM2_WARMUP_TEXT"] = candidate_warmup
    asyncio.run(_run_server(args.host, args.port, _backend=backend))


if __name__ == "__main__":
    main()
