from __future__ import annotations

import json
import asyncio
import unittest
import tempfile
from pathlib import Path

import httpx

from voice_service.openai_speech_server import (
    OpenAISpeechPCMBackend,
    _qwen_generation_budget,
)


class _ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


class _BlockingStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = asyncio.Event()

    async def __aiter__(self):
        yield b"\x01\x00"
        await self.closed.wait()

    async def aclose(self) -> None:
        self.closed.set()


class OpenAISpeechPCMBackendTests(unittest.IsolatedAsyncioTestCase):
    def _backend(
        self,
        handler,
        **overrides,
    ) -> tuple[OpenAISpeechPCMBackend, httpx.AsyncClient]:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        def factory(**_kwargs):
            return client

        kwargs = {
            "base_url": "http://speech.test",
            "model": "fishaudio/s2-pro",
            "sample_rate": 44_100,
            "reference_audio": "https://example.test/ref.wav",
            "reference_text": "准确的参考音频文本。",
            "client_factory": factory,
        }
        kwargs.update(overrides)
        return OpenAISpeechPCMBackend(**kwargs), client

    async def test_streams_aligned_pcm_and_builds_fish_reference_payload(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.method == "GET":
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(
                200,
                headers={
                    "content-type": "audio/pcm",
                    "x-sample-rate": "44100",
                },
                stream=_ChunkStream([b"\x01", b"\x00\x02\x00"]),
            )

        backend, _client = self._backend(
            handler,
            extra_body={"initial_codec_chunk_frames": 2},
        )
        await backend.start()
        chunks = [
            chunk
            async for chunk in backend.generate_pcm16("你好。", "ctx-1")
        ]
        await backend.stop()

        self.assertEqual(b"".join(chunks), b"\x01\x00\x02\x00")
        payload = json.loads(requests[-1].content)
        self.assertEqual(payload["model"], "fishaudio/s2-pro")
        self.assertEqual(payload["input"], "你好。")
        self.assertEqual(payload["response_format"], "pcm")
        self.assertIs(payload["stream"], True)
        self.assertEqual(payload["initial_codec_chunk_frames"], 2)
        self.assertEqual(
            payload["references"],
            [
                {
                    "audio_path": "https://example.test/ref.wav",
                    "text": "准确的参考音频文本。",
                }
            ],
        )

    async def test_rejects_sample_rate_mismatch_before_yielding_audio(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200)
            return httpx.Response(
                200,
                headers={"content-type": "audio/pcm", "x-sample-rate": "24000"},
                stream=_ChunkStream([b"\x01\x00"]),
            )

        backend, _client = self._backend(handler)
        await backend.start()
        with self.assertRaisesRegex(RuntimeError, "sample rate changed"):
            async for _ in backend.generate_pcm16("测试。", "ctx-rate"):
                pass
        await backend.stop()

    async def test_rejects_odd_final_pcm_byte(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200)
            return httpx.Response(
                200,
                headers={"content-type": "audio/pcm", "x-sample-rate": "44100"},
                stream=_ChunkStream([b"\x01\x00\x02"]),
            )

        backend, _client = self._backend(handler)
        await backend.start()
        chunks: list[bytes] = []
        with self.assertRaisesRegex(RuntimeError, "incomplete PCM16 sample"):
            async for chunk in backend.generate_pcm16("测试。", "ctx-odd"):
                chunks.append(chunk)
        self.assertEqual(chunks, [b"\x01\x00"])
        await backend.stop()

    async def test_reports_http_error_without_leaking_unbounded_body(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200)
            return httpx.Response(
                503,
                headers={"content-type": "application/json"},
                content=b'{"error":"not ready"}',
            )

        backend, _client = self._backend(handler)
        await backend.start()
        with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
            async for _ in backend.generate_pcm16("测试。", "ctx-error"):
                pass
        await backend.stop()

    async def test_vllm_payload_selects_raw_audio_stream_and_clone_fields(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.method == "GET":
                return httpx.Response(200)
            return httpx.Response(
                200,
                headers={"content-type": "audio/pcm", "x-sample-rate": "44100"},
                stream=_ChunkStream([b"\x01\x00"]),
            )

        backend, _client = self._backend(handler, provider="vllm")
        await backend.start()
        chunks = [chunk async for chunk in backend.generate_pcm16("测试。", "ctx-v")]
        await backend.stop()

        self.assertEqual(chunks, [b"\x01\x00"])
        payload = json.loads(requests[-1].content)
        self.assertEqual(payload["stream_format"], "audio")
        self.assertEqual(payload["ref_audio"], "https://example.test/ref.wav")
        self.assertEqual(payload["ref_text"], "准确的参考音频文本。")
        self.assertNotIn("references", payload)

    async def test_qwen_clone_payload_adds_explicit_target_language(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.method == "GET":
                return httpx.Response(200)
            return httpx.Response(
                200,
                headers={"content-type": "audio/pcm", "x-sample-rate": "44100"},
                stream=_ChunkStream([b"\x01\x00"]),
            )

        backend, _client = self._backend(
            handler,
            backend="qwen3_tts_1_7b_base",
            target_language="zh-CN",
            extra_body={"task_type": "CustomVoice"},
        )
        await backend.start()
        chunks = [chunk async for chunk in backend.generate_pcm16("测试。", "ctx-qwen")]
        await backend.stop()

        self.assertEqual(chunks, [b"\x01\x00"])
        payload = json.loads(requests[-1].content)
        self.assertEqual(payload["language"], "Chinese")
        self.assertEqual(payload["task_type"], "Base")
        self.assertEqual(payload["references"][0]["text"], "准确的参考音频文本。")

    async def test_qwen_cross_language_uses_xvector_and_dynamic_budget(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.method == "GET":
                return httpx.Response(200)
            return httpx.Response(
                200,
                headers={"content-type": "audio/pcm", "x-sample-rate": "24000"},
                stream=_ChunkStream([b"\x01\x00"]),
            )

        backend, _client = self._backend(
            handler,
            backend="qwen3_tts_1_7b_base",
            sample_rate=24_000,
            reference_language="en-US",
            target_language="zh-CN",
            extra_body={
                "max_new_tokens": 320,
                "x_vector_only_mode": False,
            },
        )
        text = "\u6c49" * 48 + "\u3002"
        await backend.start()
        chunks = [chunk async for chunk in backend.generate_pcm16(text, "ctx-xvec")]
        await backend.stop()

        self.assertEqual(chunks, [b"\x01\x00"])
        payload = json.loads(requests[-1].content)
        self.assertIs(payload["x_vector_only_mode"], True)
        self.assertGreater(payload["max_new_tokens"], 96)
        self.assertLessEqual(payload["max_new_tokens"], 320)

    async def test_qwen_same_language_keeps_icl_mode(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.method == "GET":
                return httpx.Response(200)
            return httpx.Response(
                200,
                headers={"content-type": "audio/pcm", "x-sample-rate": "24000"},
                stream=_ChunkStream([b"\x01\x00"]),
            )

        backend, _client = self._backend(
            handler,
            backend="qwen3_tts_1_7b_base",
            sample_rate=24_000,
            reference_language="zh-CN",
            target_language="zh-CN",
            extra_body={"max_new_tokens": 320},
        )
        await backend.start()
        chunks = [
            chunk
            async for chunk in backend.generate_pcm16("\u4f60\u597d\u3002", "ctx-icl")
        ]
        await backend.stop()

        self.assertEqual(chunks, [b"\x01\x00"])
        payload = json.loads(requests[-1].content)
        self.assertNotIn("x_vector_only_mode", payload)

    async def test_qwen_auto_reference_language_prefers_xvector_mode(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.method == "GET":
                return httpx.Response(200)
            return httpx.Response(
                200,
                headers={"content-type": "audio/pcm", "x-sample-rate": "24000"},
                stream=_ChunkStream([b"\x01\x00"]),
            )

        backend, _client = self._backend(
            handler,
            backend="qwen3_tts_1_7b_base",
            sample_rate=24_000,
            reference_language="auto",
            target_language="zh-CN",
            extra_body={"max_new_tokens": 320},
        )
        await backend.start()
        chunks = [
            chunk
            async for chunk in backend.generate_pcm16("\u4f60\u597d\u3002", "ctx-auto")
        ]
        await backend.stop()

        self.assertEqual(chunks, [b"\x01\x00"])
        self.assertIs(json.loads(requests[-1].content)["x_vector_only_mode"], True)

    def test_qwen_generation_budget_scales_and_respects_hard_cap(self):
        short = _qwen_generation_budget("\u4f60\u597d\u3002", hard_cap=320)
        medium = _qwen_generation_budget("\u6c49" * 48 + "\u3002", hard_cap=320)
        long_number = _qwen_generation_budget("13800138000", hard_cap=320)
        capped = _qwen_generation_budget("\u6c49" * 200 + "\u3002", hard_cap=320)

        self.assertGreaterEqual(short, 48)
        self.assertLess(short, 96)
        self.assertGreater(medium, 96)
        self.assertLess(medium, 320)
        self.assertGreater(long_number, 48)
        self.assertEqual(capped, 320)

    def test_qwen_rejects_an_unsafe_static_hard_cap(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200)

        with self.assertRaisesRegex(ValueError, "hard cap must be between"):
            self._backend(
                handler,
                backend="qwen3_tts_1_7b_base",
                extra_body={"max_new_tokens": 96},
            )

    async def test_qwen_logs_when_pcm_duration_reaches_its_dynamic_budget(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200)
            return httpx.Response(
                200,
                headers={"content-type": "audio/pcm", "x-sample-rate": "24"},
                stream=_ChunkStream([b"\x01\x00" * 96]),
            )

        backend, _client = self._backend(
            handler,
            backend="qwen3_tts_1_7b_base",
            sample_rate=24,
            reference_language="en-US",
            target_language="zh-CN",
            extra_body={"max_new_tokens": 192},
        )
        await backend.start()
        with self.assertLogs(
            "voice_service.openai_speech_server", level="WARNING"
        ) as captured:
            chunks = [
                chunk async for chunk in backend.generate_pcm16("Hi.", "ctx-length")
            ]
        await backend.stop()

        self.assertEqual(chunks, [b"\x01\x00" * 96])
        self.assertIn("probable_length_stop=True", "\n".join(captured.output))

    async def test_fish_payload_does_not_gain_an_unsupported_language_field(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.method == "GET":
                return httpx.Response(200)
            return httpx.Response(
                200,
                headers={"content-type": "audio/pcm", "x-sample-rate": "44100"},
                stream=_ChunkStream([b"\x01\x00"]),
            )

        backend, _client = self._backend(
            handler,
            backend="fish_s2_pro",
            target_language="zh-CN",
        )
        await backend.start()
        chunks = [chunk async for chunk in backend.generate_pcm16("测试。", "ctx-fish")]
        await backend.stop()

        self.assertEqual(chunks, [b"\x01\x00"])
        self.assertNotIn("language", json.loads(requests[-1].content))

    async def test_vllm_local_reference_is_sent_as_file_uri(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.method == "GET":
                return httpx.Response(200)
            return httpx.Response(
                200,
                headers={"content-type": "audio/pcm", "x-sample-rate": "44100"},
                stream=_ChunkStream([b"\x01\x00"]),
            )

        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.wav"
            reference.write_bytes(b"RIFFtest")
            backend, _client = self._backend(
                handler,
                provider="vllm",
                reference_audio=str(reference),
            )
            await backend.start()
            chunks = [
                chunk async for chunk in backend.generate_pcm16("测试。", "ctx-file")
            ]
            await backend.stop()

        self.assertEqual(chunks, [b"\x01\x00"])
        payload = json.loads(requests[-1].content)
        self.assertTrue(payload["ref_audio"].startswith("file://"))

    async def test_rejects_wav_container_even_when_http_succeeds(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200)
            return httpx.Response(
                200,
                headers={"content-type": "audio/wav"},
                stream=_ChunkStream([b"RIFFbad"]),
            )

        backend, _client = self._backend(handler)
        await backend.start()
        with self.assertRaisesRegex(RuntimeError, "non-PCM content"):
            async for _ in backend.generate_pcm16("测试。", "ctx-wav"):
                pass
        await backend.stop()

    async def test_rejects_big_endian_audio_l16(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200)
            return httpx.Response(
                200,
                headers={"content-type": "audio/L16"},
                stream=_ChunkStream([b"\x00\x01"]),
            )

        backend, _client = self._backend(handler)
        await backend.start()
        with self.assertRaisesRegex(RuntimeError, "non-PCM content"):
            async for _ in backend.generate_pcm16("测试。", "ctx-l16"):
                pass
        await backend.stop()

    async def test_closing_generator_closes_active_http_stream(self):
        stream = _BlockingStream()

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200)
            return httpx.Response(
                200,
                headers={"content-type": "audio/pcm", "x-sample-rate": "44100"},
                stream=stream,
            )

        backend, _client = self._backend(handler)
        await backend.start()
        generator = backend.generate_pcm16("测试。", "ctx-cancel")
        self.assertEqual(await anext(generator), b"\x01\x00")
        await generator.aclose()

        self.assertTrue(stream.closed.is_set())
        self.assertNotIn("ctx-cancel", backend._active_responses)
        await backend.stop()

    async def test_health_rechecks_upstream_after_startup(self):
        health_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal health_calls
            if request.method == "GET":
                health_calls += 1
                return httpx.Response(200 if health_calls == 1 else 503)
            raise AssertionError("unexpected synthesis request")

        backend, _client = self._backend(handler)
        await backend.start()
        with self.assertRaises(httpx.HTTPStatusError):
            await backend.health_check()
        self.assertEqual(health_calls, 2)
        await backend.stop()

    def test_reference_audio_and_transcript_are_atomic(self):
        with self.assertRaisesRegex(ValueError, "must be set together"):
            OpenAISpeechPCMBackend(
                base_url="http://speech.test",
                model="fishaudio/s2-pro",
                reference_audio="ref.wav",
            )


if __name__ == "__main__":
    unittest.main()
