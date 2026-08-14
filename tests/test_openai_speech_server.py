from __future__ import annotations

import json
import asyncio
import unittest
import tempfile
from pathlib import Path

import httpx

from voice_service.openai_speech_server import OpenAISpeechPCMBackend


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
