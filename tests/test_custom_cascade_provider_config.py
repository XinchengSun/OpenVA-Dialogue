from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, call, patch

from pipecat_dystream.custom_cascade import (
    CustomCascadeComponents,
    _llm_extra_body,
    _llm_provider,
    _llm_warmup_models,
    _realtime_search_mode,
)


class CustomCascadeProviderConfigTests(unittest.TestCase):
    def test_deepseek_v4_uses_official_non_thinking_schema(self):
        env = {
            "PIPECAT_LLM_ENABLE_THINKING": "false",
            "PIPECAT_LLM_EXTRA_BODY_JSON": '{"user_id":"latency-ab"}',
        }
        with patch.dict(os.environ, env, clear=True):
            body = _llm_extra_body(
                "deepseek-v4-flash",
                "https://api.deepseek.com",
            )

        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["user_id"], "latency-ab")
        self.assertNotIn("enable_thinking", body)

    def test_dashscope_keeps_verified_enable_thinking_field(self):
        with patch.dict(
            os.environ,
            {"PIPECAT_LLM_ENABLE_THINKING": "false"},
            clear=True,
        ):
            body = _llm_extra_body(
                "qwen3.7-flash",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

        self.assertEqual(body, {"enable_thinking": False})

    def test_unset_thinking_does_not_invent_provider_field(self):
        with patch.dict(
            os.environ,
            {"PIPECAT_LLM_EXTRA_BODY_JSON": '{"service_tier":"priority"}'},
            clear=True,
        ):
            body = _llm_extra_body("third-party-model", "https://llm.test/v1")

        self.assertEqual(body, {"service_tier": "priority"})

    def test_generic_provider_rejects_dashscope_thinking_toggle(self):
        with patch.dict(
            os.environ,
            {"PIPECAT_LLM_ENABLE_THINKING": "false"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "provider-specific"):
                _llm_extra_body("third-party-model", "https://llm.test/v1")

    def test_provider_can_be_explicit(self):
        with patch.dict(
            os.environ,
            {"PIPECAT_LLM_PROVIDER": "deepseek"},
            clear=True,
        ):
            self.assertEqual(_llm_provider("alias", "https://proxy.test/v1"), "deepseek")

    def test_extra_body_must_be_object(self):
        with patch.dict(
            os.environ,
            {"PIPECAT_LLM_EXTRA_BODY_JSON": "[]"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "JSON object"):
                _llm_extra_body("model", "https://llm.test/v1")

    def test_realtime_search_accepts_dashscope_and_normalizes_router_alias(self):
        with patch.dict(
            os.environ,
            {"PIPECAT_REALTIME_SEARCH_MODE": "router"},
            clear=True,
        ):
            mode = _realtime_search_mode(
                "qwen3.7-flash",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        self.assertEqual(mode, "smart")

    def test_realtime_search_fails_closed_on_unsupported_provider(self):
        with patch.dict(
            os.environ,
            {"PIPECAT_REALTIME_SEARCH_MODE": "smart"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "requires the DashScope"):
                _realtime_search_mode(
                    "deepseek-v4-flash",
                    "https://api.deepseek.com",
                )

    def test_warmup_models_include_distinct_active_search_model(self):
        self.assertEqual(
            _llm_warmup_models("qwen3.7-flash", "smart", "qwen-flash"),
            ("qwen3.7-flash", "qwen-flash"),
        )
        self.assertEqual(
            _llm_warmup_models("qwen-flash", "smart", "qwen-flash"),
            ("qwen-flash",),
        )
        self.assertEqual(
            _llm_warmup_models("qwen3.7-flash", "off", "qwen-flash"),
            ("qwen3.7-flash",),
        )


class CustomCascadeWarmupTests(unittest.IsolatedAsyncioTestCase):
    @patch("pipecat_dystream.custom_cascade._warm_llm", new_callable=AsyncMock)
    async def test_wait_ready_warms_base_and_search_models(self, warm_llm):
        tts = SimpleNamespace(
            wait_ready=AsyncMock(),
            warmup=AsyncMock(),
            ready=True,
        )
        components = CustomCascadeComponents(
            stt=SimpleNamespace(pre_roll_secs=0.3),
            llm=object(),
            tts=tts,
            user_aggregator=object(),
            assistant_aggregator=object(),
            llm_model="qwen3.7-flash",
            llm_warmup_models=("qwen3.7-flash", "qwen-flash"),
            llm_extra={"extra_body": {"enable_thinking": False}},
            log=lambda _message: None,
        )

        await components.wait_ready(timeout=1.0)

        tts.wait_ready.assert_awaited_once_with(timeout=1.0)
        tts.warmup.assert_awaited_once()
        self.assertEqual(tts.warmup.await_args.args, ("你好。",))
        self.assertGreater(tts.warmup.await_args.kwargs["timeout"], 0)
        self.assertEqual(
            warm_llm.await_args_list,
            [
                call(
                    components.llm,
                    "qwen3.7-flash",
                    components.llm_extra,
                ),
                call(
                    components.llm,
                    "qwen-flash",
                    components.llm_extra,
                ),
            ],
        )
        self.assertTrue(components.ready)

    @patch("pipecat_dystream.custom_cascade._warm_llm", new_callable=AsyncMock)
    async def test_wait_ready_can_disable_tts_warmup(self, warm_llm):
        tts = SimpleNamespace(
            wait_ready=AsyncMock(),
            warmup=AsyncMock(),
            ready=True,
        )
        components = CustomCascadeComponents(
            stt=SimpleNamespace(pre_roll_secs=0.3),
            llm=object(),
            tts=tts,
            user_aggregator=object(),
            assistant_aggregator=object(),
            llm_model="qwen3.7-flash",
            llm_warmup_models=("qwen3.7-flash",),
            llm_extra={},
            log=lambda _message: None,
        )

        with patch.dict(os.environ, {"PIPECAT_TTS_WARMUP_TEXT": ""}, clear=False):
            await components.wait_ready(timeout=1.0)

        tts.warmup.assert_not_awaited()
        warm_llm.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
