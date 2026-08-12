from __future__ import annotations

from datetime import datetime, timezone
import unittest

from pipecat_dystream.realtime_search import (
    RealtimeSearchDecision,
    RealtimeSearchPolicy,
    add_dashscope_search_params,
    last_user_text,
    normalize_realtime_search_mode,
    normalize_realtime_search_strategy,
)


class FakeContext:
    def __init__(self, messages):
        self._messages = messages

    def get_messages(self):
        return self._messages


class RealtimeSearchPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = RealtimeSearchPolicy("smart")

    def assertSearch(self, query: str, reason: str | None = None):
        decision = self.policy.decide(query)
        self.assertTrue(decision.enabled, query)
        self.assertTrue(decision.forced, query)
        if reason is not None:
            self.assertEqual(decision.reason, reason)

    def assertBypass(self, query: str):
        decision = self.policy.decide(query)
        self.assertFalse(decision.enabled, query)
        self.assertFalse(decision.forced, query)

    def test_routes_multiple_real_time_domains(self):
        for query in (
            "杭州天气怎么样？",
            "今天有什么重要新闻？",
            "英伟达现在的股价是多少？",
            "北京飞深圳的航班有没有延误？",
            "湖人今天比赛比分是多少？",
            "这个 Python 库的最新版本是什么？",
            "谁是现任美国总统？",
            "美元兑人民币汇率是多少？",
        ):
            with self.subTest(query=query):
                self.assertSearch(query)

    def test_explicit_search_is_generic(self):
        self.assertSearch("帮我联网查一下这个项目最近的更新", "explicit_search")
        self.assertSearch("search the web for the latest release", "explicit_search")

    def test_conversational_now_and_today_stay_on_fast_path(self):
        for query in (
            "你现在在干嘛？",
            "你今天心情怎么样？",
            "现在给我讲个笑话。",
            "请解释什么是流式推理。",
            "我们刚刚讨论到哪里了？",
            "我们现在用的是什么模型？",
            "解释天气和气候有什么区别。",
            "什么是股票？",
            "股票是什么？",
        ):
            with self.subTest(query=query):
                self.assertBypass(query)

    def test_mode_semantics_and_aliases(self):
        self.assertFalse(RealtimeSearchPolicy("off").decide("杭州天气").enabled)
        auto = RealtimeSearchPolicy("provider_auto").decide("你好")
        self.assertTrue(auto.enabled)
        self.assertFalse(auto.forced)
        self.assertEqual(auto.reason, "provider_auto")
        always = RealtimeSearchPolicy("always").decide("你好")
        self.assertTrue(always.enabled)
        self.assertTrue(always.forced)
        self.assertEqual(normalize_realtime_search_mode("router"), "smart")

    def test_invalid_mode_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "must be one of"):
            RealtimeSearchPolicy("magic")


class RealtimeSearchRequestTests(unittest.TestCase):
    def test_preserves_existing_provider_fields_without_mutating_input(self):
        original = {
            "model": "qwen3.7-flash",
            "messages": [
                {"role": "system", "content": "原系统指令"},
                {"role": "user", "content": "杭州天气"},
            ],
            "extra_body": {
                "enable_thinking": False,
                "search_options": {"enable_source": False},
            },
        }
        updated = add_dashscope_search_params(
            original,
            RealtimeSearchDecision(True, True, "live_subject"),
            current_time=datetime(2026, 8, 11, 6, 30, tzinfo=timezone.utc),
        )

        self.assertEqual(original["extra_body"], {
            "enable_thinking": False,
            "search_options": {"enable_source": False},
        })
        self.assertIsNot(updated, original)
        self.assertEqual(original["messages"][0]["content"], "原系统指令")
        self.assertIn("2026-08-11 14:30", updated["messages"][0]["content"])
        self.assertIn("不超过二十个汉字", updated["messages"][0]["content"])
        self.assertIn("原系统指令", updated["messages"][0]["content"])
        self.assertEqual(updated["extra_body"]["enable_thinking"], False)
        self.assertEqual(updated["extra_body"]["enable_search"], True)
        self.assertEqual(
            updated["extra_body"]["search_options"],
            {
                "enable_source": False,
                "search_strategy": "turbo",
                "forced_search": True,
            },
        )

    def test_auto_mode_does_not_force_provider_search(self):
        updated = add_dashscope_search_params(
            {"model": "qwen3.7-flash"},
            RealtimeSearchDecision(True, False, "provider_auto"),
        )
        self.assertEqual(
            updated["extra_body"],
            {
                "enable_search": True,
                "search_options": {"search_strategy": "turbo"},
            },
        )

    def test_strategy_is_validated(self):
        self.assertEqual(normalize_realtime_search_strategy("TURBO"), "turbo")
        with self.assertRaisesRegex(ValueError, "must be one of"):
            normalize_realtime_search_strategy("fastest")

    def test_disabled_decision_returns_exact_request_object(self):
        original = {"model": "qwen3.7-flash"}
        updated = add_dashscope_search_params(
            original,
            RealtimeSearchDecision(False, False, "ordinary_chat"),
        )
        self.assertIs(updated, original)

    def test_last_user_text_supports_multimodal_message_content(self):
        context = FakeContext(
            [
                {"role": "user", "content": "旧问题"},
                {"role": "assistant", "content": "旧答案"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "杭州"},
                        {"type": "text", "text": "天气怎么样"},
                    ],
                },
            ]
        )
        self.assertEqual(last_user_text(context), "杭州 天气怎么样")


if __name__ == "__main__":
    unittest.main()
