import argparse
import os
import unittest
from unittest.mock import patch

from scripts import probe_qwen_audio_s2s as probe


class QwenAudioProbeTests(unittest.TestCase):
    def _args(self, **overrides):
        values = {
            "base_url": "",
            "legacy_public_endpoint": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_workspace_builds_official_beijing_maas_url(self):
        with patch.dict(
            os.environ,
            {"PIPECAT_S2S_WORKSPACE_ID": "workspace-123"},
            clear=True,
        ):
            actual = probe._base_url(self._args())
        self.assertEqual(
            actual,
            "wss://workspace-123.cn-beijing.maas.aliyuncs.com"
            "/api-ws/v1/realtime",
        )

    def test_explicit_url_has_precedence(self):
        with patch.dict(
            os.environ,
            {"PIPECAT_S2S_WORKSPACE_ID": "workspace-123"},
            clear=True,
        ):
            actual = probe._base_url(
                self._args(base_url="wss://example.invalid/realtime")
            )
        self.assertEqual(actual, "wss://example.invalid/realtime")

    def test_diagnostic_legacy_flag_overrides_environment_routes(self):
        with patch.dict(
            os.environ,
            {
                "PIPECAT_S2S_BASE_URL": "wss://configured.invalid/realtime",
                "PIPECAT_S2S_WORKSPACE_ID": "workspace-123",
            },
            clear=True,
        ):
            actual = probe._base_url(
                self._args(legacy_public_endpoint=True)
            )
        self.assertEqual(actual, probe.LEGACY_PUBLIC_ENDPOINT)

    def test_missing_workspace_fails_without_diagnostic_legacy_flag(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                RuntimeError,
                "PIPECAT_S2S_WORKSPACE_ID or PIPECAT_S2S_BASE_URL",
            ):
                probe._base_url(self._args())

    def test_model_query_is_appended_once(self):
        base = "wss://workspace.invalid/realtime"
        self.assertEqual(
            probe._url(base, "qwen-audio-3.0-realtime-flash"),
            base + "?model=qwen-audio-3.0-realtime-flash",
        )
        configured = base + "?model=already-set"
        self.assertEqual(probe._url(configured, "ignored"), configured)


if __name__ == "__main__":
    unittest.main()
