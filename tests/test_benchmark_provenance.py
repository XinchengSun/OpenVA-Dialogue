"""CPU regressions for traceability and honest performance comparisons."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.benchmark_provenance import checkout_identity, collect_provenance
from scripts.compare_benchmarks import compare, delta
from scripts import benchmark_single_gpu as benchmark


class ProvenanceTests(unittest.TestCase):
    def test_transport_metrics_allowlist_never_copies_health_secrets(self):
        payload = {"status": "ok", "launch_token": "private-token", "media_transport": [
            {"client_id": 7, "pending_callbacks": 3, "dropped_best_effort": 9,
             "launch_token": "private-token", "unknown": "private-text"},
        ], "log_transport": [{"pending_callbacks": float("nan")}]}
        with patch.object(benchmark.urllib.request, "urlopen", return_value=io.StringIO(json.dumps(payload))):
            health = benchmark.health_snapshot("http://example.invalid/health")
        self.assertEqual(health["media_transport"], [{"client_id": 7, "pending_callbacks": 3, "dropped_best_effort": 9}])
        self.assertEqual(health["log_transport"], [{}])
        self.assertNotIn("private-", json.dumps(health))

    def test_no_private_configuration_or_input_is_serialized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "private.env"
            config.write_text("API_KEY=private-test-key\nPORT=1234\n", encoding="utf-8")
            wav = root / "private-name.wav"
            wav.write_bytes(b"private-test-audio")
            (root / "weights-manifest.json").write_text('{"models": []}', encoding="utf-8")
            args = SimpleNamespace(profile="custom", runtime_config=[config], input_wav=wav,
                                   service_commit="A" * 40,
                                   instruction="private-test-instruction", duration_sec=30,
                                   sample_sec=1, input_delay_sec=2, turns=3,
                                   turn_timeout_sec=20, voice_threshold=500)
            with patch.dict(os.environ, {"SECRET_TEST": "private-environment-value"}):
                report = collect_provenance(root, args)
            encoded = json.dumps(report)
            for private in ("private-test-key", "private-name.wav", "private-test-audio",
                            "private-test-instruction", "private-environment-value", directory):
                self.assertNotIn(private, encoded)
            self.assertEqual(report["runtime_config_sha256"], [hashlib.sha256(config.read_bytes()).hexdigest()])
            self.assertEqual(report["input_wav_sha256"], hashlib.sha256(wav.read_bytes()).hexdigest())
            self.assertIsNone(report["checkout"]["commit"])
            self.assertEqual(report["declared_service_commit"], "a" * 40)

    def test_dirty_staged_unstaged_and_untracked_state_is_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
            git("init")
            source = root / "source.py"
            source.write_text("print(1)\n", encoding="utf-8")
            git("add", "source.py")
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
            clean = checkout_identity(root)
            self.assertFalse(clean["dirty_tracked"])
            source.write_text("print(2)\n", encoding="utf-8")
            dirty = checkout_identity(root)
            self.assertTrue(dirty["dirty_tracked"])
            self.assertNotEqual(clean["tracked_diff_sha256"], dirty["tracked_diff_sha256"])
            git("add", "source.py")
            self.assertEqual(dirty["tracked_diff_sha256"], checkout_identity(root)["tracked_diff_sha256"])
            (root / "new.py").write_text("pass\n", encoding="utf-8")
            self.assertEqual(checkout_identity(root)["untracked_file_count"], 1)


class ComparisonTests(unittest.TestCase):
    def report(self):
        return {
            "status": "pass",
            "provenance": {
                "checkout": {"commit": "a" * 40, "dirty_tracked": False, "untracked_file_count": 0},
                "declared_service_commit": "a" * 40,
                "measurement_source_sha256": {"benchmark_single_gpu.py": "a" * 64,
                                              "benchmark_provenance.py": "b" * 64},
                "profile": "single_gpu_voxcpm2", "runtime_config_sha256": ["config-hash"],
                "declared_weights_manifest_sha256": "weights-hash", "input_wav_sha256": "input-hash",
                "instruction_sha256": "instruction-hash",
                "measurement": {"duration_sec": 60, "sample_sec": 1, "input_delay_sec": 2,
                                "turns": 3, "turn_timeout_sec": 20, "voice_threshold": 500},
                "measurement_runtime": {"python": "3.11", "system": "Linux", "release": "test",
                                        "machine": "x86_64", "packages": {}},
            },
            "capture": {"samples": [{"gpu": {"target": {"uuid": "GPU-test", "name": "GPU", "memory_total_mib": 24000}}}]},
            "decoded_frames_delivered_per_wall_second": 10,
            "total_target_gpu_memory_mib": {"max": 13000},
            "frame_freshness": {"latest_frame_age_sec": {"p95": .2}},
        }

    def test_matched_runs_keep_metric_boundaries(self):
        before = self.report()
        after = copy.deepcopy(before)
        after["provenance"]["checkout"]["commit"] = "b" * 40
        after["decoded_frames_delivered_per_wall_second"] = 12
        result = compare(before, after)
        self.assertEqual(result["status"], "matched_recorded_conditions")
        self.assertEqual(result["metrics"]["delivered_fps"]["relative_delta_pct"], 20)
        self.assertNotIn("model_generation_fps", result["metrics"])

    def test_faster_failed_run_cannot_pass_comparison(self):
        after = self.report()
        after["status"] = "fail"
        after["decoded_frames_delivered_per_wall_second"] = 100
        result = compare(self.report(), after)
        self.assertEqual(result["status"], "review_required")
        self.assertIn("candidate: benchmark status is not pass", result["comparison_gaps"])

    def test_changed_input_config_hardware_or_method_is_flagged(self):
        for field in ("input_wav_sha256", "runtime_config_sha256", "measurement", "measurement_runtime",
                      "measurement_source_sha256"):
            with self.subTest(field=field):
                after = self.report()
                after["provenance"][field] = "changed"
                self.assertIn(f"mismatched provenance: {field}", compare(self.report(), after)["comparison_gaps"])
        after = self.report()
        after["capture"]["samples"][0]["gpu"]["target"]["uuid"] = "another-GPU"
        self.assertEqual(compare(self.report(), after)["status"], "review_required")

    def test_legacy_missing_provenance_and_uncommitted_code_need_review(self):
        for change in (lambda p: p.pop("provenance"),
                       lambda p: p["provenance"]["checkout"].update(dirty_tracked=True),
                       lambda p: p["provenance"]["checkout"].update(untracked_file_count=1)):
            after = self.report()
            change(after)
            self.assertEqual(compare(self.report(), after)["status"], "review_required")

    def test_zero_missing_and_nonfinite_values_never_become_speedups(self):
        for before, after in ((None, 1), (float("nan"), 1), (1, float("inf")), (True, 2)):
            self.assertIsNone(delta(before, after)["relative_delta_pct"])
        self.assertIsNone(delta(0, 2)["relative_delta_pct"])

    def test_nonfinite_metric_is_reviewable_valid_json(self):
        after = self.report()
        after["decoded_frames_delivered_per_wall_second"] = float("nan")
        result = compare(self.report(), after)
        self.assertEqual(result["status"], "review_required")
        self.assertIsNone(result["metrics"]["delivered_fps"]["candidate"])
        json.dumps(result, allow_nan=False)

    def test_matching_missing_gpu_fields_do_not_establish_hardware_identity(self):
        before = self.report()
        before["capture"]["samples"][0]["gpu"]["target"] = {"index": "0"}
        self.assertEqual(compare(before, copy.deepcopy(before))["status"], "review_required")

    def test_matching_missing_methodology_cannot_be_declared_matched(self):
        for field in ("declared_weights_manifest_sha256", "measurement", "measurement_runtime",
                      "measurement_source_sha256"):
            before = self.report()
            del before["provenance"][field]
            self.assertEqual(compare(before, copy.deepcopy(before))["status"], "review_required")


if __name__ == "__main__":
    unittest.main()
