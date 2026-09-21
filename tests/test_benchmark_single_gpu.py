"""CPU-only regressions for CUDA thread-ID ownership, with no live GPU access."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "benchmark_single_gpu", Path(__file__).resolve().parents[1] / "scripts/benchmark_single_gpu.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class ComputeProcessOwnershipTests(unittest.TestCase):
    def resolve(self, reported_pid, identities, tasks=None):
        with patch.object(benchmark.os, "name", "posix"), patch.object(
            benchmark, "linux_process_identity", side_effect=lambda pid: identities.get(pid)
        ):
            return benchmark.compute_process_owner(
                reported_pid, {100}, {100: "owner-start"},
                tasks if tasks is not None else {100: (100, "owner-start"), 123: (100, "thread-start")},
            )

    def test_cuda_thread_is_assigned_to_owning_service(self):
        actual = self.resolve(123, {100: (100, "owner-start"), 123: (100, "thread-start")})
        self.assertEqual(actual, {
            "owner_pid": 100, "is_test_service": True, "service_identity_unverified": False,
        })

    def test_unrelated_cuda_thread_is_not_a_test_service(self):
        actual = self.resolve(999, {999: (900, "unrelated-start")})
        self.assertEqual(actual["owner_pid"], 900)
        self.assertFalse(actual["is_test_service"])
        self.assertFalse(actual["service_identity_unverified"])

    def test_reused_thread_id_does_not_inherit_old_ownership(self):
        actual = self.resolve(123, {100: (100, "owner-start"), 123: (100, "new-thread-start")})
        self.assertFalse(actual["is_test_service"])
        self.assertTrue(actual["service_identity_unverified"])

    def test_reused_owner_pid_does_not_inherit_old_ownership(self):
        actual = self.resolve(123, {100: (100, "new-owner-start"), 123: (100, "thread-start")})
        self.assertFalse(actual["is_test_service"])
        self.assertTrue(actual["service_identity_unverified"])

    def test_service_thread_created_during_sample_is_inconclusive(self):
        actual = self.resolve(124, {100: (100, "owner-start"), 124: (100, "new-thread")})
        self.assertFalse(actual["is_test_service"])
        self.assertTrue(actual["service_identity_unverified"])

    def test_disappeared_service_thread_does_not_silently_pass(self):
        actual = self.resolve(123, {100: (100, "owner-start")})
        self.assertFalse(actual["is_test_service"])
        self.assertTrue(actual["service_identity_unverified"])

    def test_direct_service_process_still_resolves(self):
        actual = self.resolve(100, {100: (100, "owner-start")})
        self.assertTrue(actual["is_test_service"])
        self.assertEqual(actual["owner_pid"], 100)


class GraphicsProcessSamplingTests(unittest.TestCase):
    XML = """<nvidia_smi_log>
      <gpu><uuid>GPU-zero</uuid><product_name>Test 0</product_name>
        <fb_memory_usage><used>6 MiB</used><total>24564 MiB</total></fb_memory_usage>
        <utilization><gpu_util>0 %</gpu_util></utilization>
        <processes><process_info><pid>123</pid><type>G</type>
          <process_name>python</process_name><used_memory>6 MiB</used_memory>
        </process_info></processes></gpu>
      <gpu><uuid>GPU-one</uuid><product_name>Test 1</product_name>
        <fb_memory_usage><used>1000 MiB</used><total>24564 MiB</total></fb_memory_usage>
        <utilization><gpu_util>80 %</gpu_util></utilization>
        <processes><process_info><pid>100</pid><type>C</type>
          <process_name>python</process_name><used_memory>1000 MiB</used_memory>
        </process_info></processes></gpu>
    </nvidia_smi_log>"""

    def test_graphics_thread_on_other_gpu_fails_strict_single_gpu_check(self):
        identities = {100: (100, "owner-start"), 123: (100, "thread-start")}
        with patch.object(benchmark, "os", SimpleNamespace(name="posix")), \
             patch.object(benchmark, "process_table", return_value={100: (1, "owner-start")}), \
             patch.object(benchmark, "service_task_snapshot", return_value=identities), \
             patch.object(benchmark, "linux_process_identity", side_effect=identities.get), \
             patch.object(benchmark, "checked", return_value=self.XML) as checked:
            snapshot = benchmark.gpu_snapshot([100], {}, "1")
        checked.assert_called_once_with(["nvidia-smi", "-q", "-x"])
        graphics = snapshot["gpu_processes"][0]
        self.assertEqual(graphics["type"], "G")
        self.assertEqual(graphics["owner_pid"], 100)
        self.assertTrue(graphics["is_test_service"])
        self.assertEqual(graphics["memory_used_mib"], 6)
        args = SimpleNamespace(server_pid=[100], duration_sec=2, input_wav=None,
                               max_memory_mib=None, min_delivered_fps=None,
                               base="ws://127.0.0.1:7871", gpu="1")
        captured = {"samples": [{"gpu": snapshot, "health": {"status": "ok"}}],
                    "health": [{"status": "ok"}], "errors": [], "epochs": [],
                    "connected_at_sec": 0, "captured_until_sec": 2}
        videos = [{"decoded_frame_count": 25, "ffprobe_returncode": 0,
                   "decoder_error_present": False}]
        report = benchmark.summarize(args, captured, videos)
        self.assertFalse(report["checks"]["service_single_gpu"])
        self.assertEqual(report["status"], "fail")


class AssistantTurnLifecycleTests(unittest.TestCase):
    @staticmethod
    def event(kind, turn=1, generation=0):
        return {"type": "assistant_" + kind, "turn_id": turn, "generation": generation}

    def test_valid_ordered_lifecycle(self):
        state = benchmark.AssistantTurnLifecycle(5)
        self.assertFalse(state.observe(6, self.event("turn_started")))
        self.assertFalse(state.observe(7, self.event("media_boundary")))
        self.assertTrue(state.observe(8, self.event("media_ended")))
        self.assertEqual(state.key, (1, 0))
        self.assertEqual(list(state.marks.values()), [6, 7, 8])

    def test_ended_event_alone_cannot_complete_turn(self):
        state = benchmark.AssistantTurnLifecycle(5)
        self.assertFalse(state.observe(8, self.event("media_ended")))
        self.assertEqual(state.marks, {})

    def test_events_from_different_turns_cannot_be_combined(self):
        state = benchmark.AssistantTurnLifecycle(5)
        state.observe(6, self.event("turn_started"))
        self.assertFalse(state.observe(7, self.event("media_boundary", turn=2)))
        self.assertFalse(state.observe(8, self.event("media_ended")))
        self.assertNotIn("assistant_media_boundary", state.marks)

    def test_events_from_different_generations_cannot_be_combined(self):
        state = benchmark.AssistantTurnLifecycle(5)
        state.observe(6, self.event("turn_started"))
        self.assertFalse(state.observe(7, self.event("media_boundary", generation=1)))
        self.assertFalse(state.observe(8, self.event("media_ended")))

    def test_partial_reply_started_during_input_is_ignored(self):
        state = benchmark.AssistantTurnLifecycle(5)
        self.assertFalse(state.observe(4, self.event("turn_started")))
        self.assertFalse(state.observe(6, self.event("media_boundary")))
        self.assertFalse(state.observe(7, self.event("media_ended")))
        self.assertEqual(state.marks, {})
        state.observe(8, self.event("turn_started", turn=2))
        state.observe(9, self.event("media_boundary", turn=2))
        self.assertTrue(state.observe(10, self.event("media_ended", turn=2)))
        self.assertEqual(state.key, (2, 0))

    def test_end_before_boundary_does_not_complete_later(self):
        state = benchmark.AssistantTurnLifecycle(5)
        state.observe(6, self.event("turn_started"))
        self.assertFalse(state.observe(7, self.event("media_ended")))
        self.assertFalse(state.observe(8, self.event("media_boundary")))
        self.assertNotIn("assistant_media_ended", state.marks)

    def test_new_start_discards_incomplete_previous_lifecycle(self):
        state = benchmark.AssistantTurnLifecycle(5)
        state.observe(6, self.event("turn_started"))
        state.observe(7, self.event("media_boundary"))
        state.observe(8, self.event("turn_started", turn=2))
        self.assertFalse(state.observe(9, self.event("media_ended")))
        self.assertFalse(state.observe(10, self.event("media_ended", turn=2)))
        state.observe(11, self.event("media_boundary", turn=2))
        self.assertTrue(state.observe(12, self.event("media_ended", turn=2)))
        self.assertEqual(state.marks["assistant_turn_started"], 8)

    def test_dialogue_summary_requires_verified_lifecycle_not_just_length(self):
        args = SimpleNamespace(server_pid=[], duration_sec=2, input_wav=Path("fixture.wav"),
                               turns=1, max_memory_mib=None, min_delivered_fps=None,
                               base="ws://127.0.0.1:7871", gpu="1")
        captured = {"samples": [], "health": [], "errors": [], "epochs": [],
                    "connected_at_sec": 0, "captured_until_sec": 2,
                    "dialogue": {"turns": [{"events_at_sec": {"assistant_media_ended": 1}}]}}
        report = benchmark.summarize(args, captured, [])
        self.assertFalse(report["checks"]["dialogue_completed"])


if __name__ == "__main__":
    unittest.main()
