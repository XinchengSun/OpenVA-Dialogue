import unittest
from pathlib import Path

from dual_gpu_mic_realtime_ui_v8_playbuffer import (
    MOTION_AUDIO_HISTORY_HARD_CAP_SEC,
    motion_history_hard_cap_samples,
    motion_history_would_exceed_cap,
)


class MotionHistoryCompactionTests(unittest.TestCase):
    def test_hard_cap_triggers_at_twelve_seconds_for_either_branch(self):
        self.assertEqual(MOTION_AUDIO_HISTORY_HARD_CAP_SEC, 12)
        cap = motion_history_hard_cap_samples(16_000, 64_000)
        self.assertEqual(cap, 192_000)
        self.assertFalse(
            motion_history_would_exceed_cap(188_800, 188_800, 3_200, cap)
        )
        self.assertTrue(
            motion_history_would_exceed_cap(192_000, 0, 3_200, cap)
        )
        self.assertTrue(
            motion_history_would_exceed_cap(0, 192_000, 3_200, cap)
        )

    def test_large_keep_window_retains_headroom_without_compacting_every_hop(self):
        audio_sr = 16_000
        hop_samples = 3_200
        keep_samples = 12 * audio_sr
        cap = motion_history_hard_cap_samples(audio_sr, keep_samples)

        self.assertEqual(cap, 16 * audio_sr)
        current = keep_samples
        self.assertFalse(
            motion_history_would_exceed_cap(
                current,
                current,
                hop_samples,
                cap,
            )
        )
        hops_before_next_compact = 0
        while not motion_history_would_exceed_cap(
            current,
            current,
            hop_samples,
            cap,
        ):
            current += hop_samples
            hops_before_next_compact += 1
        self.assertEqual(current, cap)
        self.assertEqual(hops_before_next_compact, 20)

    def test_hard_cap_compacts_before_appending_the_next_hop(self):
        worker = (
            Path(__file__).resolve().parents[1]
            / "dual_gpu_mic_realtime_ui_v8_playbuffer.py"
        ).read_text(encoding="utf-8")
        consume_loop = worker.split(
            "while pending.shape[0] >= hop_samples", 1
        )[1].split("# Original app does:", 1)[0]

        self.assertIn("motion_history_would_exceed_cap(", consume_loop)
        self.assertIn("real_audio.shape[0]", consume_loop)
        self.assertIn("real_audio_other.shape[0]", consume_loop)
        self.assertIn("hop_samples", consume_loop)
        self.assertIn("history_hard_cap_samples", consume_loop)
        self.assertIn("compact_stream_history()", consume_loop)
        self.assertLess(
            consume_loop.index("compact_stream_history()"),
            consume_loop.index("real_audio = np.concatenate"),
        )

    def test_pre_append_compaction_keeps_all_outputs_from_current_hop(self):
        context_frames = 94
        keep_frames = 100
        window = 96
        feature_lag_frames = 3
        hop_frames = 5
        generated_after_compact = (
            context_frames
            + keep_frames
            - window
            + 1
            - feature_lag_frames
        )
        target_after_append = (
            context_frames
            + keep_frames
            + hop_frames
            - window
            + 1
            - feature_lag_frames
        )

        self.assertEqual(generated_after_compact, 96)
        self.assertEqual(target_after_append, 101)
        self.assertEqual(target_after_append - generated_after_compact, hop_frames)


if __name__ == "__main__":
    unittest.main()
