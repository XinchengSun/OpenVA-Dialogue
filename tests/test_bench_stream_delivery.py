from __future__ import annotations

import unittest

from scripts.bench_voxcpm2_prompt_cache import _stream_delivery_stats


class StreamDeliveryStatsTests(unittest.TestCase):
    def test_reports_starvation_after_buffer_is_exhausted(self) -> None:
        stats = _stream_delivery_stats(
            [(0.0, 2_000), (0.5, 2_000), (2.5, 2_000)],
            sample_rate=1_000,
        )

        self.assertEqual(stats["pcm_chunks"], 3)
        self.assertAlmostEqual(stats["max_pcm_interarrival_ms"], 2_000.0)
        self.assertAlmostEqual(stats["playback_starvation_total_ms"], 500.0)
        self.assertAlmostEqual(stats["playback_starvation_max_ms"], 500.0)

    def test_empty_stream_has_zero_delivery_metrics(self) -> None:
        stats = _stream_delivery_stats([], sample_rate=44_100)

        self.assertEqual(stats["pcm_chunks"], 0)
        self.assertEqual(stats["playback_starvation_total_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
