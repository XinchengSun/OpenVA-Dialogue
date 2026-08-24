import unittest

from gpu_scheduler import MAX_CARDS, choose


class GpuSchedulerTests(unittest.TestCase):
    def test_selects_two_least_loaded_idle_cards(self):
        cards = [
            {"index": 0, "utilization": 3, "memory_used_mib": 1000, "memory_total_mib": 24000},
            {"index": 1, "utilization": 90, "memory_used_mib": 1000, "memory_total_mib": 24000},
            {"index": 2, "utilization": 0, "memory_used_mib": 2000, "memory_total_mib": 24000},
        ]
        self.assertEqual(choose(cards), [2, 0])

    def test_refuses_when_two_idle_cards_are_not_available(self):
        cards = [
            {"index": 0, "utilization": 0, "memory_used_mib": 1000, "memory_total_mib": 24000},
            {"index": 1, "utilization": 99, "memory_used_mib": 23000, "memory_total_mib": 24000},
        ]
        with self.assertRaises(RuntimeError):
            choose(cards)

    def test_hard_limit_is_four_cards(self):
        self.assertEqual(MAX_CARDS, 4)
        with self.assertRaises(ValueError):
            choose([], MAX_CARDS + 1)


if __name__ == "__main__":
    unittest.main()
