import unittest

import torch

from tests.bench_text_embedding_memory import bits_equal


class EmbeddingJudgeTests(unittest.TestCase):
    def test_exact_values_are_accepted(self):
        value = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8)
        bits_equal(value, value.clone())

    def test_perturbation_nan_shape_and_dtype_are_rejected(self):
        value = torch.ones(2, 4, dtype=torch.bfloat16)
        changed = value.clone()
        changed[0, 0] += 0.125
        invalid = value.clone()
        invalid[0, 0] = float("nan")
        for candidate in (changed, invalid, value[:1], value.float()):
            with self.assertRaises(ValueError):
                bits_equal(value, candidate)


if __name__ == "__main__":
    unittest.main()
