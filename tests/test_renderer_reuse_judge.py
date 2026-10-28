import unittest

import numpy as np

from tests.bench_renderer_reuse import compare_outputs, compare_records


class RendererReuseJudgeTests(unittest.TestCase):
    def test_exact_output_is_accepted(self):
        reference = np.ones((4, 3, 4, 4), dtype=np.float32)
        compare_outputs(reference, reference.copy())

    def test_negative_controls_are_rejected(self):
        reference = np.ones((4, 3, 4, 4), dtype=np.float32)
        perturbed = reference.copy()
        perturbed.flat[0] = np.nextafter(np.float32(1), np.float32(2))
        invalid = reference.copy()
        invalid.flat[0] = np.nan
        for candidate in (perturbed, reference[:-1], invalid, reference.astype(np.float64)):
            with self.assertRaises(ValueError):
                compare_outputs(reference, candidate)

    def test_signed_zero_is_also_a_bitwise_difference(self):
        reference = np.zeros(1, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "bits"):
            compare_outputs(reference, -reference)

    def test_changed_workload_or_intermediate_is_rejected(self):
        reference = dict(state_sha256="same", input_shapes=[[1, 512]], math_modes={"tf32": False},
                         layer_outputs=[{"name": "layer", "sha256": "same"}])
        compare_records(reference, dict(reference))
        for key in reference:
            with self.assertRaisesRegex(ValueError, key):
                compare_records(reference, {**reference, key: "changed"})


if __name__ == "__main__":
    unittest.main()
