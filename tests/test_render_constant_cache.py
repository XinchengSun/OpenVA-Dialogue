from __future__ import annotations

import unittest
from unittest import mock

import torch
from torch import nn

from scripts.benchmark_render_constants import OriginalDirection, OriginalWSConv3d, load_production_class


Direction = load_production_class("flow_estimator.py", "Direction")
FaceGenerator = load_production_class("face_generator.py", "FaceGenerator")
WSConv3d = load_production_class("../VASA1/building_blocks.py", "WSConv3d")


def grid_holder():
    # Exercise the production grid-cache method with real tensor/grid_sample
    # operations, without allocating the unrelated full-size face decoder.
    module = FaceGenerator.__new__(FaceGenerator)
    nn.Module.__init__(module)
    grids = torch.meshgrid(*(torch.linspace(-1, 1, n) for n in (4, 8, 8)), indexing="ij")
    module.identity_grid = torch.stack(grids, dim=-1).flip(-1)
    module._cached_identity_grid = None
    module._cached_identity_grid_key = None
    return module.eval()


class DirectionCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261026)
        self.module = Direction(64, 8).eval()
        self.values = torch.randn(5, 1, 8)

    def test_outputs_are_exact_and_qr_runs_once_for_multiple_frames(self):
        reference = OriginalDirection(self.module.weight).eval()
        with torch.inference_mode():
            expected = [reference(value) for value in self.values]
            with mock.patch.object(torch, "qr", wraps=torch.qr) as qr:
                actual = [self.module(value) for value in self.values]
            self.assertEqual(qr.call_count, 1)
        for output, before in zip(actual, expected):
            torch.testing.assert_close(output, before, rtol=0, atol=0)

    def test_mutation_load_and_parameter_replacement_invalidate_basis(self):
        value = self.values[0]
        with torch.no_grad(), mock.patch.object(torch, "qr", wraps=torch.qr) as qr:
            self.module(value)
            self.module.weight.add_(0.1)
            self.module(value)
            self.module.load_state_dict({"weight": torch.randn_like(self.module.weight)})
            self.module(value)
            self.module.weight = nn.Parameter(torch.randn_like(self.module.weight))
            actual = self.module(value)
            self.assertEqual(qr.call_count, 4)
        with torch.no_grad():
            torch.testing.assert_close(actual, OriginalDirection(self.module.weight)(value), rtol=0, atol=0)
        self.assertEqual(set(self.module.state_dict()), {"weight"})

    def test_training_and_grad_enabled_eval_keep_original_gradients(self):
        with torch.inference_mode():
            self.module(self.values[0])
        for training in (True, False):
            self.module.train(training)
            reference = OriginalDirection(self.module.weight)
            value = self.values[0].clone().requires_grad_()
            other_value = value.detach().clone().requires_grad_()
            self.module.zero_grad()
            actual = self.module(value)
            expected = reference(other_value)
            actual.square().sum().backward()
            expected.square().sum().backward()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(self.module.weight.grad, reference.weight.grad, rtol=0, atol=0)
            torch.testing.assert_close(value.grad, other_value.grad, rtol=0, atol=0)
            self.assertIsNone(self.module._cached_q)
        with torch.no_grad():
            self.module(self.values[0])
            self.assertIsNotNone(self.module._cached_q)

    def test_dtype_change_recomputes_and_returned_basis_is_not_mutable_cache(self):
        with torch.no_grad(), mock.patch.object(torch, "qr", wraps=torch.qr) as qr:
            basis = self.module(None)
            basis.zero_()
            actual = self.module(self.values[0])
            self.assertGreater(actual.abs().sum().item(), 0)
            self.module.double()
            self.module(self.values[0].double())
            self.assertEqual(qr.call_count, 2)

    def test_training_without_grad_does_not_reuse_inference_basis(self):
        with torch.no_grad(), mock.patch.object(torch, "qr", wraps=torch.qr) as qr:
            self.module(self.values[0])
            self.module.train()
            self.module(self.values[0])
            self.module(self.values[0])
            self.module.eval()
            self.module(self.values[0])
            self.assertEqual(qr.call_count, 4)

    def test_dtype_round_trip_invalidates_even_without_intermediate_forward(self):
        with torch.no_grad(), mock.patch.object(torch, "qr", wraps=torch.qr) as qr:
            self.module(self.values[0])
            self.module.half().float()
            self.assertIsNone(self.module._cached_q)
            actual = self.module(self.values[0])
            self.assertEqual(qr.call_count, 2)
            torch.testing.assert_close(actual, OriginalDirection(self.module.weight)(self.values[0]), rtol=0, atol=0)


class IdentityGridCacheTests(unittest.TestCase):
    def test_grid_sample_preserves_original_outputs_and_one_device_conversion(self):
        module = grid_holder()
        volume = torch.randn(1, 2, 4, 8, 8)
        deltas = torch.randn(1, 4, 8, 8, 3) * 0.1
        original_to = torch.Tensor.to
        conversions = []

        def tracked_to(tensor, *args, **kwargs):
            if tensor is module.identity_grid:
                conversions.append(args)
            return original_to(tensor, *args, **kwargs)

        with torch.inference_mode():
            expected = torch.nn.functional.grid_sample(
                volume, module.identity_grid[None].to(volume.device) + deltas,
                mode="bilinear", align_corners=False,
            )
            with mock.patch.object(torch.Tensor, "to", tracked_to):
                for _ in range(5):
                    actual = torch.nn.functional.grid_sample(
                        volume, module._identity_grid_on(volume.device)[None] + deltas,
                        mode="bilinear", align_corners=False,
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(len(conversions), 1)

    def test_grid_keeps_fp32_when_model_dtype_changes_and_refreshes_mutations(self):
        module = grid_holder()
        with torch.inference_mode():
            module._identity_grid_on(torch.device("cpu"))
        module.half()
        with torch.no_grad():
            self.assertEqual(module._identity_grid_on(torch.device("cpu")).dtype, torch.float32)
            old_key = module._cached_identity_grid_key
            module.identity_grid.add_(0.1)
            module._identity_grid_on(torch.device("cpu"))
            self.assertNotEqual(module._cached_identity_grid_key, old_key)
            module.identity_grid = module.identity_grid.clone()
            old_key = module._cached_identity_grid_key
            module._identity_grid_on(torch.device("cpu"))
            self.assertNotEqual(module._cached_identity_grid_key, old_key)

    def test_grad_and_training_paths_bypass_cache(self):
        module = grid_holder()
        with torch.inference_mode():
            module._identity_grid_on(torch.device("cpu"))
        module._identity_grid_on(torch.device("cpu"))
        self.assertIsNone(module._cached_identity_grid)
        with torch.no_grad():
            module.train()
            module._identity_grid_on(torch.device("cpu"))
            self.assertIsNone(module._cached_identity_grid)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available; no GPU equivalence claim")
    def test_gpu_direction_and_grid_are_exact_across_device_moves(self):
        module = Direction(512, 64).eval()
        grid = grid_holder()
        with torch.inference_mode():
            for device in (torch.device("cuda"), torch.device("cpu"), torch.device("cuda")):
                module.to(device)
                value = torch.randn(1, 64, device=device)
                torch.testing.assert_close(module(value), OriginalDirection(module.weight)(value), rtol=0, atol=0)
                torch.testing.assert_close(grid._identity_grid_on(device), grid.identity_grid.to(device), rtol=0, atol=0)


class StandardizedConvolutionCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261026)
        self.module = WSConv3d(4, 8, 3, padding=1, bias=True).eval()
        self.reference = OriginalWSConv3d(4, 8, 3, padding=1, bias=True).eval()
        self.reference.load_state_dict(self.module.state_dict())
        self.values = torch.randn(3, 1, 4, 4, 4, 4)

    def test_real_convolution_is_exact_and_standardizes_once(self):
        with torch.inference_mode(), mock.patch.object(
            self.module, "_standardized_weight", wraps=self.module._standardized_weight,
        ) as normalize:
            for value in self.values:
                torch.testing.assert_close(self.module(value), self.reference(value), rtol=0, atol=0)
            self.assertEqual(normalize.call_count, 1)
        self.assertEqual(set(self.module.state_dict()), {"weight", "bias"})

    def test_mutation_load_replacement_and_dtype_round_trip_refresh_weights(self):
        value = self.values[0]
        with torch.no_grad(), mock.patch.object(
            self.module, "_standardized_weight", wraps=self.module._standardized_weight,
        ) as normalize:
            self.module(value)
            self.module.weight.add_(torch.randn_like(self.module.weight) * 0.1)
            self.module(value)
            self.module.load_state_dict(self.reference.state_dict())
            self.module(value)
            self.module.weight = nn.Parameter(torch.randn_like(self.module.weight))
            self.module(value)
            self.module.half().float()
            self.assertIsNone(self.module._cached_standardized_weight)
            actual = self.module(value)
            self.reference.load_state_dict(self.module.state_dict())
            torch.testing.assert_close(actual, self.reference(value), rtol=0, atol=0)
            self.assertEqual(normalize.call_count, 5)

    def test_training_and_grad_enabled_eval_preserve_gradients(self):
        with torch.inference_mode():
            self.module(self.values[0])
        for training in (True, False):
            self.module.train(training)
            self.module.zero_grad()
            self.reference.zero_grad()
            value = self.values[0].clone().requires_grad_()
            other_value = value.detach().clone().requires_grad_()
            actual = self.module(value)
            expected = self.reference(other_value)
            actual.square().mean().backward()
            expected.square().mean().backward()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(self.module.weight.grad, self.reference.weight.grad, rtol=0, atol=0)
            torch.testing.assert_close(self.module.bias.grad, self.reference.bias.grad, rtol=0, atol=0)
            torch.testing.assert_close(value.grad, other_value.grad, rtol=0, atol=0)
            self.assertIsNone(self.module._cached_standardized_weight)

    def test_training_without_grad_does_not_cache(self):
        self.module.train()
        with torch.no_grad(), mock.patch.object(
            self.module, "_standardized_weight", wraps=self.module._standardized_weight,
        ) as normalize:
            self.module(self.values[0])
            self.module(self.values[0])
            self.assertEqual(normalize.call_count, 2)
            self.assertIsNone(self.module._cached_standardized_weight)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available; no GPU convolution equivalence claim")
    def test_cuda_convolution_and_cpu_offload(self):
        self.module.cuda()
        self.reference.cuda()
        with torch.inference_mode():
            for value in self.values:
                value = value.cuda()
                torch.testing.assert_close(self.module(value), self.reference(value), rtol=0, atol=0)
            self.module.cpu()
            self.assertIsNone(self.module._cached_standardized_weight)
            self.reference.cpu()
            torch.testing.assert_close(self.module(self.values[0]), self.reference(self.values[0]), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
