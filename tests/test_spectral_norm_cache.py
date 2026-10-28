import unittest

import torch
from torch import nn
from torch.nn.utils import spectral_norm

from inference_memory import cache_renderer_spectral_norm


def network(device="cpu"):
    return nn.Sequential(spectral_norm(nn.Conv2d(3, 6, 3, padding=1)), nn.Tanh(),
                         spectral_norm(nn.Conv2d(6, 3, 1))).eval().to(device)


class SpectralNormCacheTests(unittest.TestCase):
    def test_cpu_and_training_are_untouched(self):
        model = network()
        self.assertFalse(cache_renderer_spectral_norm(torch, model)["enabled"])
        self.assertFalse(cache_renderer_spectral_norm(torch, model.train())["enabled"])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.no_grad()
    def test_exact_reuse_checkpoint_and_parameter_buffer_invalidation(self):
        model = network("cuda")
        reference = network("cuda")
        reference.load_state_dict(model.state_dict())
        inputs = torch.randn(4, 1, 3, 8, 8, device="cuda")
        state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        report = cache_renderer_spectral_norm(torch, model)
        self.assertEqual(report["modules_cached"], 2)
        for value in inputs:
            torch.testing.assert_close(model(value), reference(value), rtol=0, atol=0)
        weight = model[0].weight
        model(inputs[0])
        self.assertIs(model[0].weight, weight)
        self.assertEqual(model[0].weight_orig.device.type, "cpu")
        self.assertEqual(model[0].weight_u.device.type, "cpu")
        self.assertEqual(tuple(model.state_dict()), tuple(state))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value.cpu(), state[key], rtol=0, atol=0)
        for suffix in ("_orig", "_u", "_v"):
            getattr(model[0], "weight" + suffix).add_(0.01)
            reference.load_state_dict(model.state_dict())
            torch.testing.assert_close(model(inputs[0]), reference(inputs[0]), rtol=0, atol=0)
        changed = network("cuda")
        model.load_state_dict(changed.state_dict())
        torch.testing.assert_close(model(inputs[0]), changed(inputs[0]), rtol=0, atol=0)
        model[0].weight_orig = nn.Parameter(torch.randn_like(model[0].weight_orig))
        reference.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(inputs[0]), reference(inputs[0]), rtol=0, atol=0)
        self.assertEqual(cache_renderer_spectral_norm(torch, model)["reason"], "already_cached")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.no_grad()
    def test_device_and_dtype_roundtrip_refreshes_weight(self):
        model = network("cuda")
        cache_renderer_spectral_norm(torch, model)
        value = torch.randn(1, 3, 8, 8, device="cuda")
        model(value)
        cached = model[0].weight
        model.half().float()
        reference = network("cuda")
        reference.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(value), reference(value), rtol=0, atol=0)
        self.assertIsNot(cached, model[0].weight)
        model.cpu()
        cpu_reference = network()
        cpu_reference.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(value.cpu()), cpu_reference(value.cpu()), rtol=0, atol=0)
        model.cuda()
        torch.testing.assert_close(model(value), reference(value), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_training_and_grad_eval_preserve_outputs_gradients_and_power_iteration(self):
        for training in (True, False):
            model, reference = network("cuda"), network("cuda")
            reference.load_state_dict(model.state_dict())
            cache_renderer_spectral_norm(torch, model)
            model.train(training)
            reference.train(training)
            value = torch.randn(1, 3, 8, 8, device="cuda", requires_grad=True)
            other = value.detach().clone().requires_grad_()
            actual, expected = model(value), reference(other)
            actual.square().mean().backward()
            expected.square().mean().backward()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(value.grad, other.grad, rtol=0, atol=0)
            for a, b in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            for key, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, reference.state_dict()[key], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_shared_or_unversioned_state_is_rejected_before_mutation(self):
        model = network("cuda")
        model[2].weight_u = model[0].weight_u
        self.assertFalse(cache_renderer_spectral_norm(torch, model)["enabled"])
        self.assertTrue(model[0].weight_orig.is_cuda)
        with torch.inference_mode():
            other = network("cuda")
        self.assertFalse(cache_renderer_spectral_norm(torch, other)["enabled"])


if __name__ == "__main__":
    unittest.main()
