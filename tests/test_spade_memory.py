import ast
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F


PATH = Path(__file__).resolve().parents[1] / "tools/visualization_0416/utils/model_0506/model/head_animation/LIA_3d/util.py"
node = next(n for n in ast.parse(PATH.read_text(encoding="utf-8")).body
            if isinstance(n, ast.ClassDef) and n.name == "SPADE")
namespace = dict(torch=torch, nn=nn, F=F)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(PATH), "exec"), namespace)
SPADE = namespace["SPADE"]


def original(module, x, segmap):
    normalized = module.param_free_norm(x)
    segmap = F.interpolate(segmap, size=x.size()[2:], mode="nearest")
    activation = module.mlp_shared(segmap)
    gamma, beta = module.mlp_gamma(activation), module.mlp_beta(activation)
    return normalized * (1 + gamma) + beta


class SpadeMemoryTests(unittest.TestCase):
    def check_inference(self, device):
        torch.manual_seed(417)
        module = SPADE(8, 4).eval().to(device)
        x = torch.randn(2, 8, 16, 16, device=device)
        segment = torch.randn(2, 4, 8, 8, device=device)
        before_x, before_segment = x.clone(), segment.clone()
        with torch.inference_mode():
            torch.testing.assert_close(module(x, segment), original(module, x, segment), rtol=0, atol=0)
        torch.testing.assert_close(x, before_x, rtol=0, atol=0)
        torch.testing.assert_close(segment, before_segment, rtol=0, atol=0)

    def test_cpu_inference_exact_and_inputs_unchanged(self):
        self.check_inference("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_inference_exact_and_inputs_unchanged(self):
        self.check_inference("cuda")

    def test_training_and_grad_enabled_eval_preserve_gradients(self):
        torch.manual_seed(418)
        for training in (True, False):
            module = SPADE(8, 4).train(training)
            reference = SPADE(8, 4).train(training)
            reference.load_state_dict(module.state_dict())
            x, segment = torch.randn(2, 8, 8, 8, requires_grad=True), torch.randn(2, 4, 4, 4, requires_grad=True)
            other_x, other_segment = x.detach().clone().requires_grad_(), segment.detach().clone().requires_grad_()
            actual, expected = module(x, segment), original(reference, other_x, other_segment)
            actual.square().mean().backward()
            expected.square().mean().backward()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for a, b in zip(module.parameters(), reference.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            torch.testing.assert_close(x.grad, other_x.grad, rtol=0, atol=0)
            torch.testing.assert_close(segment.grad, other_segment.grad, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
