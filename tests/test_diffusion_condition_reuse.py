"""Real PyTorch head tests; no weights, Wav2Vec downloads or GPU required.

The AST loader isolates the production diffusion classes from model loading.
It uses the actual timm Mlp used by production, not a numerical model stub.
CUDA graph replay is tested separately when a CUDA device is available.
"""

import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
from timm.layers import Mlp


MODEL_PATH = (Path(__file__).resolve().parents[1] / "model" / "motion_generation"
              / "motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder.py")
SOURCE = ast.parse(MODEL_PATH.read_text(encoding="utf-8"))
NAMES = {"DiffusionBlock", "DiffusionHead", "CUDAGraphDiffusionHeadSingleStep"}
NAMESPACE = {"torch": torch, "nn": nn, "Mlp": Mlp}
exec(compile(ast.Module(body=[node for node in SOURCE.body
                             if isinstance(node, ast.ClassDef) and node.name in NAMES],
                        type_ignores=[]), str(MODEL_PATH), "exec"), NAMESPACE)
DiffusionHead = NAMESPACE["DiffusionHead"]
Runner = NAMESPACE["CUDAGraphDiffusionHeadSingleStep"]


def original_forward(head, noisy, gpt, temb):
    """Pre-change arithmetic, using the exact same real modules and weights."""
    x = head.noisy_proj(noisy)
    gpt_hidden = head.gpt_proj(gpt)
    for block in head.blocks:
        shift, scale, gate = block.adaLN_modulation1(temb).chunk(3, dim=-1)
        shift2, scale2, gate2 = block.adaLN_modulation2(gpt_hidden).chunk(3, dim=-1)
        normed = (block.norm1(x.float()) * (1 + scale) + shift).type_as(x)
        x = (x.float() + block.mlp1(normed).float() * gate).type_as(x)
        normed = (block.norm2(x.float()) * (1 + scale2) + shift2).type_as(x)
        x = (x.float() + block.mlp2(normed).float() * gate2).type_as(x)
    return head.output_proj(head.output_norm(x))


class DiffusionConditionReuseTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(93)
        self.head = DiffusionHead(face_dim=16, hidden_size=32, num_layers=6,
                                  num_heads=4, dropout=0.1).eval()

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    @torch.inference_mode()
    def test_full_ten_step_outputs_match_original_for_all_cfg_batches(self):
        # Full step outputs and evolving latents, using a fixed synthetic
        # Euler schedule; this does not claim full-model/checkpoint acceptance.
        for batch in (1, 4, 5, 8, 10):
            for length in (1, 3):
                gpt = torch.randn(batch, length, 16)
                baseline = torch.randn(batch, length, 16)
                candidate = baseline.clone()
                prepared = self.head.prepare_gpt_condition(gpt)
                time_batch = batch if batch in (8, 10) else 1
                times = torch.randn(10, time_batch, length, 32)
                for step in range(10):
                    expected = original_forward(self.head, baseline, gpt, times[step])
                    actual = self.head(candidate, None, times[step], gpt_condition=prepared)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    sigma = 1.0 - step / 10
                    baseline = baseline - 0.1 * ((baseline - expected) / (sigma + 1e-9))
                    candidate = candidate - 0.1 * ((candidate - actual) / (sigma + 1e-9))
                    torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)

    @torch.inference_mode()
    def test_condition_linear_calls_reduce_from_steps_to_one(self):
        modules = [self.head.gpt_proj] + [b.adaLN_modulation2[1] for b in self.head.blocks]
        counts = [0] * len(modules)
        def record(index):
            def hook(*_):
                counts[index] += 1
            return hook
        handles = [m.register_forward_hook(record(i)) for i, m in enumerate(modules)]
        gpt, noisy, temb = torch.randn(4, 1, 16), torch.randn(4, 1, 16), torch.randn(1, 1, 32)
        try:
            for _ in range(10):
                self.head(noisy, gpt, temb)
            self.assertEqual(counts, [10] * 7)
            counts[:] = [0] * 7
            prepared = self.head.prepare_gpt_condition(gpt)
            for _ in range(10):
                self.head(noisy, None, temb, gpt_condition=prepared)
            self.assertEqual(counts, [1] * 7)
        finally:
            for handle in handles:
                handle.remove()

    @torch.inference_mode()
    def test_new_frame_condition_is_fresh_and_has_no_persistent_model_state(self):
        keys = tuple(self.head.state_dict())
        gpt1, gpt2 = torch.randn(4, 1, 16), torch.randn(4, 1, 16)
        noisy, temb = torch.randn(4, 1, 16), torch.randn(1, 1, 32)
        first = self.head.prepare_gpt_condition(gpt1)
        second = self.head.prepare_gpt_condition(gpt2)
        actual = self.head(noisy, None, temb, gpt_condition=second)
        expected = original_forward(self.head, noisy, gpt2, temb)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        stale = self.head(noisy, None, temb, gpt_condition=first)
        self.assertFalse(torch.equal(actual, stale))
        self.assertEqual(tuple(self.head.state_dict()), keys)

    def test_default_training_forward_and_gradients_match_original(self):
        self.head.train()
        noisy = torch.randn(4, 2, 16, requires_grad=True)
        gpt = torch.randn(4, 2, 16, requires_grad=True)
        temb = torch.randn(1, 2, 32, requires_grad=True)
        torch.manual_seed(321)
        expected = original_forward(self.head, noisy, gpt, temb)
        expected.sum().backward()
        grads = [value.grad.clone() for value in (noisy, gpt, temb)]
        self.head.zero_grad(set_to_none=True)
        for value in (noisy, gpt, temb):
            value.grad = None
        torch.manual_seed(321)
        actual = self.head(noisy, gpt, temb)
        actual.sum().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for value, expected_grad in zip((noisy, gpt, temb), grads):
            torch.testing.assert_close(value.grad, expected_grad, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA graph requires actual CUDA device")
    @torch.inference_mode()
    def test_cuda_graph_updates_condition_for_every_frame(self):
        head = self.head.cuda()
        for batch, time_batch in ((4, 1), (5, 1), (8, 8), (10, 10)):
            runner = Runner(head, batch_size=batch, face_dim=16, hidden_size=32,
                            device="cuda", time_embedding_batch_size=time_batch,
                            precompute_condition=True)
            runner.warmup_and_capture()
            for _ in range(3):
                gpt = torch.randn(batch, 1, 16, device="cuda")
                noisy = torch.randn(batch, 1, 16, device="cuda")
                runner.prepare_gpt_condition(gpt)
                for _ in range(10):
                    temb = torch.randn(time_batch, 1, 32, device="cuda")
                    expected = head(noisy, gpt, temb)
                    actual = runner.forward(noisy, gpt, temb, reuse_gpt_condition=True)
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                    noisy = actual
                # An ordinary caller that does not opt into reuse remains safe.
                gpt = torch.randn_like(gpt)
                actual = runner.forward(noisy, gpt, temb)
                torch.testing.assert_close(actual, head(noisy, gpt, temb), rtol=1e-5, atol=1e-6)


class DiffusionConditionGraphSignatureTests(unittest.TestCase):
    def test_default_is_off_and_flag_toggle_recaptures(self):
        from tests.test_cfg_pruning import GraphSignatureTests
        fixture = GraphSignatureTests()
        fixture.setUp()
        with patch.dict(os.environ, {"DYSTREAM_PRUNE_CFG": "1"}):
            os.environ.pop("DYSTREAM_PRECOMPUTE_DIFFUSION_CONDITION", None)
            fixture.setup(fixture.model)
            first = fixture.model.cuda_graph_diffusion
            self.assertFalse(first.kwargs["precompute_condition"])
            os.environ["DYSTREAM_PRECOMPUTE_DIFFUSION_CONDITION"] = "1"
            fixture.setup(fixture.model)
            second = fixture.model.cuda_graph_diffusion
            self.assertIsNot(second, first)
            self.assertTrue(second.kwargs["precompute_condition"])
            fixture.setup(fixture.model)
            self.assertIs(fixture.model.cuda_graph_diffusion, second)
            os.environ["DYSTREAM_PRECOMPUTE_DIFFUSION_CONDITION"] = "0"
            fixture.setup(fixture.model)
            self.assertIsNot(fixture.model.cuda_graph_diffusion, second)
            self.assertFalse(fixture.model.cuda_graph_diffusion.kwargs["precompute_condition"])


if __name__ == "__main__":
    unittest.main()
