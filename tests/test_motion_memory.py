import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from inference_memory import MemoryRecorder, cuda_storage_inventory, prepare_motion_inference_memory, place_motion_for_inference
from tests.motion_memory_fixture import make_motion_core


class MotionMemoryTests(unittest.TestCase):
    def test_training_is_rejected_before_mutation(self):
        model, _ = make_motion_core(hidden_size=24, layers=2, max_seq_len=32)
        model.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            prepare_motion_inference_memory(torch, model)

    def test_disabled_profiler_does_not_inspect_models_or_initialize_cuda(self):
        with patch.dict(os.environ, {"OPENVA_MEMORY_PROFILE_DIR": ""}), \
             patch.object(torch.cuda, "synchronize", side_effect=AssertionError("CUDA touched")):
            recorder = MemoryRecorder("test", torch, "cuda")
            self.assertIsNone(recorder.record("unused", {"invalid": object()}))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.inference_mode()
    def test_selective_loading_keeps_values_and_buffer_aliases(self):
        with torch.inference_mode(False):
            model, _ = make_motion_core(hidden_size=24, layers=2, max_seq_len=32)
            expected = {name: value.clone() for name, value in model.state_dict().items()}
            place_motion_for_inference(torch, model, "cuda")
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value.cpu(), expected[name], rtol=0, atol=0)
        for block in model.blocks:
            self.assertTrue(all(p.device.type == "cpu" for p in block.cross_attn_rope.parameters()))
        self.assertIs(model.blocks[0].self_attn_rope.rope.cos_angles,
                      model.blocks[1].cross_attn_anchor.rope.cos_angles)
        self.assertIs(model.blocks[0].self_attn_pos.pe.pe, model.blocks[1].self_attn_pos.pe.pe)

    def test_already_captured_model_cannot_change_storage(self):
        model, _ = make_motion_core(hidden_size=24, layers=2, max_seq_len=32)
        model.cuda_graph_enabled = True
        with self.assertRaisesRegex(ValueError, "before CUDA graph"):
            prepare_motion_inference_memory(torch, model)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.inference_mode()
    def test_outputs_and_checkpoint_keys_preserved_and_unused_modules_never_called(self):
        # Construct outside inference mode so buffers have normal version counters.
        with torch.inference_mode(False):
            model, _ = make_motion_core(hidden_size=24, layers=2, max_seq_len=32)
            model.cuda()
        x, audio = torch.randn(4, 12, 24, device="cuda"), torch.randn(4, 12, 24, device="cuda")
        anchor = torch.randn(4, 1, 24, device="cuda")
        mask = torch.triu(torch.full((12, 12), float("-inf"), device="cuda"), diagonal=1)
        def forward():
            result = x
            for block in model.blocks:
                result = block(result, audio, anchor, mask, mask)
            return model.output_proj(model.output_norm(result[:, -1:]))
        expected = forward().clone()
        noisy = torch.randn(4, 1, 512, device="cuda")
        condition = torch.randn_like(noisy)
        time_embedding = torch.randn(4, 1, 24, device="cuda")
        expected_diffusion = model.diffusion_head(noisy, condition, time_embedding).clone()
        keys = tuple(model.state_dict())
        before = cuda_storage_inventory(torch, {"model": model})["unique_cuda_storage_bytes"]
        report = prepare_motion_inference_memory(torch, model)
        after = cuda_storage_inventory(torch, {"model": model})["unique_cuda_storage_bytes"]
        self.assertGreater(before, after)
        self.assertEqual(tuple(model.state_dict()), keys)
        self.assertGreater(report["position_buffers_shared"], 0)
        def forbidden(*_):
            raise AssertionError("an offloaded module was used")
        hooks = []
        for block in model.blocks:
            for module in (block.cross_attn_rope, block.norm2):
                self.assertTrue(all(p.device.type == "cpu" for p in module.parameters()))
                hooks.append(module.register_forward_pre_hook(forbidden))
        for module in (model.diffusion_head.past_proj, model.diffusion_head.anchor_proj):
            hooks.append(module.register_forward_pre_hook(forbidden))
        try:
            torch.testing.assert_close(forward(), expected, rtol=0, atol=0)
            torch.testing.assert_close(model.diffusion_head(noisy, condition, time_embedding),
                                       expected_diffusion, rtol=0, atol=0)
            second = prepare_motion_inference_memory(torch, model)
            self.assertEqual(second["unused_cuda_tensor_bytes_offloaded"], 0)
            self.assertEqual(second["position_buffers_shared"], 0)
        finally:
            for hook in hooks:
                hook.remove()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.no_grad()
    def test_different_position_tables_are_not_merged(self):
        model, _ = make_motion_core(hidden_size=24, layers=2, max_seq_len=32)
        model.cuda()
        table = model.blocks[1].self_attn_pos.pe.pe
        table.add_(0.25)
        original = table.clone()
        prepare_motion_inference_memory(torch, model)
        self.assertIsNot(model.blocks[0].self_attn_pos.pe.pe, model.blocks[1].self_attn_pos.pe.pe)
        torch.testing.assert_close(model.blocks[1].self_attn_pos.pe.pe, original, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_profiler_counts_aliases_and_unregistered_caches_once(self):
        module = torch.nn.Linear(4, 4).cuda()
        module.cache = torch.ones(16, device="cuda")
        module.alias = module.cache.view(4, 4)
        inventory = cuda_storage_inventory(torch, {"one": module, "two": module})
        self.assertEqual(inventory["unique_cuda_storage_bytes"], 4 * (16 + 4 + 16))
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {"OPENVA_MEMORY_PROFILE_DIR": directory}):
            recorder = MemoryRecorder("test", torch, "cuda")
            result = recorder.record("loaded", {"model": module})
            self.assertGreaterEqual(result["allocated_bytes"], result["unique_cuda_storage_bytes"])
            self.assertEqual(len(list(Path(directory).glob("*.jsonl"))), 1)


if __name__ == "__main__":
    unittest.main()
