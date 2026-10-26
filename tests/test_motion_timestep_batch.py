"""Exercise the production timestep expression and actual WanTimeEmbedding."""

import ast
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile
from diffusers.models.embeddings import Timesteps, TimestepEmbedding


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model/motion_generation/motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder.py"
MODEL = ast.parse(MODEL_PATH.read_text(encoding="utf-8"))
MODEL_CLASS = next(node for node in MODEL.body if isinstance(node, ast.ClassDef) and node.name == "Audio2FaceGPT")
FUNCTIONS = []
for method in MODEL_CLASS.body:
    if isinstance(method, ast.FunctionDef) and method.name in {
        "one_clip_only_inference", "one_clip_only_inference_cuda_graph"
    }:
        assignment = next(node for node in ast.walk(method)
                          if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "t_batch"
                                  for target in node.targets))
        function = ast.parse("def timestep_batch(timestep, device, bs):\n    pass").body[0]
        function.body = [assignment, ast.Return(value=ast.Name(id="t_batch", ctx=ast.Load()))]
        namespace = {"torch": torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                     str(MODEL_PATH), "exec"), namespace)
        FUNCTIONS.append(namespace["timestep_batch"])

UTILS_PATH = ROOT / "model/motion_generation/motion_gen_utils_dev.py"
time_class = next(node for node in ast.parse(UTILS_PATH.read_text(encoding="utf-8")).body
                  if isinstance(node, ast.ClassDef) and node.name == "WanTimeEmbedding")
namespace = {"torch": torch, "nn": nn, "Timesteps": Timesteps, "TimestepEmbedding": TimestepEmbedding}
exec(compile(ast.Module(body=[time_class], type_ignores=[]), str(UTILS_PATH), "exec"), namespace)
WanTimeEmbedding = namespace["WanTimeEmbedding"]


class MotionTimestepBatchTests(unittest.TestCase):
    @torch.inference_mode()
    def test_fractional_values_batch_dtype_and_actual_time_embedding_match(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(105)
            embedding = WanTimeEmbedding(dim=768, time_freq_dim=768).eval()
            self.assertEqual(len(FUNCTIONS), 2)
            for make_batch in FUNCTIONS:
                for dtype in (torch.float32, torch.float64, torch.int64):
                    for value in (-999.9, -1.9, -0.1, 0, 0.1, 1.9, 999.9):
                        timestep = torch.tensor(value, dtype=dtype)
                        for batch in (1, 2, 8):
                            old = torch.full((batch,), timestep, device="cpu", dtype=torch.long)
                            new = make_batch(timestep, "cpu", batch)
                            self.assertEqual(new.dtype, torch.long)
                            self.assertEqual(new.device, old.device)
                            torch.testing.assert_close(new, old, rtol=0, atol=0)
                            torch.testing.assert_close(embedding(new), embedding(old), rtol=0, atol=0)
        finally:
            torch.set_num_threads(previous_threads)

    def test_scalar_read_is_removed_in_both_production_methods(self):
        timestep = torch.tensor(999.9)
        def scalar_reads(call):
            with profile(activities=[ProfilerActivity.CPU]) as profiler:
                for _ in range(10):
                    call()
            return sum(event.count for event in profiler.key_averages()
                       if event.key == "aten::_local_scalar_dense")
        old_count = scalar_reads(lambda: torch.full((2,), timestep, dtype=torch.long))
        self.assertEqual(old_count, 10)
        for make_batch in FUNCTIONS:
            new_count = scalar_reads(lambda: make_batch(timestep, "cpu", 2))
            self.assertEqual(new_count, 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    @torch.inference_mode()
    def test_cuda_values_and_actual_time_embedding_match(self):
        embedding = WanTimeEmbedding(dim=768, time_freq_dim=768).eval().cuda()
        for make_batch in FUNCTIONS:
            for value in (999.9, 500.5, 0.0):
                timestep = torch.tensor(value, device="cuda")
                for batch in (1, 2, 8):
                    old = torch.full((batch,), timestep, device="cuda", dtype=torch.long)
                    new = make_batch(timestep, "cuda", batch)
                    torch.testing.assert_close(new, old, rtol=0, atol=0)
                    torch.testing.assert_close(embedding(new), embedding(old), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
