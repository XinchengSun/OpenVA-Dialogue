"""CPU-only contracts for optional CFG pruning; no checkpoint/PyTorch needed."""

import ast
import itertools
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch


MODEL_PATH = (Path(__file__).resolve().parents[1] / "model" / "motion_generation"
              / "motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder.py")
SOURCE = ast.parse(MODEL_PATH.read_text(encoding="utf-8"))
HELPER_NAMES = {"_cfg_branch_plan", "_combine_cfg_predictions"}
NAMESPACE = {"math": math, "os": os}
HELPERS = [node for node in SOURCE.body
           if isinstance(node, ast.FunctionDef) and node.name in HELPER_NAMES]
MODEL_CLASS = next(node for node in SOURCE.body
                   if isinstance(node, ast.ClassDef) and node.name == "Audio2FaceGPT")
SETUP = next(node for node in MODEL_CLASS.body
             if isinstance(node, ast.FunctionDef) and node.name == "setup_cuda_graphs")
exec(compile(ast.Module(body=HELPERS + [SETUP], type_ignores=[]),
             str(MODEL_PATH), "exec"), NAMESPACE)
plan = NAMESPACE["_cfg_branch_plan"]
combine = NAMESPACE["_combine_cfg_predictions"]


def baseline(predictions, weights):
    uncond, anchor, audio, other, combined = predictions
    wa, wo, wr, wall = weights
    return (uncond + wa * (audio - uncond) + wo * (other - uncond)
            + wr * (anchor - uncond) + wall * (combined - uncond))


class CFGPruningTests(unittest.TestCase):
    def test_default_keeps_all_five_and_original_arithmetic(self):
        weights = (0.5, 0.5, 0.0, 1.0)
        predictions = (0.123456789, 3.4, -1.234, 5.678, 9.01)
        branches = plan(*weights)
        self.assertEqual(branches, (0, 1, 2, 3, 4))
        self.assertEqual(combine(predictions, branches, *weights),
                         baseline(predictions, weights))

    def test_runtime_defaults_drop_anchor_but_keep_negative_uncond(self):
        self.assertEqual(plan(0.5, 0.5, 0.0, 1.0, prune=True), (0, 2, 3, 4))

    def test_only_combined_condition_needs_no_unconditional_pass(self):
        branches = plan(0.0, 0.0, 0.0, 1.0, prune=True)
        self.assertEqual(branches, (4,))
        self.assertEqual(combine([17.0], branches, 0.0, 0.0, 0.0, 1.0), 17.0)

    def test_all_zero_guidance_keeps_unconditional(self):
        branches = plan(0.0, 0.0, 0.0, 0.0, prune=True)
        self.assertEqual(branches, (0,))
        self.assertEqual(combine([11.0], branches, 0.0, 0.0, 0.0, 0.0), 11.0)

    def test_tiny_nonzero_guidance_and_unconditional_terms_are_preserved(self):
        self.assertIn(3, plan(0.5, 1e-15, 0.0, 1.0, prune=True))
        self.assertIn(0, plan(0.0, 0.0, 0.0, 1.0 - 1e-14, prune=True))

    def test_nonfinite_values_fall_back_to_original_five_branches(self):
        for value in (math.nan, math.inf, -math.inf):
            self.assertEqual(plan(value, 0.5, 0.0, 1.0, prune=True),
                             (0, 1, 2, 3, 4))

    def test_guided_predictions_match_across_weight_combinations(self):
        rng = random.Random(20260921)
        for weights in itertools.product((-0.5, 0.0, 0.5, 1.0, 2.0), repeat=4):
            predictions = tuple(rng.uniform(-10.0, 10.0) for _ in range(5))
            branches = plan(*weights, prune=True)
            result = combine([predictions[index] for index in branches], branches, *weights)
            self.assertTrue(math.isclose(result, baseline(predictions, weights),
                                         rel_tol=1e-12, abs_tol=1e-12), weights)


class _Runner:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def warmup_and_capture(self):
        pass


class GraphSignatureTests(unittest.TestCase):
    def setUp(self):
        NAMESPACE.update(torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
                         CUDAGraphGPTRunner=_Runner,
                         CUDAGraphDiffusionHeadSingleStep=_Runner)
        parameter = SimpleNamespace(device="cuda:0", dtype="float32")
        self.model = SimpleNamespace(
            cfg_audio=0.5, cfg_audio_other=0.5, cfg_anchor=0.0, cfg_all=1.0,
            blocks=(), output_norm=None, output_proj=None, hidden_size=768,
            diffusion_head=None, face_dim=512, parameters=lambda: iter((parameter,)))
        self.setup = NAMESPACE["setup_cuda_graphs"]

    def test_switch_default_pruned_and_back_rebuilds_correct_batch(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DYSTREAM_PRUNE_CFG", None)
            self.setup(self.model)
            first = self.model.cuda_graph_gpt
            self.assertEqual(first.kwargs["batch_size"], 5)
            self.setup(self.model)
            self.assertIs(self.model.cuda_graph_gpt, first)
            os.environ["DYSTREAM_PRUNE_CFG"] = "1"
            self.setup(self.model)
            self.assertIsNot(self.model.cuda_graph_gpt, first)
            self.assertEqual(self.model.cuda_graph_gpt.kwargs["batch_size"], 4)
            os.environ["DYSTREAM_PRUNE_CFG"] = "0"
            self.setup(self.model)
            self.assertEqual(self.model.cuda_graph_gpt.kwargs["batch_size"], 5)

    def test_weight_changes_rebuild_even_when_branch_count_stays_equal(self):
        with patch.dict(os.environ, {"DYSTREAM_PRUNE_CFG": "1"}):
            self.setup(self.model)
            first = self.model.cuda_graph_gpt
            self.model.cfg_audio = 0.75
            self.setup(self.model)
            self.assertIsNot(self.model.cuda_graph_gpt, first)
            self.assertEqual(self.model.cuda_graph_gpt.kwargs["batch_size"], 4)
            self.model.cfg_anchor = 0.25
            self.setup(self.model)
            self.assertEqual(self.model.cuda_graph_gpt.kwargs["batch_size"], 5)

    def test_sample_batch_is_multiplied_by_active_branches(self):
        with patch.dict(os.environ, {"DYSTREAM_PRUNE_CFG": "1"}):
            self.setup(self.model, batch_size=2)
            self.assertEqual(self.model.cuda_graph_gpt.kwargs["batch_size"], 8)
            self.assertEqual(self.model.cuda_graph_diffusion.kwargs["batch_size"], 8)
            self.assertEqual(self.model.cuda_graph_diffusion.kwargs["time_embedding_batch_size"], 8)
            self.setup(self.model, batch_size=1)
            self.assertEqual(self.model.cuda_graph_gpt.kwargs["batch_size"], 4)
            self.assertEqual(self.model.cuda_graph_diffusion.kwargs["time_embedding_batch_size"], 1)


if __name__ == "__main__":
    unittest.main()
