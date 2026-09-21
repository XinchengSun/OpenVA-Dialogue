"""Exercise the real motion worker on CPU with a small deterministic model.

The model checks window/state bookkeeping, not perceptual or numerical fidelity:
real wav2vec is noncausal and its features change when old context is removed.
"""
import ast
import contextlib
import io
import os
from pathlib import Path
import queue
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


SOURCE = Path(__file__).resolve().parents[1] / "dual_gpu_mic_realtime_ui_v8_playbuffer.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
NAMESPACE = {"queue": queue}
exec(compile(ast.Module(body=[node for node in TREE.body if isinstance(node, ast.FunctionDef)
                             and node.name in ("motion_worker", "put_drop_old")], type_ignores=[]),
             str(SOURCE), "exec"), NAMESPACE)


class RecordingModel:
    cfg = SimpleNamespace(cbh_window_length=12)
    inpainting_length = 10
    cuda_graph_enabled = True
    samples_per_frame = 40  # 1 kHz / 25 fps; no real audio encoder is used.

    def __init__(self):
        self.feature_samples = []
        self.other_feature_samples = []
        self.calls = []

    def features(self, audio, total_len):
        # Pointwise features let the tests check exact indexing while making
        # no claim that full and cropped wav2vec features would be equal.
        return audio.reshape(1, total_len, self.samples_per_frame).mean(-1, keepdim=True)

    def get_audio2face_fea(self, audio, unused, total_len):
        self.feature_samples.append(audio.shape[-1])
        return self.features(audio, total_len)

    def get_audio2face_fea_other(self, audio, unused, total_len):
        self.other_feature_samples.append(audio.shape[-1])
        return self.features(audio, total_len)

    def one_clip_only_inference_cuda_graph(self, **kwargs):
        self.calls.append(tuple(kwargs[key].clone() for key in (
            "audio_self", "audio_other", "past_audio_self", "past_audio_other",
            "past_motion", "per_compute_audio_feature", "per_compute_audio_other_feature")))
        return kwargs["past_motion"][:, -1:] + 1


def audio_message(samples, generation=1):
    return dict(samples=samples, samples_other=-samples, visible=True,
                generation=generation, mode="ASSISTANT_SPEAKING", turn_id=generation)


def run_worker(messages, max_sec=None, hop_ms=200):
    model = RecordingModel()
    app = SimpleNamespace(
        DEVICE="cpu", load_dystream_model=lambda: None, _dystream_model=model,
        _noise_scheduler=None, _dystream_ema=None,
        _dystream_cfg=SimpleNamespace(config={}),
        OmegaConf=SimpleNamespace(select=lambda _, key, default: {
            "model.audio_sr": 1000, "model.pose_fps": 25}[key]),
    )
    args = SimpleNamespace(motion_gpu=0, hop_ms=hop_ms, feature_lag_frames=2,
                           denoising_steps=5)
    anchor_q, audio_q, motion_q = queue.Queue(), queue.Queue(), queue.Queue()
    anchor_q.put(np.zeros(4, dtype=np.float32))
    for message in messages:
        audio_q.put(message)
    audio_q.put(None)
    output = io.StringIO()
    with patch.dict(sys.modules, {"app": app}), patch.dict(os.environ, {
        "DYSTREAM_AUDIO_HISTORY_KEEP_SEC": "4", "DYSTREAM_LISTENING_CONTROLLER": "0",
        "DYSTREAM_LIVE_LOG_EVERY": "100000",
    }), patch("torch.cuda.is_available", return_value=False), contextlib.redirect_stdout(output):
        if max_sec is None:
            os.environ.pop("DYSTREAM_AUDIO_HISTORY_MAX_SEC", None)
        else:
            os.environ["DYSTREAM_AUDIO_HISTORY_MAX_SEC"] = str(max_sec)
        NAMESPACE["motion_worker"](args, anchor_q, audio_q, motion_q)
    items = []
    while not motion_q.empty():
        items.append(motion_q.get_nowait())
    return model, items, output.getvalue()


class AudioHistoryTests(unittest.TestCase):
    def frames(self, items):
        return np.concatenate([item["motion_np"] for item in items if "motion_np" in item], axis=1)

    def assert_same_stream(self, baseline, bounded):
        before, before_items, _ = baseline
        after, after_items, logs = bounded
        np.testing.assert_array_equal(self.frames(before_items), self.frames(after_items))
        self.assertEqual(len(before.calls), len(after.calls))
        for frame_idx, (old_call, new_call) in enumerate(zip(before.calls, after.calls)):
            for field_idx, (old, new) in enumerate(zip(old_call, new_call)):
                self.assertTrue(torch.equal(old, new), (frame_idx, field_idx))
        for item in after_items:
            if "motion_np" in item:
                self.assertEqual(item["produced_end"] - item["produced_start"], item["produced"])
        self.assertEqual(after.feature_samples, after.other_feature_samples)
        # The cap covers real audio; the model still receives its prefix silence.
        self.assertLessEqual(max(after.feature_samples), 8000 + 10 * 40)
        self.assertIn("trigger=audio_cap", logs)
        self.assertIn("noncausal_context_changes=1", logs)

    def test_default_keeps_cumulative_audio(self):
        samples = np.arange(20000, dtype=np.float32)
        model, items, logs = run_worker([audio_message(samples)])
        self.assertEqual(max(model.feature_samples), 20000 + 10 * 40)
        self.assertEqual(self.frames(items).shape[1], 500 - 3)
        self.assertNotIn("compact history", logs)
        self.assertNotIn("[MOTION_HISTORY]", logs)

    def test_long_response_bounded_each_hop_without_losing_outputs(self):
        # One queue item contains 60 s: checking only when reading queue items
        # would miss this case. Compaction must happen inside the per-hop loop.
        messages = [audio_message(np.arange(60000, dtype=np.float32))]
        baseline = run_worker(messages)
        bounded = run_worker(messages, max_sec=8)
        self.assert_same_stream(baseline, bounded)
        self.assertEqual(len(bounded[0].calls), 1500 - 3)
        self.assertGreaterEqual(bounded[2].count("trigger=audio_cap"), 10)

    def test_fractional_frame_hops_preserve_phase_and_output_count(self):
        # 170 ms is 4.25 pose frames; keep the fractional frame through compaction.
        messages = [audio_message(np.arange(170 * 301, dtype=np.float32))]
        self.assert_same_stream(run_worker(messages, hop_ms=170),
                                run_worker(messages, max_sec=8, hop_ms=170))

    def test_explicit_compact_preserves_pending_partial_hop(self):
        samples = np.arange(25017, dtype=np.float32)
        plain = [audio_message(samples[:10777]), audio_message(samples[10777:])]
        compacted = [plain[0], {"type": "compact", "generation": 1}, plain[1]]
        bounded = run_worker(compacted, max_sec=8)
        self.assert_same_stream(run_worker(plain), bounded)
        self.assertIn("trigger=requested", bounded[2])

    def test_reset_preserves_recurrent_state_and_discards_only_pending(self):
        samples = np.arange(28013, dtype=np.float32)
        messages = [audio_message(samples[:14057]), {"type": "reset", "generation": 2},
                    audio_message(samples[14057:], generation=2)]
        baseline, bounded = run_worker(messages), run_worker(messages, max_sec=8)
        self.assert_same_stream(baseline, bounded)
        self.assertEqual([item for item in bounded[1] if item.get("type") == "reset"],
                         [{"type": "reset", "generation": 2}])
        self.assertEqual(bounded[1][-1]["generation"], 2)
        self.assertEqual(len(bounded[0].calls), (14000 + 13800) // 40 - 3)

    def test_invalid_caps_fail_instead_of_silently_exceeding_bound(self):
        for value in (-1, "nan", "inf", 4, 4.2):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "DYSTREAM_AUDIO_HISTORY_MAX_SEC"):
                run_worker([], max_sec=value)


if __name__ == "__main__":
    unittest.main()
