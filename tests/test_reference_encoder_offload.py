"""Reference-encoder lifetime, real weights and CUDA allocator regression checks."""

import contextlib
import io
import os
import queue
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import numpy as np
from PIL import Image

import dual_gpu_mic_realtime_ui_v8_playbuffer as runtime

from tests.reference_encoder_fixture import load_encoders, load_offload, tensor_inventory


offload = load_offload()


class ReferenceEncoderOffloadTests(unittest.TestCase):
    def test_real_worker_keeps_frames_after_encoders_become_unavailable(self):
        def run(release):
            torch.manual_seed(57)
            class Encoder(torch.nn.Conv2d):
                calls = 0
                offloaded = False
                def forward(self, value):
                    if self.offloaded:
                        raise AssertionError("encoder called again after offload")
                    if release and self is face and not motion.offloaded:
                        raise AssertionError("motion encoder still resident during face extraction")
                    self.calls += 1
                    return super().forward(value)
                def cpu(self):
                    self.offloaded = True
                    return super().cpu()
            class Flow(torch.nn.Module):
                def forward(self, source, motion):
                    return source + motion
            class Generator(torch.nn.Module):
                def forward(self, target, feature):
                    return torch.tanh(feature + target[:, :3, None, None])
            face, motion = Encoder(3, 3, 1).eval(), Encoder(3, 3, 1).eval()
            def transform(image):
                return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
            context = dict(face_encoder=face, motion_encoder=motion,
                           flow_estimator=Flow(), face_generator=Generator(), transform=transform)
            def process_image(image):
                with torch.no_grad():
                    anchor = motion(transform(image)[None]).mean((2, 3)).cpu()
                return image, image, anchor
            app = SimpleNamespace(DEVICE="cpu", _vis_ctx=context,
                                  load_visualization_model=lambda: None, process_image=process_image)
            input_queue, frames, anchors = queue.Queue(), queue.Queue(), queue.Queue()
            for step in range(2):
                input_queue.put(dict(motion_np=np.full((1, 4, 3), step * 0.1, np.float32),
                                     generation=0, produced_start=step * 4, produced_end=step * 4 + 4,
                                     mode="ASSISTANT_ACTIVE", turn_id=1, visible=True,
                                     step=step, motion_total=0.0))
            input_queue.put(None)
            with tempfile.TemporaryDirectory() as directory:
                image_path = Path(directory) / "avatar.png"
                Image.new("RGB", (4, 4), (100, 150, 200)).save(image_path)
                release_function = offload if release else lambda *_: {}
                with patch.dict(sys.modules, {"app": app}), \
                     patch.object(runtime, "get_paths", return_value=str(image_path)), \
                     patch.object(runtime, "offload_reference_encoders", side_effect=release_function), \
                     patch.object(torch.cuda, "is_available", return_value=False), \
                     patch.dict(os.environ, {"PIPE_FRAME_STRIDE": "1", "DYSTREAM_LISTENING_FACE_CONTROL": "0"}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    runtime.render_worker(SimpleNamespace(render_gpu=0, sample=0, hop_ms=200),
                                          anchors, input_queue, frames)
            return np.stack([frames.get_nowait()["frame"] for _ in range(frames.qsize())]), face, motion
        baseline, _, _ = run(False)
        candidate, face, motion = run(True)
        self.assertEqual(candidate.shape[0], 8)
        np.testing.assert_array_equal(candidate, baseline)
        self.assertTrue(face.offloaded and motion.offloaded)
        self.assertEqual((face.calls, motion.calls), (1, 1))

    def test_cpu_encoders_do_not_initialize_cuda_or_change_weights(self):
        encoders = load_encoders()
        # Small sample values plus full inventories ensure dtype/layout survive;
        # actual device transfers and full numerical equality are CUDA tests.
        before = {name: tensor_inventory(module) for name, module in encoders.items()}
        values = {name: next(module.parameters()).detach().clone()
                  for name, module in encoders.items()}
        with patch.object(torch.cuda, "synchronize", side_effect=AssertionError("unexpected CUDA")), \
             patch.object(torch.cuda, "empty_cache", side_effect=AssertionError("unexpected CUDA")):
            result = offload(torch, encoders)
            offload(torch, encoders)  # Releasing an already-CPU module is safe.
        self.assertEqual(result["encoder_cuda_tensor_bytes"], {name: 0 for name in encoders})
        for name, module in encoders.items():
            self.assertEqual(tensor_inventory(module), before[name])
            self.assertFalse(module.training)
            torch.testing.assert_close(next(module.parameters()), values[name], rtol=0, atol=0)

    def test_only_reference_encoders_are_moved(self):
        calls = []
        class Tracked(torch.nn.Linear):
            def cpu(self):
                calls.append(self)
                return super().cpu()
        face, motion = Tracked(3, 4), Tracked(4, 5)
        generator = Tracked(5, 6)
        context = dict(face_encoder=face, motion_encoder=motion, face_generator=generator)
        cached = torch.randn(1, 32, 16, 64, 64)
        original = cached.clone()
        context["face_feat"] = cached
        offload(torch, context)
        self.assertEqual(calls, [face, motion])
        self.assertIs(context["face_feat"], cached)
        torch.testing.assert_close(cached, original, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires actual CUDA allocator")
    @torch.no_grad()
    def test_real_cuda_storage_freed_and_cached_outputs_unchanged(self):
        torch.manual_seed(20261028)
        encoders = load_encoders()
        for module in encoders.values():
            module.cuda()
        expected_bytes = sum(tensor_inventory(m)["total_tensor_bytes"] for m in encoders.values())
        image = torch.randn(1, 3, 512, 512, device="cuda")
        face = encoders["face_encoder"](image).detach()
        anchor = encoders["motion_encoder"](image)[0].detach()
        expected_face = face.cpu()
        expected_anchor = anchor.cpu()
        result = offload(torch, encoders)
        self.assertGreaterEqual(result["allocated_freed_bytes"], expected_bytes)
        self.assertEqual(sum(result["encoder_cuda_tensor_bytes"].values()), expected_bytes)
        for module in encoders.values():
            self.assertTrue(all(t.device.type == "cpu" for t in module.parameters()))
            self.assertTrue(all(t.device.type == "cpu" for t in module.buffers()))
        self.assertEqual(face.device.type, "cuda")
        self.assertEqual(anchor.device.type, "cuda")
        torch.testing.assert_close(face.cpu(), expected_face, rtol=0, atol=0)
        torch.testing.assert_close(anchor.cpu(), expected_anchor, rtol=0, atol=0)
        # Moving unchanged weights back must also recover identical encoder outputs.
        for module in encoders.values():
            module.cuda()
        torch.testing.assert_close(encoders["face_encoder"](image), face, rtol=0, atol=0)
        torch.testing.assert_close(encoders["motion_encoder"](image)[0], anchor, rtol=0, atol=0)
        offload(torch, encoders)


if __name__ == "__main__":
    unittest.main()
