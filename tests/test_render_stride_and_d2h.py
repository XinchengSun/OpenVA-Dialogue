from __future__ import annotations

import unittest

import numpy as np
import torch

from dual_gpu_mic_realtime_ui_v8_playbuffer import (
    expand_rendered_frames,
    render_chunk_fp32_loop,
    select_render_offsets,
)


def _legacy_render(
    motion_chunk: torch.Tensor,
    render_src_motion: torch.Tensor,
) -> np.ndarray:
    motion_latents = motion_chunk.squeeze(0).float()
    src_motion = render_src_motion.squeeze(0).float()
    frames = []
    with torch.inference_mode():
        for index in range(motion_latents.shape[0]):
            target = src_motion + motion_latents[index : index + 1]
            reconstruction = target[:, :3].reshape(1, 3, 1, 1).expand(-1, -1, 2, 2)
            video_u8 = (
                ((reconstruction.float() + 1) / 2 * 255)
                .clamp(0, 255)
                .to(torch.uint8)
            )
            frames.append(
                video_u8.permute(0, 2, 3, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )
    return np.concatenate(frames, axis=0)


class _Flow:
    def __call__(self, source: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        return source + motion


class _Generator:
    def __call__(self, target: torch.Tensor, _face_feat) -> torch.Tensor:
        return target[:, :3].reshape(1, 3, 1, 1).expand(-1, -1, 2, 2)


class RenderStrideAndD2HTests(unittest.TestCase):
    def test_batched_d2h_is_byte_equivalent_to_legacy_per_frame_copy(self):
        motion = torch.tensor(
            [[[-0.75, -0.25, 0.25], [-0.5, 0.0, 0.5], [-0.25, 0.25, 0.75]]],
            dtype=torch.float32,
        )
        source = torch.tensor([[[0.10, -0.10, 0.05]]], dtype=torch.float32)

        expected = _legacy_render(motion, source)
        actual = render_chunk_fp32_loop(
            torch,
            np,
            motion,
            source,
            None,
            _Flow(),
            _Generator(),
            torch.device("cpu"),
        )

        self.assertEqual(actual.dtype, np.uint8)
        self.assertEqual(actual.shape, (3, 2, 2, 3))
        np.testing.assert_array_equal(actual, expected)

    def test_global_stride_phase_survives_chunk_boundaries(self):
        self.assertEqual(select_render_offsets(0, 5, 2), [0, 2, 4])
        self.assertEqual(select_render_offsets(5, 5, 2), [1, 3])
        self.assertEqual(select_render_offsets(10, 5, 2), [0, 2, 4])

    def test_expand_rendered_frames_holds_previous_sparse_render(self):
        first = np.full((2, 2, 3), 10, dtype=np.uint8)
        second = np.full((2, 2, 3), 20, dtype=np.uint8)
        held = np.full((2, 2, 3), 5, dtype=np.uint8)

        expanded, last = expand_rendered_frames(
            np.stack([first, second]),
            [1, 3],
            4,
            held,
        )

        np.testing.assert_array_equal(expanded[0], held)
        np.testing.assert_array_equal(expanded[1], first)
        np.testing.assert_array_equal(expanded[2], first)
        np.testing.assert_array_equal(expanded[3], second)
        np.testing.assert_array_equal(last, second)


if __name__ == "__main__":
    unittest.main()
