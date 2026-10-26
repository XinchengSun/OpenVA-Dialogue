"""Exercise production raw-media packing without loading models or ffmpeg."""

import ast
import __future__
from pathlib import Path
import queue
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "server_mse.py"


def load_client(source=None):
    tree = ast.parse(source if source is not None else SOURCE.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == "MediaPipeClient")
    methods = {"_write_to_ffmpeg", "_float_to_s16le", "push_segment"}
    node.body = [n for n in node.body if getattr(n, "name", None) in methods]
    namespace = dict(np=np, queue=queue)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace["MediaPipeClient"]


Client = load_client()


def make_client(cls=Client, stride=2, record=True):
    client = cls()
    client.proc = SimpleNamespace(stdin=object(), poll=lambda: None)
    client.audio_w_fd = 1
    client.frame_stride = stride
    client.samples_per_out_frame = 640 * stride
    client.output_fps = 25 / stride
    client._generation_lock = threading.Lock()
    client._active_assistant_media_key = None
    client.accepted_generation = 0
    client.enqueued_frames = 0
    client.client_id = 0
    client.running = threading.Event()
    client.running.set()
    client.segment_q = queue.Queue()
    client.av_pair_q = queue.Queue()
    client.engine = SimpleNamespace(log=lambda message: None)
    client.units = []

    def enqueue(*args):
        if record:
            client.units.append(args)
        return True

    client._enqueue_av_unit = enqueue
    return client


class MediaCopyPerformanceTests(unittest.TestCase):
    def test_encoder_bytes_match_for_contiguous_strided_and_float_frames(self):
        base = np.random.default_rng(17).integers(0, 256, (4, 8, 8, 3), dtype=np.uint8)
        audio = np.linspace(-1.1, 1.1, 4 * 640, dtype=np.float32)
        for frames in (base, base[:, ::2, ::2], base.astype(np.float32) + 0.25):
            for stride in (1, 2):
                with self.subTest(dtype=frames.dtype, stride=stride, shape=frames.shape):
                    client = make_client(stride=stride)
                    client._write_to_ffmpeg(0, frames, audio)
                    expected = frames.astype(np.uint8)
                    self.assertEqual(len(client.units), 4 // stride)
                    for unit, index in zip(client.units, range(0, 4, stride)):
                        self.assertEqual(unit[1], expected[index].tobytes())
                        self.assertEqual(unit[2], client._float_to_s16le(
                            audio[index * 640:(index + stride) * 640]))

    def test_normal_input_reuses_segment_allocation(self):
        frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
        audio = np.zeros(4 * 640, dtype=np.float32)
        conversions = []
        original = np.ascontiguousarray

        def capture(value, *args, **kwargs):
            result = original(value, *args, **kwargs)
            conversions.append(result)
            return result

        with patch.object(np, "ascontiguousarray", side_effect=capture):
            make_client()._write_to_ffmpeg(0, frames, audio)
        self.assertIs(conversions[0], frames)

    def test_async_queue_still_owns_frames_and_audio(self):
        client = make_client()
        frames = np.full((4, 8, 8, 3), 123, dtype=np.uint8)
        audio = np.full(4 * 640, 0.5, dtype=np.float32)
        self.assertTrue(client.push_segment(frames, audio, generation=0))
        frames.fill(0)
        audio.fill(0)
        _, queued_frames, queued_audio, _ = client.segment_q.get_nowait()
        np.testing.assert_array_equal(queued_frames, 123)
        np.testing.assert_array_equal(queued_audio, 0.5)


if __name__ == "__main__":
    unittest.main()
