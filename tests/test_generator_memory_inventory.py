import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os

import torch

from inference_memory import suspended_generator_cuda_tensors, cuda_storage_inventory, MemoryRecorder


def nested_generator(tensor):
    text = "PRIVATE_SENTENCE_MUST_NOT_BE_RECORDED"
    private_map = {"PRIVATE_KEY_MUST_NOT_BE_RECORDED": tensor}
    alias = tensor.view(-1)
    def inner():
        kept = tensor
        yield kept
    stream = inner()
    yield next(stream)
    yield text, alias


class GeneratorMemoryInventoryTests(unittest.TestCase):
    def test_cpu_data_is_not_retained_and_generator_is_not_advanced(self):
        stream = nested_generator(torch.ones(8))
        next(stream)
        self.assertEqual(suspended_generator_cuda_tensors(torch, stream), {})
        self.assertEqual(next(stream)[0], "PRIVATE_SENTENCE_MUST_NOT_BE_RECORDED")
        stream.close()
        self.assertEqual(suspended_generator_cuda_tensors(torch, stream), {})

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_actual_suspended_cuda_storage_is_counted_once_without_text(self):
        tensor = torch.ones(1024**2, device="cuda")
        stream = nested_generator(tensor)
        next(stream)
        captured = suspended_generator_cuda_tensors(torch, stream)
        inventory = cuda_storage_inventory(torch, {}, captured)
        self.assertEqual(inventory["unique_cuda_storage_bytes"], 4 * 1024**2)
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root) as directory, \
                patch.dict(os.environ, {"OPENVA_MEMORY_PROFILE_DIR": directory}):
            assert Path(directory).resolve().is_relative_to(root.resolve())
            recorder = MemoryRecorder("test", torch, "cuda")
            recorder.record("suspended", {}, captured)
            serialized = recorder.path.read_text(encoding="utf-8")
            self.assertNotIn("PRIVATE_SENTENCE_MUST_NOT_BE_RECORDED", serialized)
            self.assertNotIn("PRIVATE_KEY_MUST_NOT_BE_RECORDED", serialized)
            self.assertEqual(json.loads(serialized)["unique_cuda_storage_bytes"], 4 * 1024**2)
        self.assertEqual(next(stream)[0], "PRIVATE_SENTENCE_MUST_NOT_BE_RECORDED")
        stream.close()


if __name__ == "__main__":
    unittest.main()
