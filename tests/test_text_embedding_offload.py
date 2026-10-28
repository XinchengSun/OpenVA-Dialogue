import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from tests.bench_text_embedding_memory import bits_equal
from voice_service.text_embedding_offload import offload_text_embedding


def holder(device="cpu", **kwargs):
    model = nn.Module()
    model.base_lm = nn.Module()
    model.base_lm.embed_tokens = nn.Embedding(32, 16, **kwargs).eval().to(device)
    return model.eval()


class TextEmbeddingOffloadTests(unittest.TestCase):
    def test_cpu_training_and_mutable_norm_are_skipped(self):
        for model in (holder(), holder().train(), holder(max_norm=1)):
            self.assertFalse(offload_text_embedding(model)["enabled"])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.no_grad()
    def test_lookup_state_reload_mutation_and_dtype_device_transitions(self):
        model = holder("cuda")
        embedding = model.base_lm.embed_tokens
        expected = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        ids = torch.tensor([[0, 1, 5, 31, 5]], device="cuda")
        baseline = embedding(ids)
        self.assertTrue(offload_text_embedding(model)["enabled"])
        bits_equal(baseline, embedding(ids))
        self.assertEqual(embedding.weight.device.type, "cpu")
        self.assertEqual(tuple(model.state_dict()), tuple(expected))
        self.assertTrue(all(torch.equal(v.cpu(), expected[k]) for k, v in model.state_dict().items()))
        self.assertEqual(offload_text_embedding(model)["reason"], "already_offloaded")
        model.half().float()
        self.assertTrue(embedding._openva_device_marker.is_cuda)
        reference = holder("cuda")
        reference.load_state_dict(model.state_dict())
        bits_equal(embedding(ids), reference.base_lm.embed_tokens(ids))
        embedding.weight.add_(0.125)
        reference.load_state_dict(model.state_dict())
        bits_equal(embedding(ids), reference.base_lm.embed_tokens(ids))
        model.load_state_dict(expected)
        bits_equal(embedding(ids), baseline)
        model.cpu()
        self.assertEqual(embedding(ids.cpu()).device.type, "cpu")
        bits_equal(embedding(ids.cpu()), baseline.cpu())
        model.cuda()
        bits_equal(embedding(ids), baseline)
        self.assertEqual(embedding.weight.device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_grad_enabled_and_training_preserve_original_gradients(self):
        for training in (False, True):
            model, reference = holder("cuda"), holder("cuda")
            reference.load_state_dict(model.state_dict())
            offload_text_embedding(model)
            model.train(training)
            reference.train(training)
            ids = torch.tensor([1, 3, 6], device="cuda")
            a, b = model.base_lm.embed_tokens(ids), reference.base_lm.embed_tokens(ids)
            a.square().sum().backward()
            b.square().sum().backward()
            bits_equal(a.detach(), b.detach())
            bits_equal(model.base_lm.embed_tokens.weight.grad, reference.base_lm.embed_tokens.weight.grad)
            self.assertTrue(model.base_lm.embed_tokens.weight.is_cuda)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_tied_weights_are_rejected_before_mutation(self):
        model = holder("cuda")
        model.tied = nn.Linear(16, 32, bias=False).cuda()
        model.tied.weight = model.base_lm.embed_tokens.weight
        self.assertEqual(offload_text_embedding(model)["reason"], "shared_embedding_storage")
        self.assertTrue(model.base_lm.embed_tokens.weight.is_cuda)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.no_grad()
    def test_cpu_indices_keep_original_success_or_error_behavior(self):
        model, reference = holder("cuda"), holder("cuda")
        reference.load_state_dict(model.state_dict())
        offload_text_embedding(model)
        ids = torch.tensor([1, 2])
        try:
            expected = reference.base_lm.embed_tokens(ids)
        except RuntimeError:
            with self.assertRaises(RuntimeError):
                model.base_lm.embed_tokens(ids)
        else:
            actual = model.base_lm.embed_tokens(ids)
            self.assertEqual(actual.device, expected.device)
            bits_equal(actual, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_backend_opt_in_after_prompt_cache_and_during_generation(self):
        from tests.test_official_voxcpm2_backend import _FakeTTSModel, _FakeModel, _Factory
        from voice_service.official_voxcpm2_backend import OfficialPromptCacheBackend

        class TTS(_FakeTTSModel, nn.Module):
            def __init__(self):
                nn.Module.__init__(self)
                _FakeTTSModel.__init__(self)
                self.base_lm = holder("cuda").base_lm
                self.eval()

            def build_prompt_cache(self, **kwargs):
                assert self.base_lm.embed_tokens.weight.is_cuda
                return super().build_prompt_cache(**kwargs)

            def generate_with_prompt_cache_streaming(self, **kwargs):
                with torch.inference_mode():
                    prefill = torch.ones(1024, device="cuda")
                    output = self.base_lm.embed_tokens(torch.tensor([1, 3], device="cuda"))
                    assert output.is_cuda and self.base_lm.embed_tokens.weight.device.type == "cpu"
                    yield from super().generate_with_prompt_cache_streaming(**kwargs)

        async def check():
            root = Path(__file__).resolve().parents[1]
            with tempfile.TemporaryDirectory(dir=root) as directory:
                assert Path(directory).resolve().is_relative_to(root.resolve())
                wav = Path(directory) / "reference.wav"
                wav.touch()
                tts = TTS()
                ledger = Path(directory) / "ledgers"
                with patch.dict(os.environ, {"VOXCPM2_CPU_TEXT_EMBEDDING": "1",
                                             "OPENVA_MEMORY_PROFILE_DIR": str(ledger)}):
                    backend = OfficialPromptCacheBackend(model_path="test", reference_wav=str(wav),
                                                         model_factory=_Factory(_FakeModel(tts)))
                    await backend.start()
                    self.assertEqual(tts.base_lm.embed_tokens.weight.device.type, "cpu")
                    chunks = [x async for x in backend.generate_pcm16("hello", "case")]
                    self.assertTrue(chunks)
                    await backend.stop()
                records = [json.loads(line) for path in ledger.glob("*.jsonl")
                           for line in path.read_text(encoding="utf-8").splitlines()]
                streaming = next(record for record in records if record["stage"] == "stream_chunks_1")
                self.assertTrue(any("prefill" in key and size == 4096
                                    for key, size in streaming["storage_bytes"].items()))
        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
