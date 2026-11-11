import asyncio
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn.utils import weight_norm

from inference_memory import enable_prompt_encoder_offload, cache_decoder_weight_norm


def make_vae(device="cpu"):
    def network():
        return nn.Sequential(weight_norm(nn.Conv1d(3, 4, 3, padding=1)), nn.Tanh(),
                             weight_norm(nn.Conv1d(4, 3, 1)))
    return SimpleNamespace(encoder=network().eval().to(device), decoder=network().eval().to(device))


class TTSMemoryTests(unittest.TestCase):
    def test_cpu_and_absent_vae_are_left_unchanged(self):
        for model in (SimpleNamespace(), SimpleNamespace(audio_vae=make_vae())):
            self.assertFalse(enable_prompt_encoder_offload(torch, model)["enabled"])
            self.assertFalse(cache_decoder_weight_norm(torch, model)["enabled"])

    def test_shared_encoder_decoder_state_is_rejected(self):
        vae = make_vae()
        vae.decoder[0].weight_v = vae.encoder[0].weight_v
        result = enable_prompt_encoder_offload(torch, SimpleNamespace(audio_vae=vae))
        self.assertEqual(result["reason"], "shared_encoder_decoder_state")
        self.assertFalse(cache_decoder_weight_norm(torch, SimpleNamespace(audio_vae=vae))["enabled"])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.no_grad()
    def test_prompt_encoder_offload_moves_plain_weights_and_reloads_exactly(self):
        vae = make_vae("cuda")
        value = torch.randn(2, 3, 32, device="cuda")
        expected = vae.encoder(value).cpu()
        model = SimpleNamespace(audio_vae=vae)
        result = enable_prompt_encoder_offload(torch, model)
        self.assertTrue(result["enabled"])
        self.assertGreater(result["allocated_freed_bytes"], 0)
        for _ in range(2):
            torch.testing.assert_close(vae.encoder(value).cpu(), expected, rtol=0, atol=0)
            self.assertTrue(all(p.device.type == "cpu" for p in vae.encoder.parameters()))
            self.assertEqual(vae.encoder[0].weight.device.type, "cpu")
        self.assertTrue(enable_prompt_encoder_offload(torch, model)["already_enabled"])
        self.assertTrue(all(p.is_cuda for p in vae.decoder.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.no_grad()
    def test_decoder_exact_reuse_and_checkpoint_reload(self):
        vae = make_vae("cuda")
        model = SimpleNamespace(audio_vae=vae)
        value = torch.randn(2, 3, 32, device="cuda")
        expected = vae.decoder(value)
        state = {k: v.cpu().clone() for k, v in vae.decoder.state_dict().items()}
        self.assertTrue(cache_decoder_weight_norm(torch, model)["enabled"])
        torch.testing.assert_close(vae.decoder(value), expected, rtol=0, atol=0)
        weight = vae.decoder[0].weight
        torch.testing.assert_close(vae.decoder(value), expected, rtol=0, atol=0)
        self.assertIs(weight, vae.decoder[0].weight)
        self.assertEqual(vae.decoder[0].weight_v.device.type, "cpu")
        self.assertEqual(tuple(vae.decoder.state_dict()), tuple(state))
        for key, tensor in vae.decoder.state_dict().items():
            torch.testing.assert_close(tensor.cpu(), state[key], rtol=0, atol=0)
        state["0.weight_v"].add_(0.125)
        reference = make_vae("cuda").decoder
        reference.load_state_dict(state)
        vae.decoder.load_state_dict(state)
        torch.testing.assert_close(vae.decoder(value), reference(value), rtol=0, atol=0)
        self.assertIsNot(weight, vae.decoder[0].weight)
        self.assertEqual(cache_decoder_weight_norm(torch, model)["reason"], "already_cached")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.no_grad()
    def test_decoder_dtype_and_device_roundtrip_invalidate_cache(self):
        vae = make_vae("cuda")
        cache_decoder_weight_norm(torch, SimpleNamespace(audio_vae=vae))
        value = torch.randn(2, 3, 32, device="cuda")
        vae.decoder(value)
        old_weight = vae.decoder[0].weight
        vae.decoder.half().float()
        state = vae.decoder.state_dict()
        reference = make_vae("cuda").decoder
        reference.load_state_dict(state)
        expected = reference(value)
        torch.testing.assert_close(vae.decoder(value), expected, rtol=0, atol=0)
        self.assertIsNot(vae.decoder[0].weight, old_weight)
        vae.decoder.cpu()
        cpu_reference = make_vae().decoder
        cpu_reference.load_state_dict(vae.decoder.state_dict())
        torch.testing.assert_close(vae.decoder(value.cpu()), cpu_reference(value.cpu()), rtol=0, atol=0)
        vae.decoder.cuda()
        torch.testing.assert_close(vae.decoder(value), expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_decoder_training_and_grad_eval_use_original_weight_norm(self):
        for training in (False, True):
            vae = make_vae("cuda")
            reference = make_vae("cuda").decoder
            reference.load_state_dict(vae.decoder.state_dict())
            cache_decoder_weight_norm(torch, SimpleNamespace(audio_vae=vae))
            vae.decoder.train(training)
            reference.train(training)
            value = torch.randn(2, 3, 32, device="cuda", requires_grad=True)
            other = value.detach().clone().requires_grad_()
            actual, expected = vae.decoder(value), reference(other)
            actual.square().mean().backward()
            expected.square().mean().backward()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(value.grad, other.grad, rtol=0, atol=0)
            for a, b in zip(vae.decoder.parameters(), reference.parameters()):
                self.assertTrue(a.is_cuda)
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_inference_parameters_without_version_counters_are_not_cached(self):
        with torch.inference_mode():
            vae = make_vae("cuda")
        self.assertFalse(cache_decoder_weight_norm(torch, SimpleNamespace(audio_vae=vae))["enabled"])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_backend_applies_memory_policy_after_prompt_build(self):
        from tests.test_official_voxcpm2_backend import _FakeTTSModel, _FakeModel, _Factory
        from voice_service.official_voxcpm2_backend import OfficialPromptCacheBackend

        class TTS(_FakeTTSModel):
            def __init__(self):
                super().__init__()
                self.audio_vae = make_vae("cuda")

            def build_prompt_cache(self, **kwargs):
                self.assert_cuda = all(p.is_cuda for p in self.audio_vae.encoder.parameters())
                with torch.no_grad():
                    self.audio_vae.encoder(torch.zeros(1, 3, 8, device="cuda"))
                return super().build_prompt_cache(**kwargs)

        async def check():
            tts = TTS()
            with tempfile.TemporaryDirectory() as directory:
                wav = Path(directory) / "reference.wav"
                wav.touch()
                backend = OfficialPromptCacheBackend(model_path="test", reference_wav=str(wav),
                                                     model_factory=_Factory(_FakeModel(tts)))
                await backend.start()
                self.assertTrue(tts.assert_cuda)
                self.assertTrue(all(p.device.type == "cpu" for p in tts.audio_vae.encoder.parameters()))
                self.assertEqual(tts.audio_vae.decoder[0].weight_v.device.type, "cpu")
                chunks = [chunk async for chunk in backend.generate_pcm16("hello", "one")]
                self.assertTrue(chunks)
                await backend.stop()
                self.assertIsNone(backend._memory_recorder)
        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
