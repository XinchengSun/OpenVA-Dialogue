import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn.utils import spectral_norm


ROOT = Path(__file__).resolve().parents[1]


class RendererMemoryIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_loader_flags_apply_only_after_loading_original_checkpoint(self):
        function = next(node for node in ast.parse((ROOT / "app.py").read_text(encoding="utf-8")).body
                        if isinstance(node, ast.FunctionDef) and node.name == "load_visualization_model")
        for enabled in (False, True, None):
            head = nn.Module()
            head.face_generator = spectral_norm(nn.Conv2d(3, 3, 1))
            for name in ("flow_estimator", "face_encoder", "motion_encoder"):
                setattr(head, name, nn.Identity())
            state = {k: value.clone() for k, value in head.state_dict().items()}
            loader = SimpleNamespace(exec_module=lambda module: setattr(module, "instantiate", lambda *a, **kw: lambda **kwargs: head))
            transforms = SimpleNamespace(Compose=lambda value: value, Resize=lambda *a: None,
                                         ToTensor=lambda: None, Normalize=lambda *a: None)
            namespace = dict(_vis_ctx=None, VIS_DIR=str(ROOT), VIS_MODEL_DIR=str(ROOT), os=os, torch=torch,
                             _ilu=SimpleNamespace(spec_from_file_location=lambda *a: SimpleNamespace(loader=loader),
                                                  module_from_spec=lambda spec: SimpleNamespace()),
                             OmegaConf=SimpleNamespace(load=lambda *a: SimpleNamespace(resume_ckpt="test", model={})),
                             DEVICE="cuda", T=transforms)
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / "app.py"), "exec"), namespace)
            settings = dict(os.environ)
            for key in ("DYSTREAM_CACHE_SPECTRAL_NORM", "DYSTREAM_RELEASE_RENDER_INTERMEDIATES"):
                if enabled is None:
                    settings.pop(key, None)
                else:
                    settings[key] = str(int(enabled))
            expected_enabled = enabled is not False
            with patch.dict(os.environ, settings, clear=True), \
                    patch.object(torch, "load", return_value={"state_dict": state}):
                namespace["load_visualization_model"]()
            renderer = namespace["_vis_ctx"]["face_generator"]
            self.assertIs(renderer, head.face_generator)
            self.assertEqual(renderer._release_render_intermediates, expected_enabled)
            self.assertEqual(renderer.weight_orig.device.type, "cpu" if expected_enabled else "cuda")
            for key, value in head.state_dict().items():
                torch.testing.assert_close(value.cpu(), state[key], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.inference_mode()
    def test_full_renderer_lifetimes_with_and_without_occlusion(self):
        from tests.bench_renderer_memory import make_renderer
        with torch.inference_mode(False):
            torch.manual_seed(20261028)
            model = make_renderer().cuda()
        latent = torch.randn(1, 512, device="cuda") * 0.1
        feature = torch.randn(1, 32, 16, 64, 64, device="cuda") * 0.1
        for occlusion in (True, False):
            model.flag_estimate_occlusion_map = occlusion
            model._release_render_intermediates = False
            expected = model(latent, feature).clone()
            model._release_render_intermediates = True
            actual = model(latent, feature)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))


if __name__ == "__main__":
    unittest.main()
