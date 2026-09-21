"""Check the real loader's EMA folding without importing the Gradio/vis stack."""
import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch_ema import ExponentialMovingAverage


class InferenceEMATests(unittest.TestCase):
    def load(self, *, fold, with_ema=True):
        source = Path(__file__).resolve().parents[1] / 'app.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'load_dystream_model')
        model = torch.nn.Linear(3, 2)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(1.25)
        ema = ExponentialMovingAverage(model.parameters(), decay=0.99)
        for parameter in ema.shadow_params:
            parameter.fill_(2.5)
        checkpoint = {'state_dict': model.state_dict()}
        if with_ema:
            checkpoint['ema_state'] = ema.state_dict()
        namespace = dict(
            os=os, torch=torch, PROJECT_ROOT=str(source.parent), DEVICE='cpu',
            _dystream_model=None, _dystream_ema=None,
            Config=lambda *_: SimpleNamespace(
                model=SimpleNamespace(module_name='mock', class_name='mock', ema_decay=0.99),
                resume_ckpt='test.ckpt', noise_scheduler_kwargs={}),
            instantiate_motion_gen=lambda **_: torch.nn.Linear(3, 2),
            ExponentialMovingAverage=ExponentialMovingAverage,
            OmegaConf=SimpleNamespace(to_container=lambda value, **_: value),
            FlowMatchEulerDiscreteScheduler=lambda **_: object(),
        )
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        with patch.dict(os.environ, {'DYSTREAM_FOLD_EMA': str(int(fold))}), \
                patch('os.path.exists', return_value=True), \
                patch('torch.load', return_value=checkpoint):
            namespace['load_dystream_model']()
        return namespace

    def test_folded_weights_match_legacy_ema_exactly(self):
        baseline, folded = self.load(fold=False), self.load(fold=True)
        self.assertIsNone(folded['_dystream_ema'])
        with baseline['_dystream_ema'].average_parameters(baseline['_dystream_model'].parameters()):
            for expected, actual in zip(baseline['_dystream_model'].parameters(),
                                        folded['_dystream_model'].parameters()):
                self.assertTrue(torch.equal(expected, actual))

    def test_checkpoint_without_ema_retains_model_weights(self):
        folded = self.load(fold=True, with_ema=False)
        for parameter in folded['_dystream_model'].parameters():
            self.assertTrue(torch.equal(parameter, torch.full_like(parameter, 1.25)))

    def test_repeated_loading_keeps_folded_model(self):
        folded = self.load(fold=True)
        model = folded['_dystream_model']
        folded['load_dystream_model']()
        self.assertIs(folded['_dystream_model'], model)
        self.assertIsNone(folded['_dystream_ema'])


if __name__ == '__main__':
    unittest.main()
