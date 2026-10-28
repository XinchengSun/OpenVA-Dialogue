"""Load real reference encoder classes without the demo/model download stack."""

import ast
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
HEAD = ROOT / "tools/visualization_0416/utils/model_0506/model/head_animation"


def _class(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def load_encoders():
    # Execute the actual utility files: all tensor arithmetic remains production
    # code. Extract only the encoder classes to avoid import-path modifications.
    util_path = HEAD / "LIA_3d/util.py"
    util = {}
    exec(compile(util_path.read_text(encoding="utf-8"), str(util_path), "exec"), util)
    face = _class(HEAD / "LIA_3d/face_encoder.py", "FaceEncoder", {
        "torch": torch, "nn": nn, "F": F,
        **{name: util[name] for name in ("SameBlock2d", "DownBlock2d", "ResBlock3d")},
    })
    modules_path = HEAD / "LIA/modules.py"
    modules = {}
    exec(compile(modules_path.read_text(encoding="utf-8"), str(modules_path), "exec"), modules)
    motion = _class(HEAD / "LIA_3d/motion_encoder.py", "MotionEncoder", {
        "torch": torch, "nn": nn, "F": F,
        **{name: modules[name] for name in ("ConvLayer", "ResBlock", "EqualConv2d")},
    })
    return {
        "face_encoder": face(image_size=512, image_channel=3, block_expansion=64,
                             num_down_blocks=3, max_features=512, reshape_channel=32,
                             reshape_depth=16, num_resblocks=6).eval(),
        "motion_encoder": motion(latent_dim=512, size=512).eval(),
    }


def tensor_inventory(module):
    parameters = tuple(module.parameters())
    buffers = tuple(module.buffers())
    return {
        "parameters": sum(t.numel() for t in parameters),
        "parameter_bytes": sum(t.numel() * t.element_size() for t in parameters),
        "buffer_bytes": sum(t.numel() * t.element_size() for t in buffers),
        "total_tensor_bytes": sum(t.numel() * t.element_size() for t in parameters + buffers),
    }


def load_offload():
    path = ROOT / "dual_gpu_mic_realtime_ui_v8_playbuffer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "offload_reference_encoders")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]
