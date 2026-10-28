"""Real production motion transformer/head; excludes Wav2Vec and checkpoints."""

import ast
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import Mlp


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model/motion_generation/motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder.py"


def load_classes():
    names = {"RoPEEncoding", "SelfAttention_Rope", "CrossAttention_Rope", "SelfAttention_Pos",
             "PositionalEncoding", "Audio2FaceGPTBlock", "DiffusionBlock", "DiffusionHead",
             "CUDAGraphGPTRunner"}
    nodes = [n for n in ast.parse(MODEL.read_text(encoding="utf-8")).body
             if isinstance(n, ast.ClassDef) and n.name in names]
    namespace = dict(torch=torch, nn=nn, F=F, math=math, Mlp=Mlp)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MODEL), "exec"), namespace)
    return namespace


def make_motion_core(hidden_size=768, layers=12, max_seq_len=1024):
    classes = load_classes()
    model = nn.Module()
    model.blocks = nn.ModuleList([
        classes["Audio2FaceGPTBlock"](hidden_size, 12, max_seq_len=max_seq_len)
        for _ in range(layers)])
    model.diffusion_head = classes["DiffusionHead"](
        face_dim=512, hidden_size=hidden_size, num_layers=6, num_heads=12, max_seq_len=max_seq_len)
    model.output_norm = nn.LayerNorm(hidden_size)
    model.output_proj = nn.Linear(hidden_size, 512)
    return model.eval(), classes
