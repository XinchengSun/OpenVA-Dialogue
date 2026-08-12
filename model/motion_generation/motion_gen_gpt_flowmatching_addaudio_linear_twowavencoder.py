import os
from pathlib import Path
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import PretrainedConfig
from timm.models.vision_transformer import Mlp
from transformers import PreTrainedModel, Wav2Vec2Processor, Wav2Vec2Model, PretrainedConfig
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Attention
from typing import Optional, Tuple, Union
from .motion_gen_utils_dev import WanTimeEmbedding
import time


WAV2VEC_DIR = os.getenv(
    "DYSTREAM_WAV2VEC_DIR",
    str(Path(__file__).resolve().parents[2] / "tools" / "hf_models" / "wav2vec2-base-960h"),
)



# ================= CUDA Graph minimal runners =================

class CUDAGraphGPTRunner:
    """
    CUDA Graph runner for the existing GPT/AR blocks.
    It does not add parameters and does not change checkpoint structure.
    """
    def __init__(self, blocks, output_norm, output_proj, hidden_size=768, batch_size=5, device="cuda"):
        self.blocks = blocks
        self.output_norm = output_norm
        self.output_proj = output_proj
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = device
        self.graphs = {}
        self.static_buffers = {}

    def warmup_and_capture_for_length(self, seq_len):
        if seq_len in self.graphs:
            return

        static_x = torch.empty(self.batch_size, seq_len, self.hidden_size, device=self.device)
        static_audio = torch.empty(self.batch_size, seq_len, self.hidden_size, device=self.device)
        static_anchor = torch.empty(self.batch_size, 1, self.hidden_size, device=self.device)
        static_causal_mask = torch.zeros(seq_len, seq_len, device=self.device)
        static_cross_mask = torch.zeros(seq_len, seq_len, device=self.device)

        static_x.normal_()
        static_audio.normal_()
        static_anchor.normal_()

        for _ in range(3):
            x = static_x
            for block in self.blocks:
                x = block(x, static_audio, static_anchor, static_causal_mask, static_cross_mask)
            x_t = self.output_norm(x[:, -1:])
            _ = self.output_proj(x_t)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            x = static_x
            for block in self.blocks:
                x = block(x, static_audio, static_anchor, static_causal_mask, static_cross_mask)
            x_t = self.output_norm(x[:, -1:])
            static_output = self.output_proj(x_t)

        self.graphs[seq_len] = graph
        self.static_buffers[seq_len] = {
            "x": static_x,
            "audio": static_audio,
            "anchor": static_anchor,
            "causal_mask": static_causal_mask,
            "cross_mask": static_cross_mask,
            "output": static_output,
        }
        print(f"[CUDA Graph] GPT graph captured for seq_len={seq_len}")

    def forward(self, x, audio_hidden, anchor_hidden, causal_mask, cross_causal_mask):
        seq_len = x.shape[1]

        if seq_len not in self.graphs:
            self.warmup_and_capture_for_length(seq_len)

        buffers = self.static_buffers[seq_len]
        buffers["x"].copy_(x)
        buffers["audio"].copy_(audio_hidden)
        buffers["anchor"].copy_(anchor_hidden)
        buffers["causal_mask"].copy_(causal_mask)
        buffers["cross_mask"].copy_(cross_causal_mask)

        self.graphs[seq_len].replay()
        return buffers["output"].clone()


class CUDAGraphDiffusionHeadSingleStep:
    """
    CUDA Graph runner for one DiffusionHead forward step.
    It only captures diffusion_head(noisy, gpt, temb).
    Scheduler step is intentionally kept outside the graph.
    """
    def __init__(self, diffusion_head, batch_size=5, face_dim=512, hidden_size=768, device="cuda"):
        self.diffusion_head = diffusion_head
        self.batch_size = batch_size
        self.face_dim = face_dim
        self.hidden_size = hidden_size
        self.device = device

        self.graph = None
        self.static_noisy = None
        self.static_gpt = None
        self.static_temb = None
        self.static_output = None

    def warmup_and_capture(self):
        self.static_noisy = torch.empty(self.batch_size, 1, self.face_dim, device=self.device)
        self.static_gpt = torch.empty(self.batch_size, 1, self.face_dim, device=self.device)
        self.static_temb = torch.empty(1, 1, self.hidden_size, device=self.device)

        self.static_noisy.normal_()
        self.static_gpt.normal_()
        self.static_temb.normal_()

        for _ in range(3):
            _ = self.diffusion_head(self.static_noisy, self.static_gpt, temb=self.static_temb)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.diffusion_head(
                self.static_noisy,
                self.static_gpt,
                temb=self.static_temb,
            )

        print("[CUDA Graph] DiffusionHead single-step graph captured")

    def forward(self, noisy_input, gpt_output, temb):
        self.static_noisy.copy_(noisy_input)
        self.static_gpt.copy_(gpt_output)
        self.static_temb.copy_(temb)

        self.graph.replay()
        return self.static_output.clone()

# ================= End CUDA Graph minimal runners =================


class RoPEEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=dropout)
        theta = 10000 ** (-2 * torch.arange(0, d_model // 2, dtype=torch.float) / d_model)
        positions = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        angles = positions * theta
        self.register_buffer('cos_angles', angles.cos())
        self.register_buffer('sin_angles', angles.sin())

    def forward(self, x):
        seq_len = x.size(1)
        cos_angles = self.cos_angles[:seq_len].to(x.device)
        sin_angles = self.sin_angles[:seq_len].to(x.device)
        x_even = x[:, :, 0::2]
        x_odd = x[:, :, 1::2]
        x_rot = x.clone()
        x_rot[:, :, 0::2] = x_even * cos_angles - x_odd * sin_angles
        x_rot[:, :, 1::2] = x_even * sin_angles + x_odd * cos_angles
        return self.dropout(x_rot)


class SelfAttention_Rope(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, max_seq_len=128):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)
        self.rope = RoPEEncoding(d_model, dropout=dropout, max_seq_len=max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ):
        x = self.self_attn(
            self.rope(x),
            self.rope(x),
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = self.dropout(x)
        return x


class CrossAttention_Rope(nn.Module):
    def __init__(self, d_model: int, d_cond: int, num_heads: int, dropout: float = 0.1, max_seq_len=128):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=d_cond,
            vdim=d_cond,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.rope = RoPEEncoding(d_model, dropout=dropout, max_seq_len=max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ):
        x = self.cross_attn(
            self.rope(x),
            self.rope(cond),
            cond,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = self.dropout(x)
        return x


class SelfAttention_Pos(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, max_seq_len=128):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)
        self.pe = PositionalEncoding(
            d_model, dropout=dropout, max_seq_len=max_seq_len
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ):
        x = self.self_attn(
            self.pe(x),
            self.pe(x),
            self.pe(x),
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = self.dropout(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_seq_len=128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class AutoModelConfig(PretrainedConfig):
    def __init__(self, config_obj=None, **kwargs):
        if config_obj is not None:
            cfg_dict = OmegaConf.to_container(config_obj, resolve=True)
            kwargs.update(cfg_dict)
            self.model_type = kwargs.pop("model_type", "my_model")
        super().__init__(**kwargs)


class Audio2FaceGPTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1, max_seq_len=128):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.norm4 = nn.LayerNorm(hidden_size)
        self.norm_anchor = nn.LayerNorm(hidden_size)
        self.self_attn_rope = SelfAttention_Rope(hidden_size, num_heads, dropout, max_seq_len=max_seq_len)
        self.cross_attn_rope = CrossAttention_Rope(hidden_size, hidden_size, num_heads, dropout, max_seq_len=max_seq_len)
        self.self_attn_pos = SelfAttention_Pos(hidden_size, num_heads, dropout, max_seq_len=max_seq_len)
        self.cross_attn_anchor = CrossAttention_Rope(hidden_size, hidden_size, num_heads, dropout, max_seq_len=max_seq_len)
        self.cross_linear_audio = nn.Linear(hidden_size, hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)

    def forward(self, x, audio_features, anchor_hidden, causal_mask=None, cross_causal_mask=None):
        residual = x
        x = self.norm1(x)
        x = self.self_attn_rope(x, attn_mask=causal_mask)
        x = residual + x

        residual = x
        x = residual + self.cross_linear_audio(audio_features)

        residual = x
        x = self.norm_anchor(x)
        x = self.cross_attn_anchor(x, anchor_hidden, attn_mask=None)
        x = residual + x

        residual = x
        x = self.norm3(x)
        x = self.self_attn_pos(x, attn_mask=causal_mask)
        x = residual + x

        residual = x
        x = self.norm4(x)
        x = self.mlp(x)
        x = residual + x

        return x


def make_attention_causal(attn: Wav2Vec2Attention):
    q_proj, k_proj, v_proj, out_proj = attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj
    n_head, head_dim, p = attn.num_heads, attn.head_dim, attn.dropout

    def f(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        **_,
    ):
        B, T, _ = x.shape
        q = q_proj(x).view(B, T, n_head, head_dim).transpose(1, 2)
        k = k_proj(x).view(B, T, n_head, head_dim).transpose(1, 2)
        v = v_proj(x).view(B, T, n_head, head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=p if self.training else 0.0,
            is_causal=False,
        )
        y = out_proj(y.transpose(1, 2).reshape(B, T, n_head * head_dim))
        return (y, None, None) if output_attentions else (y, None, None)

    attn.forward = f.__get__(attn, attn.__class__)


class WrapedWav2Vec(nn.Module):
    def __init__(self, layers: int = 1):
        super().__init__()
        base = Wav2Vec2Model.from_pretrained(WAV2VEC_DIR, local_files_only=True)
        self.feature_extractor = base.feature_extractor
        self.feature_projection = base.feature_projection
        self.encoder = base.encoder
        self.encoder.layers = self.encoder.layers[:layers]
        for l in self.encoder.layers:
            make_attention_causal(l.attention)

    def forward(
        self,
        x: torch.Tensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        **_,
    ):
        low = self.feature_extractor(x).transpose(1, 2)
        h, _ = self.feature_projection(low.detach())
        enc = self.encoder(
            h,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return {"low_level": low, "high_level": enc[0]}


class DiffusionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1, max_seq_len=128):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp1 = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)
        self.mlp2 = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)
        self.adaLN_modulation1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True),
        )
        self.adaLN_modulation2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True),
        )

    def forward(self, hidden_states, gpt_hidden, temb=None):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation1(temb).chunk(3, dim=-1)
        shift_mlp2, scale_mlp2, gate_mlp2 = self.adaLN_modulation2(gpt_hidden).chunk(3, dim=-1)

        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_mlp) + shift_mlp).type_as(hidden_states)
        ff_output = self.mlp1(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * gate_mlp).type_as(hidden_states)

        norm_hidden_states = (self.norm2(hidden_states.float()) * (1 + scale_mlp2) + shift_mlp2).type_as(hidden_states)
        ff_output = self.mlp2(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * gate_mlp2).type_as(hidden_states)

        return hidden_states


class DiffusionHead(nn.Module):
    def __init__(
        self,
        face_dim=512,
        hidden_size=768,
        num_layers=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        max_seq_len=128,
    ):
        super().__init__()
        self.face_dim = face_dim
        self.hidden_size = hidden_size
        self.noisy_proj = nn.Linear(face_dim, hidden_size)
        self.gpt_proj = nn.Linear(face_dim, hidden_size)
        self.past_proj = nn.Linear(face_dim, hidden_size)
        self.anchor_proj = nn.Linear(face_dim, hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiffusionBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, face_dim)

    def forward(self, noisy_face_latent, gpt_output, temb=None):
        bs = noisy_face_latent.shape[0]
        device = noisy_face_latent.device
        noisy_hidden = self.noisy_proj(noisy_face_latent)
        gpt_hidden = self.gpt_proj(gpt_output)
        x = noisy_hidden
        for block in self.blocks:
            x = block(x, gpt_hidden, temb)
        x = self.output_norm(x)
        denoised = self.output_proj(x)
        return denoised


class Audio2FaceGPT(nn.Module):
    def __init__(
        self,
        cfg=None,
        audio_dim=768,
        face_dim=512,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        max_seq_len=1024,
    ):
        super().__init__()
        self.cfg = cfg
        self.audio_encoder_face = WrapedWav2Vec(layers=self.cfg.wav2vec_layer)
        self.audio_encoder_face_other = WrapedWav2Vec(layers=self.cfg.wav2vec_layer)
        self.audio_processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_DIR, local_files_only=True)
        self.audio_dim = audio_dim
        self.face_dim = face_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.audio_proj = nn.Linear(audio_dim, hidden_size)
        self.audio_other_proj = nn.Linear(audio_dim, hidden_size)
        self.audio_audioother_fusion = nn.Linear(2 * hidden_size, hidden_size)
        self.face_embed = nn.Linear(face_dim, hidden_size)
        self.anchor_embed = nn.Linear(face_dim, hidden_size)
        self.time_embed = WanTimeEmbedding(
            dim=hidden_size,
            time_freq_dim=hidden_size,
        )
        self.blocks = nn.ModuleList(
            [
                Audio2FaceGPTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, face_dim)
        self.inpainting_length = cfg.cbh_window_length - 2
        self.diffusion_head = DiffusionHead(
            face_dim=face_dim,
            hidden_size=hidden_size,
            num_layers=6,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.cfg_all = cfg.cfg_all
        self.drop_gpt = cfg.drop_gpt
        self.cfg_audio = cfg.cfg_audio
        self.drop_audio = cfg.drop_audio
        self.cfg_audio_other = cfg.cfg_audio_other
        self.drop_audio_other = cfg.drop_audio_other
        self.cfg_anchor = cfg.cfg_anchor
        self.drop_anchor = cfg.drop_anchor
        self.cfg_audio_anchor = cfg.cfg_audio_anchor

    def generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def generate_cross_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def get_audio2face_fea(self, audio, prev_audio, n):
        if prev_audio is not None:
            audio = torch.cat([prev_audio, audio], dim=1)
        audio_list = [i.cpu().numpy() for i in audio]
        inputs = self.audio_processor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True).to(audio.device)
        audio2face_fea = self.audio_encoder_face(inputs.input_values)["high_level"]
        audio2face_fea = F.interpolate(
            audio2face_fea.transpose(1, 2), scale_factor=(self.cfg.pose_fps / 50), mode="linear", align_corners=True
        ).transpose(1, 2)
        if prev_audio is not None:
            audio2face_fea = audio2face_fea[:, -n:]
        else:
            if audio2face_fea.shape[1] > n:
                audio2face_fea = audio2face_fea[:, :n]
            if audio2face_fea.shape[1] < n:
                audio2face_fea = torch.cat(
                    [audio2face_fea, audio2face_fea[:, -(n - audio2face_fea.shape[1]) :]],
                    dim=1,
                )
        return audio2face_fea

    def get_audio2face_fea_other(self, audio, prev_audio, n):
        if prev_audio is not None:
            audio = torch.cat([prev_audio, audio], dim=1)
        audio_list = [i.cpu().numpy() for i in audio]
        inputs = self.audio_processor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True).to(audio.device)
        audio2face_fea = self.audio_encoder_face_other(inputs.input_values)["high_level"]
        audio2face_fea = F.interpolate(
            audio2face_fea.transpose(1, 2), scale_factor=(self.cfg.pose_fps / 50), mode="linear", align_corners=True
        ).transpose(1, 2)
        if prev_audio is not None:
            audio2face_fea = audio2face_fea[:, -n:]
        else:
            if audio2face_fea.shape[1] > n:
                audio2face_fea = audio2face_fea[:, :n]
            if audio2face_fea.shape[1] < n:
                audio2face_fea = torch.cat(
                    [audio2face_fea, audio2face_fea[:, -(n - audio2face_fea.shape[1]) :]],
                    dim=1,
                )
        return audio2face_fea

    def forward(self, face_latent_gt, noise_face_latent, time_step, audio, audio_other, prev_audio, prev_audio_other, anchor_latent):
        bs, n, _ = face_latent_gt.shape
        audio2face_fea = self.get_audio2face_fea(audio, prev_audio, n)
        audio2face_fea_other = self.get_audio2face_fea_other(audio_other, prev_audio_other, n)
        device = audio2face_fea.device
        bs, seq_len, _ = audio2face_fea.shape
        audio_hidden = self.audio_proj(audio2face_fea)
        audio_hidden = audio_hidden[:, 1:]
        drop_audio_mask = torch.rand(bs, seq_len - 1, 1, device=face_latent_gt.device) < self.drop_audio
        drop_audio_mask = drop_audio_mask.float()
        audio_hidden = audio_hidden * (1 - drop_audio_mask)
        audio_other_hidden = self.audio_other_proj(audio2face_fea_other)
        audio_other_hidden = audio_other_hidden[:, 1:]
        drop_audio_other_mask = torch.rand(bs, seq_len - 1, 1, device=face_latent_gt.device) < self.drop_audio_other
        drop_audio_other_mask = drop_audio_other_mask.float()
        audio_other_hidden = audio_other_hidden * (1 - drop_audio_other_mask)
        audio_hidden = self.audio_audioother_fusion(torch.cat([audio_hidden, audio_other_hidden], dim=-1))
        anchor_hidden = self.anchor_embed(anchor_latent)
        causal_mask = self.generate_causal_mask(seq_len - 1, device)
        cross_causal_mask = self.generate_cross_causal_mask(seq_len - 1, device)
        face_hidden = self.face_embed(face_latent_gt[:, :-1])
        drop_anchor_mask = torch.rand(bs, 1, 1, device=face_hidden.device) < self.drop_anchor
        drop_anchor_mask = drop_anchor_mask.float()
        x = face_hidden
        for block in self.blocks:
            x = block(
                x,
                audio_hidden,
                anchor_hidden * (1 - drop_anchor_mask),
                causal_mask,
                cross_causal_mask,
            )
        x = self.output_norm(x)
        gpt_output = self.output_proj(x)
        time_embedding = self.time_embed(time_step).unsqueeze(1)
        output = self.diffusion_head(
            noise_face_latent[:, 1:],
            gpt_output,
            temb=time_embedding if time_step is not None else None,
        )
        return output

    def one_clip_only_inference(
        self,
        per_compute_audio_feature,
        audio_self,
        past_audio_self,
        anchor_latent,
        past_motion,
        gen_frames,
        per_compute_audio_other_feature=None,
        audio_other=None,
        past_audio_other=None,
        noise_scheduler=None,
        num_inference_steps=10,
    ):
        use_pre_compute_audio_feature = True
        audio = audio_self
        n = gen_frames + self.inpainting_length + 1
        audio2face_fea = self.get_audio2face_fea(audio_self, past_audio_self, n)
        audio2face_fea_other = self.get_audio2face_fea_other(audio_other, past_audio_other, n)
        device = audio.device
        if noise_scheduler is not None:
            noise_scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = noise_scheduler.timesteps
        audio_features = audio2face_fea
        audio_other_features = audio2face_fea_other
        if use_pre_compute_audio_feature:
            audio_features = per_compute_audio_feature
            audio_other_features = per_compute_audio_other_feature
        audio_features = audio_features[:, 1:]
        audio_other_features = audio_other_features[:, 1:]
        bs, seq_len, _ = audio_features.shape
        device = audio_features.device
        audio_hidden = self.audio_proj(audio_features)
        audio_other_hidden = self.audio_other_proj(audio_other_features)
        audio_hidden_0 = self.audio_audioother_fusion(torch.cat([audio_hidden * 0, audio_other_hidden * 0], dim=-1))
        audio_hidden_1 = self.audio_audioother_fusion(torch.cat([audio_hidden * 1, audio_other_hidden * 0], dim=-1))
        audio_hidden_2 = self.audio_audioother_fusion(torch.cat([audio_hidden * 0, audio_other_hidden * 1], dim=-1))
        audio_hidden_3 = self.audio_audioother_fusion(torch.cat([audio_hidden * 1, audio_other_hidden * 1], dim=-1))
        anchor_hidden = self.anchor_embed(anchor_latent)
        causal_mask = self.generate_causal_mask(seq_len, device)
        cross_causal_mask = self.generate_cross_causal_mask(seq_len, device)
        face_hidden_last = self.face_embed(past_motion)
        face_hidden = torch.zeros(bs, seq_len, self.hidden_size, device=device)
        face_hidden[:, :self.inpainting_length] = face_hidden_last
        face_outputs = []
        for t in range(self.inpainting_length, seq_len):
            x = face_hidden[:, :t]
            x = torch.cat([x] * 5, dim=0)
            audio_hidden_input = torch.cat(
                [
                    audio_hidden_0[:, :t],
                    audio_hidden_0[:, :t],
                    audio_hidden_1[:, :t],
                    audio_hidden_2[:, :t],
                    audio_hidden_3[:, :t],
                ],
                dim=0,
            )
            anchor_hidden_input = torch.cat(
                [
                    anchor_hidden * 0,
                    anchor_hidden * 1,
                    anchor_hidden * 0,
                    anchor_hidden * 0,
                    anchor_hidden * 1,
                ],
                dim=0,
            )
            for block in self.blocks:
                x = block(
                    x,
                    audio_hidden_input,
                    anchor_hidden_input,
                    causal_mask[:t, :t],
                    cross_causal_mask[:t, :t],
                )
            x_t = self.output_norm(x[:, -1:])
            gpt_output_t = self.output_proj(x_t)
            if noise_scheduler is not None:
                latent_t = torch.randn_like(gpt_output_t[:bs])
                for i, timestep in enumerate(timesteps):
                    t_batch = torch.full((bs,), timestep, device=device, dtype=torch.long)
                    latent_model_input = latent_t
                    time_embedding = self.time_embed(t_batch).unsqueeze(1)
                    output_batch = self.diffusion_head(
                        latent_model_input,
                        gpt_output_t,
                        temb=time_embedding,
                    )
                    noise_pred_uncond, noise_pred_cond_anchor, noise_pred_cond_audio, noise_pred_cond_audio_other, noise_pred_cond_all = output_batch.chunk(5, dim=0)
                    noise_pred = (
                        noise_pred_uncond
                        + self.cfg_audio * (noise_pred_cond_audio - noise_pred_uncond)
                        + self.cfg_audio_other * (noise_pred_cond_audio_other - noise_pred_uncond)
                        + self.cfg_anchor * (noise_pred_cond_anchor - noise_pred_uncond)
                        + self.cfg_all * (noise_pred_cond_all - noise_pred_uncond)
                    )
                    sigma_idx = noise_scheduler.step_index
                    if sigma_idx is None:
                        noise_scheduler._init_step_index(timestep)
                        sigma_idx = noise_scheduler.step_index
                    sigma = noise_scheduler.sigmas[sigma_idx].to(device=device)
                    velocity = (latent_t - noise_pred) / (sigma + 1e-9)
                    latent_t = noise_scheduler.step(
                        velocity, timestep, latent_t, return_dict=False
                    )[0]
                denoised_output_t = latent_t
            face_outputs.append(denoised_output_t)
            if t < seq_len:
                face_hidden[:, t] = self.face_embed(denoised_output_t.squeeze(1))
        output = torch.cat(face_outputs, dim=1)
        return output


    def setup_cuda_graphs(self, num_inference_steps=5):
        """
        Minimal CUDA Graph setup. This does not add trainable parameters.
        """
        if getattr(self, "cuda_graph_enabled", False):
            return

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; cannot setup CUDA Graphs.")

        print("[CUDA Graph] Setting up minimal CUDA Graph runners...")

        self.cuda_graph_gpt = CUDAGraphGPTRunner(
            blocks=self.blocks,
            output_norm=self.output_norm,
            output_proj=self.output_proj,
            hidden_size=self.hidden_size,
            batch_size=5,
            device="cuda",
        )

        self.cuda_graph_diffusion = CUDAGraphDiffusionHeadSingleStep(
            diffusion_head=self.diffusion_head,
            batch_size=5,
            face_dim=self.face_dim,
            hidden_size=self.hidden_size,
            device="cuda",
        )
        self.cuda_graph_diffusion.warmup_and_capture()

        self.cuda_graph_enabled = True
        print("[CUDA Graph] Minimal setup completed.")

    def one_clip_only_inference_cuda_graph(
        self,
        per_compute_audio_feature,
        audio_self,
        past_audio_self,
        anchor_latent,
        past_motion,
        gen_frames,
        per_compute_audio_other_feature=None,
        audio_other=None,
        past_audio_other=None,
        noise_scheduler=None,
        num_inference_steps=10,
    ):
        use_pre_compute_audio_feature = True
        audio = audio_self
        n = gen_frames + self.inpainting_length + 1

        device = audio.device

        # Fast path: inference_cuda_graph() already precomputes Wav2Vec2 features
        # for the full audio. Do not run get_audio2face_fea() again per frame.
        if use_pre_compute_audio_feature:
            audio_features = per_compute_audio_feature
            audio_other_features = per_compute_audio_other_feature
        else:
            audio2face_fea = self.get_audio2face_fea(audio_self, past_audio_self, n)
            audio2face_fea_other = self.get_audio2face_fea_other(audio_other, past_audio_other, n)
            audio_features = audio2face_fea
            audio_other_features = audio2face_fea_other

        audio_features = audio_features[:, 1:]
        audio_other_features = audio_other_features[:, 1:]

        bs, seq_len, _ = audio_features.shape
        device = audio_features.device

        audio_hidden = self.audio_proj(audio_features)
        audio_other_hidden = self.audio_other_proj(audio_other_features)

        # Keep exactly the same four-condition fusion as the stable path.
        audio_hidden_0 = self.audio_audioother_fusion(torch.cat([audio_hidden * 0, audio_other_hidden * 0], dim=-1))
        audio_hidden_1 = self.audio_audioother_fusion(torch.cat([audio_hidden * 1, audio_other_hidden * 0], dim=-1))
        audio_hidden_2 = self.audio_audioother_fusion(torch.cat([audio_hidden * 0, audio_other_hidden * 1], dim=-1))
        audio_hidden_3 = self.audio_audioother_fusion(torch.cat([audio_hidden * 1, audio_other_hidden * 1], dim=-1))

        anchor_hidden = self.anchor_embed(anchor_latent)

        causal_mask = self.generate_causal_mask(seq_len, device)
        cross_causal_mask = self.generate_cross_causal_mask(seq_len, device)

        face_hidden_last = self.face_embed(past_motion)
        face_hidden = torch.zeros(bs, seq_len, self.hidden_size, device=device)
        face_hidden[:, :self.inpainting_length] = face_hidden_last

        face_outputs = []

        for t in range(self.inpainting_length, seq_len):
            x = face_hidden[:, :t]
            x = torch.cat([x] * 5, dim=0)

            audio_hidden_input = torch.cat(
                [
                    audio_hidden_0[:, :t],
                    audio_hidden_0[:, :t],
                    audio_hidden_1[:, :t],
                    audio_hidden_2[:, :t],
                    audio_hidden_3[:, :t],
                ],
                dim=0,
            )

            anchor_hidden_input = torch.cat(
                [
                    anchor_hidden * 0,
                    anchor_hidden * 1,
                    anchor_hidden * 0,
                    anchor_hidden * 0,
                    anchor_hidden * 1,
                ],
                dim=0,
            )

            gpt_output_t = self.cuda_graph_gpt.forward(
                x,
                audio_hidden_input,
                anchor_hidden_input,
                causal_mask[:t, :t],
                cross_causal_mask[:t, :t],
            )

            if noise_scheduler is not None:
                # Important for streaming gen_frames > 1:
                # reset scheduler for every generated frame.
                noise_scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps = noise_scheduler.timesteps

                latent_t = torch.randn_like(gpt_output_t[:bs])

                for i, timestep in enumerate(timesteps):
                    t_batch = torch.full((bs,), timestep, device=device, dtype=torch.long)
                    time_embedding = self.time_embed(t_batch).unsqueeze(1)

                    output_batch = self.cuda_graph_diffusion.forward(
                        latent_t.expand(5, -1, -1),
                        gpt_output_t,
                        temb=time_embedding,
                    )

                    noise_pred_uncond, noise_pred_cond_anchor, noise_pred_cond_audio, noise_pred_cond_audio_other, noise_pred_cond_all = output_batch.chunk(5, dim=0)

                    noise_pred = (
                        noise_pred_uncond
                        + self.cfg_audio * (noise_pred_cond_audio - noise_pred_uncond)
                        + self.cfg_audio_other * (noise_pred_cond_audio_other - noise_pred_uncond)
                        + self.cfg_anchor * (noise_pred_cond_anchor - noise_pred_uncond)
                        + self.cfg_all * (noise_pred_cond_all - noise_pred_uncond)
                    )

                    sigma_idx = noise_scheduler.step_index
                    if sigma_idx is None:
                        noise_scheduler._init_step_index(timestep)
                        sigma_idx = noise_scheduler.step_index

                    sigma = noise_scheduler.sigmas[sigma_idx].to(device=device)
                    velocity = (latent_t - noise_pred) / (sigma + 1e-9)

                    latent_t = noise_scheduler.step(
                        velocity,
                        timestep,
                        latent_t,
                        return_dict=False,
                    )[0]

                denoised_output_t = latent_t

            face_outputs.append(denoised_output_t)

            if t < seq_len:
                face_hidden[:, t] = self.face_embed(denoised_output_t.squeeze(1))

        output = torch.cat(face_outputs, dim=1)
        return output

    def inference_cuda_graph(
        self,
        audio,
        audio_other=None,
        init_motion=None,
        cond_motion=None,
        anchor_motion=None,
        noise_scheduler=None,
        num_inference_steps=10,
    ):
        if not getattr(self, "cuda_graph_enabled", False):
            self.setup_cuda_graphs(num_inference_steps=num_inference_steps)

        inpainting_length = self.inpainting_length
        length = cond_motion.shape[1]
        bs = audio.shape[0] if audio is not None else audio_other.shape[0]
        device = audio.device if audio is not None else audio_other.device

        fake_motion = torch.zeros(bs, length, self.cfg.vae_codebook_size).to(device)
        if cond_motion is not None:
            fake_motion[:, : cond_motion.shape[1]] = cond_motion
        cond_motion = fake_motion

        generator = torch.Generator(device=device)
        generator.manual_seed(self.cfg.seed)

        bs, total_len, c = cond_motion.shape
        window = self.cfg.cbh_window_length
        stride = 1

        rec_all_face = []
        past_motion = cond_motion[:, :inpainting_length, :]
        past_audio = audio[:, : inpainting_length * (self.cfg.audio_fps // self.cfg.pose_fps)]
        past_audio_other = audio_other[:, : inpainting_length * (self.cfg.audio_fps // self.cfg.pose_fps)]

        past_audio_self = None
        past_audio_other = None

        rec_all_face.append(past_motion[:, :inpainting_length])

        # Same audio precompute path as stable inference.
        audio_list = [i.cpu().numpy() for i in audio]
        inputs = self.audio_processor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True).to(audio.device)
        audio2face_fea = self.audio_encoder_face(
            torch.concat([inputs.input_values, torch.zeros([1, 80], device=inputs.input_values.device)], dim=-1)
        )["high_level"]
        audio2face_fea = F.interpolate(
            audio2face_fea.transpose(1, 2),
            scale_factor=(self.cfg.pose_fps / 50),
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)

        audio_other_list = [i.cpu().numpy() for i in audio_other]
        inputs = self.audio_processor(audio_other_list, sampling_rate=16000, return_tensors="pt", padding=True).to(audio.device)
        audio_other2face_fea = self.audio_encoder_face_other(
            torch.concat([inputs.input_values, torch.zeros([1, 80], device=inputs.input_values.device)], dim=-1)
        )["high_level"]
        audio_other2face_fea = F.interpolate(
            audio_other2face_fea.transpose(1, 2),
            scale_factor=(self.cfg.pose_fps / 50),
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)

        for i in range(0, total_len, stride):
            start_idx = i
            end_idx = min(start_idx + window, total_len)
            window_size = end_idx - start_idx

            if window_size < window:
                break

            audio_slice_len = window_size * (self.cfg.audio_fps // self.cfg.pose_fps)
            audio_slice_start = start_idx * (self.cfg.audio_fps // self.cfg.pose_fps)

            audio_slice = audio[:, audio_slice_start : audio_slice_start + audio_slice_len] if audio is not None else None
            audio_slice_other = audio_other[:, audio_slice_start : audio_slice_start + audio_slice_len] if audio_other is not None else None

            out = self.one_clip_only_inference_cuda_graph(
                per_compute_audio_feature=audio2face_fea[:, start_idx:end_idx],
                per_compute_audio_other_feature=audio_other2face_fea[:, start_idx:end_idx],
                past_audio_self=past_audio,
                audio_self=audio_slice,
                past_audio_other=past_audio_other,
                audio_other=audio_slice_other,
                past_motion=past_motion,
                gen_frames=stride,
                anchor_latent=anchor_motion,
                noise_scheduler=noise_scheduler,
                num_inference_steps=num_inference_steps,
            )

            face_latent = out
            past_motion = torch.concat([past_motion, out], dim=1)[:, -inpainting_length:]
            past_audio = audio_slice[:, : -stride * (self.cfg.audio_fps // self.cfg.pose_fps)]
            rec_all_face.append(face_latent)

        rec_all_face = torch.cat(rec_all_face, dim=1)
        return rec_all_face


    def inference(
        self,
        audio,
        audio_other=None,
        init_motion=None,
        cond_motion=None,
        anchor_motion=None,
        noise_scheduler=None,
        num_inference_steps=10,
    ):
        inpainting_length = self.inpainting_length
        length = cond_motion.shape[1]
        bs = audio.shape[0] if audio is not None else audio_other.shape[0]
        device = audio.device if audio is not None else audio_other.device
        fake_motion = torch.zeros(bs, length, self.cfg.vae_codebook_size).to(device)
        if cond_motion is not None:
            fake_motion[:, : cond_motion.shape[1]] = cond_motion
        cond_motion = fake_motion
        generator = torch.Generator(device=device)
        generator.manual_seed(self.cfg.seed)
        bs, total_len, c = cond_motion.shape
        window = self.cfg.cbh_window_length
        pre_frames = self.inpainting_length
        prev_audio_frames = self.cfg.prev_audio_frames
        stride = 1
        rec_all_face = []
        past_motion = cond_motion[:, :pre_frames, :]
        past_audio = audio[:, : pre_frames * (self.cfg.audio_fps // self.cfg.pose_fps)]
        past_audio_other = audio_other[:, : pre_frames * (self.cfg.audio_fps // self.cfg.pose_fps)]
        past_audio_self = None
        past_audio_other = None
        rec_all_face.append(past_motion[:, :inpainting_length])
        audio_list = [i.cpu().numpy() for i in audio]
        inputs = self.audio_processor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True).to(audio.device)
        audio2face_fea = self.audio_encoder_face(
            torch.concat([inputs.input_values, torch.zeros([1, 80], device=inputs.input_values.device)], dim=-1)
        )["high_level"]
        audio2face_fea = F.interpolate(
            audio2face_fea.transpose(1, 2), scale_factor=(self.cfg.pose_fps / 50), mode="linear", align_corners=True
        ).transpose(1, 2)
        audio_other_list = [i.cpu().numpy() for i in audio_other]
        inputs = self.audio_processor(audio_other_list, sampling_rate=16000, return_tensors="pt", padding=True).to(audio.device)
        audio_other2face_fea = self.audio_encoder_face_other(
            torch.concat([inputs.input_values, torch.zeros([1, 80], device=inputs.input_values.device)], dim=-1)
        )["high_level"]
        audio_other2face_fea = F.interpolate(
            audio_other2face_fea.transpose(1, 2), scale_factor=(self.cfg.pose_fps / 50), mode="linear", align_corners=True
        ).transpose(1, 2)
        for i in range(0, total_len, stride):
            start_idx = i
            end_idx = min(start_idx + window, total_len)
            window_size = end_idx - start_idx
            if window_size < window:
                break
            audio_slice_len = window_size * (self.cfg.audio_fps // self.cfg.pose_fps)
            audio_slice_start = start_idx * (self.cfg.audio_fps // self.cfg.pose_fps)
            audio_slice = audio[:, audio_slice_start : audio_slice_start + audio_slice_len] if audio is not None else None
            audio_slice_other = audio_other[:, audio_slice_start : audio_slice_start + audio_slice_len] if audio_other is not None else None
            out = self.one_clip_only_inference(
                per_compute_audio_feature=audio2face_fea[:, start_idx:end_idx],
                per_compute_audio_other_feature=audio_other2face_fea[:, start_idx:end_idx],
                past_audio_self=past_audio,
                audio_self=audio_slice,
                past_audio_other=past_audio_other,
                audio_other=audio_slice_other,
                past_motion=past_motion,
                gen_frames=stride,
                anchor_latent=anchor_motion,
                noise_scheduler=noise_scheduler,
                num_inference_steps=num_inference_steps,
            )
            face_latent = out
            past_motion = torch.concat([past_motion, out], dim=1)[:, -inpainting_length:]
            past_audio = audio_slice[:, : -stride * (self.cfg.audio_fps // self.cfg.pose_fps)]
            rec_all_face.append(face_latent)
        rec_all_face = torch.cat(rec_all_face, dim=1)
        return rec_all_face