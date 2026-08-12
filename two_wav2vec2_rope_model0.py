from diffusers.models.embeddings import Timesteps, TimestepEmbedding, get_1d_rotary_pos_embed
import diffusers
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import PretrainedConfig
from timm.models.vision_transformer import Mlp
from transformers import  Wav2Vec2Processor, Wav2Vec2Model, PretrainedConfig
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Attention
from typing import Optional, Tuple, Union
import time
import torch.cuda


# ============== CUDA Graph 辅助类 ==============

class CUDAGraphDiffusionRunner:
    """
    使用CUDA Graph加速DiffusionHead的多步去噪推理
    将整个denoising loop封装在一个CUDA Graph中
    """
    def __init__(self, diffusion_head, time_embed, num_inference_steps, batch_size=5, face_dim=512, hidden_size=768, device='cuda'):
        self.diffusion_head = diffusion_head
        self.time_embed = time_embed
        self.num_inference_steps = num_inference_steps
        self.batch_size = batch_size
        self.face_dim = face_dim
        self.hidden_size = hidden_size
        self.device = device
        
        # CUDA Graph相关
        self.graph = None
        self.graph_captured = False
        
        # 静态缓冲区
        self.static_latent = None
        self.static_gpt_output = None
        self.static_temb_all = None
        self.static_output = None
        
    def warmup_and_capture(self, noise_scheduler):
        """预热并捕获CUDA Graph"""
        if self.graph_captured:
            return
            
        # 设置scheduler
        noise_scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        
        # 分配静态缓冲区
        self.static_latent = torch.randn(1, 1, self.face_dim, device=self.device)
        self.static_gpt_output = torch.randn(self.batch_size, 1, self.face_dim, device=self.device)
        
        # 预计算时间嵌入
        timesteps_tensor = torch.tensor(noise_scheduler.timesteps.tolist(), device=self.device, dtype=torch.long)
        self.static_temb_all = self.time_embed(timesteps_tensor)  # [num_steps, hidden]
        
        # 预热
        for _ in range(3):
            latent = self.static_latent.clone()
            for i in range(self.num_inference_steps):
                temb = self.static_temb_all[i:i+1].unsqueeze(1)
                _ = self.diffusion_head(latent.expand(self.batch_size, -1, -1), self.static_gpt_output, temb=temb)
        torch.cuda.synchronize()
        
        self.graph_captured = True
        print(f"[CUDA Graph] DiffusionRunner warmup completed for {self.num_inference_steps} steps")
        
    def run_single_step(self, latent_input, gpt_output, temb):
        """执行单步diffusion（不使用graph，因为每步需要scheduler state）"""
        return self.diffusion_head(latent_input, gpt_output, temb=temb)


class CUDAGraphGPTRunner:
    """
    使用CUDA Graph加速GPT Blocks的推理
    为每个序列长度预先捕获CUDA Graph
    """
    def __init__(self, blocks, output_norm, output_proj, max_seq_len=95, hidden_size=768, batch_size=5, device='cuda'):
        self.blocks = blocks
        self.output_norm = output_norm
        self.output_proj = output_proj
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = device
        
        # 为每个序列长度存储Graph和静态缓冲区
        self.graphs = {}
        self.static_buffers = {}
        
    def warmup_and_capture_for_length(self, seq_len):
        """为特定序列长度捕获CUDA Graph"""
        if seq_len in self.graphs:
            return
            
        # 分配静态缓冲区
        static_x = torch.empty(self.batch_size, seq_len, self.hidden_size, device=self.device)
        static_audio = torch.empty(self.batch_size, seq_len, self.hidden_size, device=self.device)
        static_anchor = torch.empty(self.batch_size, 1, self.hidden_size, device=self.device)
        static_causal_mask = torch.empty(seq_len, seq_len, device=self.device)
        static_cross_mask = torch.empty(seq_len, seq_len, device=self.device)
        
        # 预热
        for _ in range(3):
            x = static_x.clone()
            for block in self.blocks:
                x = block(x, static_audio, static_anchor, static_causal_mask, static_cross_mask)
            x_t = self.output_norm(x[:, -1:])
            _ = self.output_proj(x_t)
        torch.cuda.synchronize()
        
        # 捕获Graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            x = static_x
            for block in self.blocks:
                x = block(x, static_audio, static_anchor, static_causal_mask, static_cross_mask)
            x_t = self.output_norm(x[:, -1:])
            static_output = self.output_proj(x_t)
        
        self.graphs[seq_len] = graph
        self.static_buffers[seq_len] = {
            'x': static_x,
            'audio': static_audio,
            'anchor': static_anchor,
            'causal_mask': static_causal_mask,
            'cross_mask': static_cross_mask,
            'output': static_output
        }
        print(f"[CUDA Graph] GPT Graph captured for seq_len={seq_len}")
        
    def warmup_all_lengths(self, min_len, max_len):
        """预先捕获所有需要的长度"""
        for length in range(min_len, max_len + 1):
            self.warmup_and_capture_for_length(length)
        print(f"[CUDA Graph] All GPT Graphs captured for lengths {min_len} to {max_len}")
        
    def forward(self, x, audio_hidden, anchor_hidden, causal_mask, cross_causal_mask):
        """使用CUDA Graph执行前向传播"""
        seq_len = x.shape[1]
        
        # 如果没有对应长度的Graph，使用普通前向
        if seq_len not in self.graphs:
            # Fallback to normal forward
            for block in self.blocks:
                x = block(x, audio_hidden, anchor_hidden, causal_mask, cross_causal_mask)
            x_t = self.output_norm(x[:, -1:])
            return self.output_proj(x_t)
        
        # 使用CUDA Graph
        buffers = self.static_buffers[seq_len]
        buffers['x'].copy_(x)
        buffers['audio'].copy_(audio_hidden)
        buffers['anchor'].copy_(anchor_hidden)
        buffers['causal_mask'].copy_(causal_mask)
        buffers['cross_mask'].copy_(cross_causal_mask)
        
        # 重放Graph
        self.graphs[seq_len].replay()
        
        return buffers['output'].clone()


class CUDAGraphDiffusionHeadSingleStep:
    """
    使用CUDA Graph加速单步DiffusionHead
    """
    def __init__(self, diffusion_head, batch_size=5, face_dim=512, hidden_size=768, device='cuda'):
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
        """预热并捕获CUDA Graph"""
        # 分配静态缓冲区
        self.static_noisy = torch.empty(self.batch_size, 1, self.face_dim, device=self.device)
        self.static_gpt = torch.empty(self.batch_size, 1, self.face_dim, device=self.device)
        self.static_temb = torch.empty(1, 1, self.hidden_size, device=self.device)
        
        # 预热
        for _ in range(3):
            _ = self.diffusion_head(self.static_noisy, self.static_gpt, temb=self.static_temb)
        torch.cuda.synchronize()
        
        # 捕获Graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.diffusion_head(self.static_noisy, self.static_gpt, temb=self.static_temb)
        
        print("[CUDA Graph] DiffusionHead single-step Graph captured")
        
    def forward(self, noisy_input, gpt_output, temb):
        """使用CUDA Graph执行"""
        self.static_noisy.copy_(noisy_input)
        self.static_gpt.copy_(gpt_output)
        self.static_temb.copy_(temb)
        
        self.graph.replay()
        
        return self.static_output.clone()


# ============== 原有代码 ==============

class WanTimeEmbedding(nn.Module):
    """
    Modified from:
    Wan: Open and Advanced Large-Scale Video Generative Models
    https://huggingface.co/docs/diffusers/main/api/models/wan_transformer_3d
    """
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
    ):
        super().__init__()
        # generate sinusoidal time embeddings
        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        # project to model dimension
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim,
            time_embed_dim=dim,
        )

    def forward(self, timestep: torch.Tensor):  # timestep: (batch,)
        # 1. sinusoidal embedding: (batch, time_freq_dim)
        timestep = self.timesteps_proj(timestep)
        # ensure dtype matches embedder
        emb_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != emb_dtype and emb_dtype != torch.int8:
            timestep = timestep.to(emb_dtype)
        # 2. linear + activation: (batch, dim)
        temb = self.time_embedder(timestep)
        return temb

class RoPEEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=dropout)
        # Compute the frequencies for rotary embeddings: theta_j = 10000^(-2j/d_model)
        theta = 10000 ** (-2 * torch.arange(0, d_model//2, dtype=torch.float) / d_model)
        positions = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        angles = positions * theta
        # Precompute cosines and sines for efficiency
        self.register_buffer('cos_angles', angles.cos())
        self.register_buffer('sin_angles', angles.sin())

    def forward(self, x):
        """
        Apply rotary embeddings to the input tensor.
        Input shape: [bs, seq_len, d_model]
        Output shape: [bs, seq_len, d_model]
        """
        seq_len = x.size(1)
        # Slice precomputed angles to match the sequence length
        cos_angles = self.cos_angles[:seq_len]#.to(x.device)
        sin_angles = self.sin_angles[:seq_len]#.to(x.device)
        # Split input into even and odd indices for pairwise rotation
        x_even = x[:, :, 0::2]  # [bs, seq_len, d_model//2]
        x_odd = x[:, :, 1::2]   # [bs, seq_len, d_model//2]
        # Apply rotary transformation
        x_rot = x.clone()
        x_rot[:, :, 0::2] = x_even * cos_angles - x_odd * sin_angles
        x_rot[:, :, 1::2] = x_even * sin_angles + x_odd * cos_angles
        return self.dropout(x_rot)

class SelfAttention_Rope(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, max_seq_len=128):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        self.rope = RoPEEncoding(d_model, dropout=dropout, max_seq_len=max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ):
        """
        :param x: B x T x d_model input tensor
        :param attn_mask: T x T mask for causal attention
                          for a float mask: values will be added to attention weight
                          for a binary mask: True indicates that the element is not allowed to attend
        :param key_padding_mask: B x T mask
                          for a float mask: values will be added directly to the corresponding key values
                          for a binary mask: True indicates that the corresponding key value will be ignored
        :return: B x T x d_model output tensor
        """
        B, T, _ = x.shape
        
        # Apply RoPE to input
        x_rope = self.rope(x)
        
        # Project to Q, K, V
        q = self.q_proj(x_rope).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        k = self.k_proj(x_rope).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)       # [B, num_heads, T, head_dim]
        
        is_causal = True
        
    
        # Apply scaled dot product attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            #attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)  # [B, T, d_model]
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        return output

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
        """
        :param x: B x T_target x d_model input tensor
        :param cond: B x T_cond x d_cond condition tensor
        :param attn_mask: B * num_heads x L x S mask with L=target sequence length, S=source sequence length
                          for a float mask: values will be added to attention weight
                          for a binary mask: True indicates that the element is not allowed to attend
        :param key_padding_mask: B x S mask
                          for a float mask: values will be added directly to the corresponding key values
                          for a binary mask: True indicates that the corresponding key value will be ignored
        :return: B x T x d_model output tensor
        """
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
        """
        :param x: B x T x d_model input tensor
        :param attn_mask: B * num_heads x L x S mask with L=target sequence length, S=source sequence length
                          for a float mask: values will be added to attention weight
                          for a binary mask: True indicates that the element is not allowed to attend
        :param key_padding_mask: B x S mask
                          for a float mask: values will be added directly to the corresponding key values
                          for a binary mask: True indicates that the corresponding key value will be ignored
        :return: B x T x d_model output tensor
        """
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
    """
    GPT decoder block for Audio2Face generation with causal attention.
    包含 SelfAttention_Rope -> CrossAttention_Rope -> SelfAttention_Pos -> FFN
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1, max_seq_len=128):
        super().__init__()
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.norm4 = nn.LayerNorm(hidden_size)
        self.norm_anchor = nn.LayerNorm(hidden_size)
        
        # Attention layers
        self.self_attn_rope = SelfAttention_Rope(hidden_size, num_heads, dropout, max_seq_len=max_seq_len)
        self.cross_attn_rope = CrossAttention_Rope(hidden_size, hidden_size, num_heads, dropout, max_seq_len=max_seq_len)
        self.self_attn_pos = SelfAttention_Rope(hidden_size, num_heads, dropout, max_seq_len=max_seq_len)
        
        # 与锚点的cross attention
        self.cross_attn_anchor = CrossAttention_Rope(hidden_size, hidden_size, num_heads, dropout, max_seq_len=max_seq_len)
        self.cross_linear_audio = nn.Linear(hidden_size, hidden_size)
        # FFN
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)

    def forward(self, x, audio_features, anchor_hidden, causal_mask=None, cross_causal_mask=None):
        """
        :param x: [bs, seq_len, hidden_size] face latent features
        :param audio_features: [bs, seq_len, hidden_size] audio features
        :param anchor_hidden: [bs, 1, hidden_size] anchor latent features
        :param causal_mask: [seq_len, seq_len] causal mask for self attention
        :param cross_causal_mask: [seq_len, seq_len] causal mask for cross attention
        """
        # Self attention with RoPE
        residual = x
        x = self.norm1(x)
        x = self.self_attn_rope(x, attn_mask=causal_mask)
        x = residual + x
        
        # Cross attention with audio features
        residual = x
        # x = self.norm2(x)
        x = self.cross_attn_rope(x, audio_features, attn_mask=cross_causal_mask)
        #x = residual + self.cross_linear_audio(audio_features)
        
        # Cross attention with anchor (锚点对所有帧都可见)
        residual = x
        x = self.norm_anchor(x)
        x = self.cross_attn_anchor(x, anchor_hidden, attn_mask=None)  # 不需要mask，锚点始终可见
        #x = self.cross_linear_audio(anchor_hidden)  # 不需要mask，锚点始终可见
        x = residual + x
        
        # Self attention with positional encoding
        residual = x
        x = self.norm3(x)
        x = self.self_attn_pos(x, attn_mask=causal_mask)
        x = residual + x
        
        # FFN
        residual = x
        x = self.norm4(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


def make_attention_causal(attn: Wav2Vec2Attention):
    q_proj, k_proj, v_proj, out_proj = attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj
    n_head, head_dim, p = attn.num_heads, attn.head_dim, attn.dropout

    def f(self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        **_):
        B, T, _ = x.shape
        q = q_proj(x).view(B, T, n_head, head_dim).transpose(1, 2)
        k = k_proj(x).view(B, T, n_head, head_dim).transpose(1, 2)
        v = v_proj(x).view(B, T, n_head, head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=p if self.training else 0.0,
            is_causal=True
        )
        y = out_proj(y.transpose(1, 2).reshape(B, T, n_head * head_dim))
        return (y, None, None) if output_attentions else (y, None, None)

    attn.forward = f.__get__(attn, attn.__class__)

from safetensors import safe_open
import os
from transformers import Wav2Vec2Config, Wav2Vec2Model
# from transformers.models.wav2vec2.modeling_wav2vec2 import  Wav2Vec2RopeAttention
class WrapedWav2Vec(nn.Module):
    def __init__(self, layers: int = 1):
        super().__init__()
        
        config = Wav2Vec2Config()
        model = Wav2Vec2Model(config)

        # for layer in model.encoder.layers:
        #     old_state_dict = layer.attention.state_dict()
        #     tmp = Wav2Vec2RopeAttention(
        #         embed_dim=model.config.hidden_size,
        #         num_heads=model.config.num_attention_heads,
        #         dropout=model.config.attention_dropout,
        #         is_causal=False,
        #         config=model.config,
        #     )
        #     tmp.load_state_dict(old_state_dict)
        #     layer.attention= tmp
        
        model.encoder.pos_conv_embed = nn.Identity()
        base = model
        #print(loading_info)
                
        self.feature_extractor = base.feature_extractor
        self.feature_projection = base.feature_projection
        self.encoder = base.encoder
        self.encoder.layers = self.encoder.layers[:layers]

        # for l in self.encoder.layers:
        #     make_attention_causal(l.attention)

    def forward(
        self,
        x: torch.Tensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        **_
    ):
        low = self.feature_extractor(x).transpose(1, 2)
        h, _ = self.feature_projection(low)
        enc = self.encoder(
            h,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return {"low_level": low, "high_level": enc[0]}


class DiffusionBlock(nn.Module):
    """
    Diffusion Block 使用现有的注意力组件
    包含: SelfAttention_Rope -> CrossAttention_Rope (with GPT) -> CrossAttention_Rope (with past frames) -> SelfAttention_Pos -> FFN
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1, max_seq_len=128):
        super().__init__()
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_size,elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size,elementwise_affine=False)
        
        # FFN
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp1 = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)
        
        self.mlp2 = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)
        

        self.adaLN_modulation1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        
        self.adaLN_modulation2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        
        
    def forward(self, hidden_states, gpt_hidden, temb=None):
        """
        :param x: [bs, 1, hidden_size] 当前帧的noisy特征
        :param gpt_hidden: [bs, 1, hidden_size] GPT输出的条件
        :param past_hidden: [bs, T_past, hidden_size] 历史帧的条件
        :param time_emb: [bs, 1, hidden_size] 时间嵌入（可选）
        """
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation1(temb).chunk(3, dim=-1)
        
        shift_mlp2, scale_mlp2, gate_mlp2 = self.adaLN_modulation2(gpt_hidden).chunk(3, dim=-1)
        
        

        # 4. Feed-forward
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_mlp) + shift_mlp)
        ff_output = self.mlp1(norm_hidden_states)
        hidden_states = (hidden_states+ ff_output* gate_mlp)

        norm_hidden_states = (self.norm2(hidden_states.float()) * (1 + scale_mlp2) + shift_mlp2)
        ff_output = self.mlp2(norm_hidden_states)
        hidden_states = (hidden_states+ ff_output* gate_mlp2)

        return hidden_states
        


class DiffusionHead(nn.Module):
    """
    Diffusion Head使用现有组件，用于去噪face latent
    输入:
        - noisy_face_latent: [bs, 1, face_dim] 当前帧的加噪face latent
        - gpt_output: [bs, 1, face_dim] 当前帧的GPT输出
        - past_gt_frames: [bs, T-1, face_dim] 前面所有帧的GT
        - timestep: [bs, 1] 噪声时间步（可选）
    输出:
        - denoised_face_latent: [bs, 1, face_dim] 去噪后的face latent
    """
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
        
        # 输入投影层
        self.noisy_proj = nn.Linear(face_dim, hidden_size)
        self.gpt_proj = nn.Linear(face_dim, hidden_size)
        self.past_proj = nn.Linear(face_dim, hidden_size)
        self.anchor_proj = nn.Linear(face_dim, hidden_size)
        
        
        # Diffusion blocks
        self.blocks = nn.ModuleList([
            DiffusionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                max_seq_len=max_seq_len
            )
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, face_dim)
        
        
    def forward(self, noisy_face_latent, gpt_output, temb=None):
        """
        :param noisy_face_latent: [bs, 1, face_dim] 当前帧的加噪face latent
        :param gpt_output: [bs, 1, face_dim] 当前帧的GPT输出
        :param past_gt_frames: [bs, T-1, face_dim] 前面所有帧的GT
        :param timestep: [bs, 1] 噪声时间步（可选）
        """
        bs = noisy_face_latent.shape[0]
        device = noisy_face_latent.device
                
        # 投影各个输入
        noisy_hidden = self.noisy_proj(noisy_face_latent)  # [bs, 1, hidden_size]
        gpt_hidden = self.gpt_proj(gpt_output)  # [bs, 1, hidden_size]
        
        # 通过Diffusion blocks处理
        x = noisy_hidden
        for block in self.blocks:
            x = block(x, gpt_hidden, temb)
        
        # 输出投影
        x = self.output_norm(x)
        denoised = self.output_proj(x)  # [bs, 1, face_dim]
        
        return denoised

class Audio2FaceGPT(nn.Module):
    """
    GPT自回归模型，从audio特征生成face latent
    输入: 
        - audio2face_fea [bs, 24, 768]
        - anchor_latent [bs, 1, 512] 锚点latent
    输出: face_latent [bs, 24, 512]
    """
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
        diffusion_head_num_layers=6,
    ):
        super().__init__()
        self.cfg = cfg
        self.audio_encoder_face = WrapedWav2Vec(layers=self.cfg.wav2vec_layer) # use how many transformer layers in wav2vec2      
        self.audio_encoder_face_other = WrapedWav2Vec(layers=self.cfg.wav2vec_layer)
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_dim = audio_dim
        self.face_dim = face_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # 输入投影层
        self.audio_proj = nn.Linear(audio_dim, hidden_size)
        self.audio_other_proj = nn.Linear(audio_dim, hidden_size)
        self.audio_audioother_fusion = nn.Linear(2*hidden_size,hidden_size)
        
        self.face_embed = nn.Linear(face_dim, hidden_size)
        self.anchor_embed = nn.Linear(face_dim, hidden_size)  # 新增：锚点投影层
        
        self.time_embed = WanTimeEmbedding(
            dim=hidden_size,
            time_freq_dim=hidden_size,
        )
        # GPT decoder blocks
        self.blocks = nn.ModuleList([
            Audio2FaceGPTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                max_seq_len=max_seq_len
            )
            for _ in range(num_layers)
        ])
        
        # 输出投影层
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, face_dim)
        self.inpainting_length=cfg.cbh_window_length - 2
        
        self.diffusion_head = DiffusionHead(
            face_dim=face_dim,
            hidden_size=hidden_size,
            num_layers=diffusion_head_num_layers,  # 可以调整
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_seq_len=max_seq_len
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


        gates = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]])  # [4,2]
        self.gA = gates[:, 0].view(4, 1, 1, 1).cuda()  # 对 audio_hidden 的系数
        self.gB = gates[:, 1].view(4, 1, 1, 1).cuda()  # 对 audio_other_hidden 的系数

        # 生成causal masks
        self.causal_mask = self.generate_causal_mask(95, 'cuda')
        self.cross_causal_mask = self.generate_cross_causal_mask(95, 'cuda')
        self.face_hidden = torch.zeros(1, 95, hidden_size, device='cuda')
        
        # CUDA Graph 相关
        self.cuda_graph_enabled = False
        self.cuda_graph_gpt = None
        self.cuda_graph_diffusion = None
    
    def generate_causal_mask(self, seq_len, device):
        """生成causal mask，上三角为-inf"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def generate_cross_causal_mask(self, seq_len, device):
        """生成cross attention的causal mask，确保第i帧只能看到前i帧的音频"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    

    def get_audio2face_fea(self,audio,prev_audio,n):
        if prev_audio is not None:
            audio = torch.cat([prev_audio, audio], dim=1)
        # audio_list = [i.cpu().numpy() for i in audio]
        # inputs = self.audio_processor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True).to(audio.device)#注意了，audio在这里经过了一次norm
        # audio2face_fea = self.audio_encoder_face(inputs.input_values)["high_level"]
        #audio2face_fea = self.audio_encoder_face((audio-audio.mean(dim=1,keepdim=True))/audio.std(dim=1,keepdim=True))["high_level"]
        audio2face_fea = self.audio_encoder_face(audio)["high_level"]
        audio2face_fea = F.interpolate(
            audio2face_fea.transpose(1, 2), scale_factor=(self.cfg.pose_fps/50), mode="linear", align_corners=True
        ).transpose(1, 2)
        
        # 确保输出长度为n
        if audio2face_fea.shape[1] > n:
            audio2face_fea = audio2face_fea[:, -n:] if prev_audio is not None else audio2face_fea[:, :n]
        elif audio2face_fea.shape[1] < n:
            # 循环填充直到达到n
            current_len = audio2face_fea.shape[1]
            while audio2face_fea.shape[1] < n:
                pad_needed = n - audio2face_fea.shape[1]
                pad_len = min(pad_needed, current_len)  # 每次最多填充原始长度
                audio2face_fea = torch.cat([audio2face_fea, audio2face_fea[:, -pad_len:]], dim=1)
        
        return audio2face_fea

    def get_audio2face_fea_other(self,audio,prev_audio,n):
        if prev_audio is not None:
            audio = torch.cat([prev_audio, audio], dim=1)
        # audio_list = [i.cpu().numpy() for i in audio]
        # inputs = self.audio_processor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True).to(audio.device)#注意了，audio在这里经过了一次norm
        # audio2face_fea = self.audio_encoder_face(inputs.input_values)["high_level"]
        #audio2face_fea = self.audio_encoder_face((audio-audio.mean(dim=1,keepdim=True))/audio.std(dim=1,keepdim=True))["high_level"]
        audio2face_fea = self.audio_encoder_face_other(audio)["high_level"]
        audio2face_fea = F.interpolate(
            audio2face_fea.transpose(1, 2), scale_factor=(self.cfg.pose_fps/50), mode="linear", align_corners=True
        ).transpose(1, 2)
        
        # 确保输出长度为n
        if audio2face_fea.shape[1] > n:
            audio2face_fea = audio2face_fea[:, -n:] if prev_audio is not None else audio2face_fea[:, :n]
        elif audio2face_fea.shape[1] < n:
            # 循环填充直到达到n
            current_len = audio2face_fea.shape[1]
            while audio2face_fea.shape[1] < n:
                pad_needed = n - audio2face_fea.shape[1]
                pad_len = min(pad_needed, current_len)  # 每次最多填充原始长度
                audio2face_fea = torch.cat([audio2face_fea, audio2face_fea[:, -pad_len:]], dim=1)
        
        return audio2face_fea

    def forward(self,face_latent_gt,noise_face_latent,time_step, audio,audio_other,prev_audio,prev_audio_other, anchor_latent):
        """
        :param audio_features: [bs, seq_len, 768] 音频特征
        :param anchor_latent: [bs, 1, 512] 锚点latent
        :param face_latent_gt: [bs, seq_len, 512] ground truth face latent (用于teacher forcing)
        :return: [bs, seq_len, 512] 预测的face latent
        """
        bs, n, _ = face_latent_gt.shape

        audio2face_fea = self.get_audio2face_fea(audio,prev_audio,n)
        audio2face_fea_other = self.get_audio2face_fea_other(audio_other,prev_audio_other,n)
        device = audio2face_fea.device
        
        bs, seq_len, _ = audio2face_fea.shape
        # 投影音频特征和锚点
        audio_hidden = self.audio_proj(audio2face_fea)  # [bs, seq_len, hidden_size]        
        audio_hidden = audio_hidden[:,1:]
        drop_audio_mask = torch.rand(bs,seq_len-1,1,device =face_latent_gt.device)<self.drop_audio
        drop_audio_mask = drop_audio_mask.float()
        audio_hidden = audio_hidden*(1-drop_audio_mask)
        
        
        audio_other_hidden = self.audio_other_proj(audio2face_fea_other)
        audio_other_hidden = audio_other_hidden[:,1:]
        drop_audio_other_mask = torch.rand(bs,seq_len-1,1,device =face_latent_gt.device)<self.drop_audio_other
        drop_audio_other_mask = drop_audio_other_mask.float()
        audio_other_hidden = audio_other_hidden*(1-drop_audio_other_mask)
        
        audio_hidden = self.audio_audioother_fusion(torch.cat([audio_hidden,audio_other_hidden],dim=-1))
        
        anchor_hidden = self.anchor_embed(anchor_latent)  # [bs, 1, hidden_size]
        
        # 生成causal masks
        causal_mask = self.generate_causal_mask(seq_len-1, device)
        cross_causal_mask = self.generate_cross_causal_mask(seq_len-1, device)
        
        
        face_hidden = self.face_embed(face_latent_gt[:,:-1])  # [bs, seq_len, hidden_size]
            

        drop_anchor_mask = torch.rand(bs,1,1,device =face_hidden.device)<self.drop_anchor
        drop_anchor_mask = drop_anchor_mask.float()
        
        x = face_hidden
        for block in self.blocks:
            x = block(x, 
                      audio_hidden, 
                      anchor_hidden*(1-drop_anchor_mask), 
                      causal_mask, 
                      cross_causal_mask)
        
        # 输出投影
        x = self.output_norm(x)
        gpt_output = self.output_proj(x)  # [bs, seq_len, face_dim]
        

        time_embedding = self.time_embed(time_step).unsqueeze(1)

        

        
        # 通过diffusion head去噪
        output = self.diffusion_head(
            noise_face_latent[:,1:], 
            gpt_output, 
            temb=time_embedding if time_step is not None else None,
        )
        

        
        return output


    def one_clip_only_inference(self, per_compute_audio_feature,audio_self,past_audio_self, anchor_latent,past_motion,gen_frames,per_compute_audio_other_feature=None,audio_other=None,past_audio_other=None,noise_scheduler=None,num_inference_steps=10):
        """
        :param audio_features: [bs, seq_len, 768] 音频特征
        :param anchor_latent: [bs, 1, 512] 锚点latent
        :param face_latent_gt: [bs, seq_len, 512] ground truth face latent (用于teacher forcing)
        :return: [bs, seq_len, 512] 预测的face latent
        """
        use_pre_compute_audio_feature = False
        audio = audio_self
        n = gen_frames + self.inpainting_length + 1

        time1 = time.time()
        
        audio2face_fea = self.get_audio2face_fea(audio_self,past_audio_self,n)
        audio2face_fea_other = self.get_audio2face_fea_other(audio_other,past_audio_other,n)
        
        # time2 = time.time()
        # print("get_audio2face_fea time", time2 - time1)
        
        
        device = audio.device
        # 设置噪声调度器
        if noise_scheduler is not None:
            noise_scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = noise_scheduler.timesteps
        
        audio_features = audio2face_fea
        audio_other_features = audio2face_fea_other
        if use_pre_compute_audio_feature:
            audio_features = per_compute_audio_feature
            audio_other_features = per_compute_audio_other_feature
        audio_features = audio_features[:,1:]
        audio_other_features = audio_other_features[:,1:]
        bs, seq_len, _ = audio_features.shape
        device = audio_features.device
        
        # 投影音频特征和锚点
        audio_hidden = self.audio_proj(audio_features)  # [bs, seq_len, hidden_size]
        audio_other_hidden = self.audio_other_proj(audio_other_features)
        

        # [4, bs, seq_len, 2H]
        fusion_in = torch.cat([
            audio_hidden.unsqueeze(0) * self.gA,
            audio_other_hidden.unsqueeze(0) * self.gB
        ], dim=-1)

        # 单次前向：[4*bs, seq_len, 2H] -> [4*bs, seq_len, H]
        fusion_out = self.audio_audioother_fusion(
            fusion_in.reshape(-1, seq_len, fusion_in.size(-1))
        )

        # 还原为四路输出：[4, bs, seq_len, H]
        audio_hidden_0, audio_hidden_1, audio_hidden_2, audio_hidden_3 = \
            fusion_out.reshape(4, bs, seq_len, -1).unbind(0)
        
        anchor_hidden = self.anchor_embed(anchor_latent)  # [bs, 1, hidden_size]
        

        causal_mask = self.causal_mask
        cross_causal_mask = self.cross_causal_mask

        face_hidden_last = self.face_embed(past_motion)  # [bs, seq_len, hidden_size]
            
        face_hidden = self.face_hidden
        
        face_hidden[:, :self.inpainting_length] = face_hidden_last
        # 使用锚点初始化第一帧

        # time4 = time.time()
        # print("init face_hidden time", time4 - time1)
        face_outputs = []
        
        for t in range(self.inpainting_length, seq_len):
            # 通过所有decoder blocks
            x = face_hidden[:, :t]  # 只使用到当前时刻的序列
            x = torch.cat([x]*5,dim=0)
            audio_hidden_input = torch.cat([audio_hidden_0[:, :t],audio_hidden_0[:, :t],audio_hidden_1[:, :t],audio_hidden_2[:, :t],audio_hidden_3[:, :t]],dim=0)
            anchor_hidden_input = torch.cat([anchor_hidden*0,anchor_hidden*1,anchor_hidden*0,anchor_hidden*0,anchor_hidden*1],dim=0)
            for block in self.blocks:
                x = block(x, 
                          audio_hidden_input,
                          anchor_hidden_input, 
                            causal_mask[:t, :t],
                            cross_causal_mask[:t, :t])
            
            # 输出当前时间步
            x_t = self.output_norm(x[:, -1:])  # 取最后一个时间步
            gpt_output_t = self.output_proj(x_t)  # [bs, 1, face_dim]
            # time6 = time.time()
            # print("gpt_output_t time", time6 - time1)
            if noise_scheduler is not None:
                timesteps_list = noise_scheduler.timesteps  # or timesteps.tolist()
                # build a tensor of shape [num_steps] and compute embeddings once
                timesteps_tensor = torch.tensor(timesteps_list, device=device, dtype=torch.long)
                temb_all = self.time_embed(timesteps_tensor)  # [num_steps, hidden]
                # pull sigmas as tensor on device
                latent_t = torch.randn_like(gpt_output_t[:bs])
                # Scale model input
                latent_model_input = latent_t

                # Denoising loop
                for i, timestep in enumerate(timesteps):
                    # 通过diffusion head去噪
                    time_embedding = temb_all[i].unsqueeze(0).unsqueeze(1)
                    output_batch = self.diffusion_head(
                        latent_model_input,
                        gpt_output_t, 
                        temb=time_embedding,
                    )
                    
                    
                    # Split predictions using chunk
                    noise_pred_uncond, noise_pred_cond_anchor,noise_pred_cond_audio,noise_pred_cond_audio_other,noise_pred_cond_all = output_batch.chunk(5, dim=0)
                    
                    # Apply CFG in batch
                    noise_pred = noise_pred_uncond + \
                        self.cfg_audio * (noise_pred_cond_audio - noise_pred_uncond) + \
                        self.cfg_audio_other * (noise_pred_cond_audio_other - noise_pred_uncond) + \
                        self.cfg_anchor * (noise_pred_cond_anchor - noise_pred_uncond) + \
                        self.cfg_all * (noise_pred_cond_all - noise_pred_uncond)
                
                    
                    # # 执行一步去噪
                    # latent_t = noise_scheduler.step(
                    #     noise_pred, timestep, latent_t, return_dict=False
                    # )[0]
                    
                    sigma_idx = noise_scheduler.step_index
                    if sigma_idx is None: 
                        noise_scheduler._init_step_index(timestep)
                        sigma_idx = noise_scheduler.step_index
                    sigma = noise_scheduler.sigmas[sigma_idx].to(device=device)
                    velocity = (latent_t - noise_pred) / (sigma + 1e-9)
                    
                    latent_t = noise_scheduler.step(
                        velocity, timestep, latent_t, return_dict=False
                    )[0]
                
                
                # 使用去噪后的结果
                denoised_output_t = latent_t

            
            face_outputs.append(denoised_output_t)
            time3 = time.time()
            print("denoised_output_t time", time3 - time1)

            
            # 更新下一时间步的输入
            if t < seq_len:
                face_hidden[:, t] = self.face_embed(denoised_output_t.squeeze(1))
        
        output = torch.cat(face_outputs, dim=1)  
        return output

    def setup_cuda_graphs(self, num_inference_steps=5):
        """
        初始化CUDA Graphs以加速推理
        需要在第一次推理前调用
        """
        print("[CUDA Graph] Setting up CUDA Graphs...")
        
        # 1. 设置GPT Blocks的CUDA Graph
        self.cuda_graph_gpt = CUDAGraphGPTRunner(
            blocks=self.blocks,
            output_norm=self.output_norm,
            output_proj=self.output_proj,
            max_seq_len=95,
            hidden_size=self.hidden_size,
            batch_size=5,  # 5个CFG条件
            device='cuda'
        )
        # 预先捕获常用长度的Graph
        self.cuda_graph_gpt.warmup_all_lengths(self.inpainting_length, 94)
        
        # 2. 设置DiffusionHead的CUDA Graph
        self.cuda_graph_diffusion = CUDAGraphDiffusionHeadSingleStep(
            diffusion_head=self.diffusion_head,
            batch_size=5,  # 5个CFG条件
            face_dim=self.face_dim,
            hidden_size=self.hidden_size,
            device='cuda'
        )
        self.cuda_graph_diffusion.warmup_and_capture()
        
        self.cuda_graph_enabled = True
        print("[CUDA Graph] Setup completed!")
        
    def one_clip_only_inference_cuda_graph(self, per_compute_audio_feature, audio_self, past_audio_self, 
                                            anchor_latent, past_motion, gen_frames,
                                            per_compute_audio_other_feature=None, audio_other=None, 
                                            past_audio_other=None, noise_scheduler=None, num_inference_steps=10):
        """
        使用CUDA Graph加速的推理方法
        """
        if not self.cuda_graph_enabled:
            raise RuntimeError("CUDA Graphs not initialized. Call setup_cuda_graphs() first.")
            
        use_pre_compute_audio_feature = False
        audio = audio_self
        n = gen_frames + self.inpainting_length + 1
        torch.cuda.synchronize()
        time1 = time.time()
        
        audio2face_fea = self.get_audio2face_fea(audio_self, past_audio_self, n)
        audio2face_fea_other = self.get_audio2face_fea_other(audio_other, past_audio_other, n)
        
        device = audio.device
        # 设置噪声调度器
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
        
        # 投影音频特征和锚点
        audio_hidden = self.audio_proj(audio_features)
        audio_other_hidden = self.audio_other_proj(audio_other_features)
        
        # [4, bs, seq_len, 2H]
        fusion_in = torch.cat([
            audio_hidden.unsqueeze(0) * self.gA,
            audio_other_hidden.unsqueeze(0) * self.gB
        ], dim=-1)

        # 单次前向：[4*bs, seq_len, 2H] -> [4*bs, seq_len, H]
        fusion_out = self.audio_audioother_fusion(
            fusion_in.reshape(-1, seq_len, fusion_in.size(-1))
        )

        # 还原为四路输出：[4, bs, seq_len, H]
        audio_hidden_0, audio_hidden_1, audio_hidden_2, audio_hidden_3 = \
            fusion_out.reshape(4, bs, seq_len, -1).unbind(0)
        
        anchor_hidden = self.anchor_embed(anchor_latent)

        causal_mask = self.causal_mask
        cross_causal_mask = self.cross_causal_mask

        face_hidden_last = self.face_embed(past_motion)
        face_hidden = self.face_hidden
        face_hidden[:, :self.inpainting_length] = face_hidden_last

        # 预计算时间嵌入（只计算一次）
        if noise_scheduler is not None:
            timesteps_tensor = torch.tensor(noise_scheduler.timesteps.tolist(), device=device, dtype=torch.long)
            temb_all = self.time_embed(timesteps_tensor)  # [num_steps, hidden]

        face_outputs = []
        
        # 调试信息
        print(f"[DEBUG CUDA Graph] n={n}, seq_len={seq_len}, inpainting_length={self.inpainting_length}, gen_frames={gen_frames}")
        print(f"[DEBUG CUDA Graph] audio2face_fea.shape={audio2face_fea.shape}, audio_features.shape={audio_features.shape}")
        print(f"[DEBUG CUDA Graph] loop range: range({self.inpainting_length}, {seq_len})")
        
        for t in range(self.inpainting_length, seq_len):
            # 准备输入
            x = face_hidden[:, :t]
            x = torch.cat([x] * 5, dim=0)
            audio_hidden_input = torch.cat([
                audio_hidden_0[:, :t], audio_hidden_0[:, :t],
                audio_hidden_1[:, :t], audio_hidden_2[:, :t], audio_hidden_3[:, :t]
            ], dim=0)
            anchor_hidden_input = torch.cat([
                anchor_hidden * 0, anchor_hidden * 1,
                anchor_hidden * 0, anchor_hidden * 0, anchor_hidden * 1
            ], dim=0)
            
            # 使用CUDA Graph加速GPT blocks
            gpt_output_t = self.cuda_graph_gpt.forward(
                x, audio_hidden_input, anchor_hidden_input,
                causal_mask[:t, :t], cross_causal_mask[:t, :t]
            )
            
            if noise_scheduler is not None:
                latent_t = torch.randn_like(gpt_output_t[:bs])

                # Denoising loop - 使用CUDA Graph加速每一步
                for i, timestep in enumerate(timesteps):
                    time_embedding = temb_all[i].unsqueeze(0).unsqueeze(1)
                    
                    # 使用CUDA Graph加速diffusion head
                    output_batch = self.cuda_graph_diffusion.forward(
                        latent_t.expand(5, -1, -1),
                        gpt_output_t,
                        temb=time_embedding,
                    )
                    
                    # CFG组合
                    noise_pred_uncond, noise_pred_cond_anchor, noise_pred_cond_audio, \
                        noise_pred_cond_audio_other, noise_pred_cond_all = output_batch.chunk(5, dim=0)
                    
                    noise_pred = noise_pred_uncond + \
                        self.cfg_audio * (noise_pred_cond_audio - noise_pred_uncond) + \
                        self.cfg_audio_other * (noise_pred_cond_audio_other - noise_pred_uncond) + \
                        self.cfg_anchor * (noise_pred_cond_anchor - noise_pred_uncond) + \
                        self.cfg_all * (noise_pred_cond_all - noise_pred_uncond)
                    
                    # Scheduler step
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
            
            # 更新下一时间步的输入
            if t < seq_len:
                face_hidden[:, t] = self.face_embed(denoised_output_t.squeeze(1))
        torch.cuda.synchronize()
        time3 = time.time()
        print(f"[CUDA Graph] Total inference time: {time3 - time1:.4f}s")
        
        output = torch.cat(face_outputs, dim=1)
        return output

    def inference_cuda_graph(self,
            audio, audio_other=None, 
            init_motion=None, cond_motion=None,
            anchor_motion=None,
            noise_scheduler=None,
            num_inference_steps=10,
            ):
        """
        使用CUDA Graph加速的完整推理方法
        """
        if not self.cuda_graph_enabled:
            print("[Warning] CUDA Graphs not initialized, calling setup_cuda_graphs()...")
            self.setup_cuda_graphs(num_inference_steps)
        
        inpainting_length = self.inpainting_length
        
        length = cond_motion.shape[1]
        bs = audio.shape[0] if audio is not None else audio_other.shape[0]
        device = audio.device if audio is not None else audio_other.device
        fake_motion = torch.zeros(bs, length, self.cfg.vae_codebook_size).to(device)
        if cond_motion is not None:
            fake_motion[:, :cond_motion.shape[1]] = cond_motion 
        cond_motion = fake_motion

        generator = torch.Generator(device=device)
        generator.manual_seed(self.cfg.seed)

        bs, total_len, c = cond_motion.shape
        window = self.cfg.cbh_window_length
        pre_frames = self.inpainting_length
        stride = 1
        rec_all_face = []
        past_motion = cond_motion[:, :pre_frames, :]
        past_audio = torch.zeros([1, 80], device=audio.device)
        past_audio_other = torch.zeros([1, 80], device=audio.device)
        rec_all_face.append(past_motion[:, :inpainting_length])
        
        # 预处理音频特征
        audio_list = [i.cpu().numpy() for i in audio]
        inputs = self.audio_processor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True).to(audio.device)
        audio2face_fea = self.audio_encoder_face(torch.concat([inputs.input_values, torch.zeros([1, 80], device=inputs.input_values.device)], dim=-1))["high_level"]
        audio2face_fea = F.interpolate(
            audio2face_fea.transpose(1, 2), scale_factor=(self.cfg.pose_fps/50), mode="linear", align_corners=True
        ).transpose(1, 2)

        audio_other_list = [i.cpu().numpy() for i in audio_other]
        inputs = self.audio_processor(audio_other_list, sampling_rate=16000, return_tensors="pt", padding=True).to(audio.device)
        audio_other2face_fea = self.audio_encoder_face_other(torch.concat([inputs.input_values, torch.zeros([1, 80], device=inputs.input_values.device)], dim=-1))["high_level"]
        audio_other2face_fea = F.interpolate(
            audio_other2face_fea.transpose(1, 2), scale_factor=(self.cfg.pose_fps/50), mode="linear", align_corners=True
        ).transpose(1, 2)

        for i in range(0, total_len, stride):
            start_idx = i
            end_idx = min(start_idx + window, total_len)
            window_size = end_idx - start_idx
            if window_size < window:
                break
            
            audio_slice_len = window_size * (self.cfg.audio_fps // self.cfg.pose_fps)
            audio_slice_start = start_idx * (self.cfg.audio_fps // self.cfg.pose_fps)
            audio_slice = audio[:, audio_slice_start:audio_slice_start + audio_slice_len] if audio is not None else None
            audio_slice_other = audio_other[:, audio_slice_start:audio_slice_start + audio_slice_len] if audio_other is not None else None
            
            # 使用CUDA Graph加速的推理
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
            past_audio = audio_slice[:, :-stride*(self.cfg.audio_fps // self.cfg.pose_fps)]
            past_audio_other = audio_slice_other[:, :-stride*(self.cfg.audio_fps // self.cfg.pose_fps)]  # 同步更新past_audio_other
            rec_all_face.append(face_latent)

        rec_all_face = torch.cat(rec_all_face, dim=1)
        return rec_all_face

    def inference(self,
            audio, audio_other=None, 
            init_motion=None, cond_motion=None,
            anchor_motion=None,
            noise_scheduler=None,
            num_inference_steps=10,
            ):
        
        inpainting_length=self.inpainting_length
        
        length = cond_motion.shape[1]
        bs = audio.shape[0] if audio is not None else audio_other.shape[0]
        device = audio.device if audio is not None else audio_other.device
        fake_motion = torch.zeros(bs, length, self.cfg.vae_codebook_size).to(device)
        if cond_motion is not None:
            fake_motion[:, :cond_motion.shape[1]] = cond_motion 
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
        past_audio = torch.zeros([1,80],device=audio.device)
        past_audio_other = torch.zeros([1,80],device=audio.device)
        rec_all_face.append(past_motion[:, :inpainting_length])
        # print("total_len", total_len, window, pre_frames, stride)
        audio_list = [i.cpu().numpy() for i in audio]
        inputs = self.audio_processor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True).to(audio.device)
        audio2face_fea = self.audio_encoder_face(torch.concat([inputs.input_values,torch.zeros([1,80],device=inputs.input_values.device)],dim=-1))["high_level"]
        audio2face_fea = F.interpolate(
            audio2face_fea.transpose(1, 2), scale_factor=(self.cfg.pose_fps/50), mode="linear", align_corners=True
        ).transpose(1, 2)
      

        audio_other_list = [i.cpu().numpy() for i in audio_other]
        inputs = self.audio_processor(audio_other_list, sampling_rate=16000, return_tensors="pt", padding=True).to(audio.device)
        audio_other2face_fea = self.audio_encoder_face_other(torch.concat([inputs.input_values,torch.zeros([1,80],device=inputs.input_values.device)],dim=-1))["high_level"]
        audio_other2face_fea = F.interpolate(
            audio_other2face_fea.transpose(1, 2), scale_factor=(self.cfg.pose_fps/50), mode="linear", align_corners=True
        ).transpose(1, 2)

        
        for i in range(0, total_len, stride):
            start_idx = i
            end_idx = min(start_idx + window, total_len)
            window_size = end_idx - start_idx
            if window_size < window:
                break
            # prepare window inputs
            audio_slice_len = window_size * (self.cfg.audio_fps // self.cfg.pose_fps)
            audio_slice_start = start_idx * (self.cfg.audio_fps // self.cfg.pose_fps)
            audio_slice = audio[:, audio_slice_start:audio_slice_start + audio_slice_len] if audio is not None else None
            audio_slice_other = audio_other[:, audio_slice_start:audio_slice_start + audio_slice_len] if audio_other is not None else None
            # print("audio_slice", audio_slice.shape, past_motion.shape, window_size)
            # call one_clip_only_inference for this window
            out = self.one_clip_only_inference(
                per_compute_audio_feature = audio2face_fea[:,start_idx:end_idx],
                per_compute_audio_other_feature = audio_other2face_fea[:,start_idx:end_idx],
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
            past_motion = torch.concat([past_motion,out],dim=1)[:,-inpainting_length:]
            past_audio = audio_slice[:,:-stride*(self.cfg.audio_fps // self.cfg.pose_fps)] #比如第一次是取[0,24]的音频，那第二次是取[17,41]的音频，所以我们只要[0,17]concat
            past_audio_other = audio_slice_other[:,:-stride*(self.cfg.audio_fps // self.cfg.pose_fps)]  # 同步更新past_audio_other
            rec_all_face.append(face_latent)

        rec_all_face = torch.cat(rec_all_face, dim=1)
        return rec_all_face

from types import SimpleNamespace


if __name__ == "__main__":
    # 必需的 cfg（仅包含代码里会用到的键）
    cfg = SimpleNamespace(
        # 推理窗口与采样率
        audio_fps=16000,        # 你给的是 16,000
        pose_fps=25,
        cbh_window_length=96,

        # 采样/随机性与特征尺寸
        seed=222,
        vae_codebook_size=512,

        # 条件引导系数（在采样时组合多个条件）
        cfg_all=1.0,
        cfg_audio=0.5,
        cfg_audio_other=0.5,
        cfg_anchor=0.0,
        cfg_audio_anchor=1.0,

        # dropout 概率（在训练/推理中用于随机丢弃条件）
        drop_gpt=0.1,
        drop_audio=0.1,
        drop_audio_other=0.1,
        drop_anchor=0.1,

        # 编码器层数（用于 WrapedWav2Vec）
        wav2vec_layer=8,

        # 历史音频帧（你的列表是 32）
        prev_audio_frames=32,
    )
    cfg.num_layers=12
    cfg.diffusion_head_num_layers=6
    model = Audio2FaceGPT(cfg=cfg)
    model.eval()
    model.to("cuda")
    
    audio = torch.randn(1, 16000*10).to("cuda")
    audio_other = torch.randn(1, 16000*10).to("cuda")
    init_motion = torch.randn(1, 1, 512).to("cuda")
    cond_motion = torch.randn(1, 1000, 512).to("cuda")
    anchor_motion = torch.randn(1, 1, 512).to("cuda")
    noise_scheduler = diffusers.FlowMatchEulerDiscreteScheduler()
    num_inference_steps = 5
    
    # # ============== 测试普通推理 ==============
    # print("\n" + "="*50)
    # print("Testing NORMAL inference...")
    # print("="*50)
    # with torch.no_grad():
    #     torch.cuda.synchronize()
    #     t1 = time.time()
    #     out_normal = model.inference(audio, audio_other, init_motion, cond_motion, anchor_motion, noise_scheduler, num_inference_steps)
    #     torch.cuda.synchronize()
    #     t2 = time.time()
    # print(f"Normal inference output shape: {out_normal.shape}")
    # print(f"Normal inference time: {t2-t1:.4f}s")
    
    # ============== 测试CUDA Graph加速推理 ==============
    print("\n" + "="*50)
    print("Testing CUDA GRAPH inference...")
    print("="*50)
    
    # 初始化CUDA Graphs（只需要调用一次）
    model.setup_cuda_graphs(num_inference_steps)
    
    # 预热一次
    print("\nWarmup run...")
    with torch.no_grad():
        _ = model.inference_cuda_graph(audio, audio_other, init_motion, cond_motion, anchor_motion, noise_scheduler, num_inference_steps)
    
    # 正式测试
    print("\nBenchmark run...")
    with torch.no_grad():
        torch.cuda.synchronize()
        t1 = time.time()
        out_cuda_graph = model.inference_cuda_graph(audio, audio_other, init_motion, cond_motion, anchor_motion, noise_scheduler, num_inference_steps)
        torch.cuda.synchronize()
        t2 = time.time()
    print(f"CUDA Graph inference output shape: {out_cuda_graph.shape}")
    print(f"CUDA Graph inference time: {t2-t1:.4f}s")
    
    # ============== 对比结果 ==============
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"CUDA Graph output shape: {out_cuda_graph.shape}")
    # 注意：由于随机性，输出值可能不完全相同
