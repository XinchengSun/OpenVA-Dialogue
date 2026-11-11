import math
import torch
from torch import nn
from torch.nn import functional as F

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from model.head_animation.LIA.motion_encoder import EqualLinear

class Direction(nn.Module):
    def __init__(self, latent_dim, num_direction):
        super(Direction, self).__init__()

        self.weight = nn.Parameter(torch.randn(latent_dim, num_direction))
        self._cached_q = None
        self._cached_weight_key = None

    def _apply(self, fn, recurse=True):
        # Module dtype/device conversions may replace .data without advancing
        # the parameter version, including a half()->float() round trip.
        self._cached_q = None
        self._cached_weight_key = None
        return super()._apply(fn, recurse=recurse)

    def forward(self, input):
        # The learned direction basis is identical for every inference frame.
        # Keep the original QR and projection arithmetic; only reuse its result
        # while the parameter is unchanged and autograd is disabled.
        cacheable = (
            not self.training and not torch.is_grad_enabled()
            and not self.weight.is_inference()
        )
        if cacheable:
            key = (id(self.weight), self.weight._version,
                   self.weight.device, self.weight.dtype)
            if self._cached_q is None or self._cached_weight_key != key:
                with torch.inference_mode(False), torch.no_grad():
                    self._cached_q, _ = torch.qr(self.weight + 1e-8)
                self._cached_weight_key = key
            Q = self._cached_q
        else:
            self._cached_q = None
            self._cached_weight_key = None
            Q, _ = torch.qr(self.weight + 1e-8)

        if input is None:
            return Q.clone() if cacheable else Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)
            return out


class FlowEstimator(nn.Module):
    def __init__(self, latent_dim, motion_space=20):
        super(FlowEstimator, self).__init__()

        fc = [EqualLinear(latent_dim, latent_dim)]
        for i in range(3):
            fc.append(EqualLinear(latent_dim, latent_dim))
        fc.append(EqualLinear(latent_dim, motion_space))
        self.fc = nn.Sequential(*fc)

        self.direction = Direction(latent_dim, motion_space)

    def forward(self, ref_fea, tgt_fea):
        feats = self.fc(tgt_fea.view(tgt_fea.size(0), -1))

        ref2tgt_mapping = self.direction(feats)
        tgt_latent = ref_fea + ref2tgt_mapping # reference latent code -> target latent code
        
        return tgt_latent
