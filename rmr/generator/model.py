"""The motion generator: a small flow-matching transformer (21.8M parameters by default).

Each frame is one token: the noisy 9-DoF motion x_t (normalised), the plan interpolated to that frame
(normalised) times a ``has_plan`` flag, and the flag itself. The plan is concatenated to every frame
rather than cross-attended, so the model cannot ignore it. The flow time ``t`` enters every block through
AdaLN (zero-initialised, so each block starts as the identity). The output is the velocity
``v = x1 - x0`` (noise minus data).

Reference: ``generator/model.py`` in pham-tuan-binh/reachy-motion-generator (Apache-2.0).
"""
import math

import torch
import torch.nn as nn

N_DOF, N_PLAN = 9, 8


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def timestep_embedding(t, dim=256):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    angles = t[:, None] * freqs[None] * 1000.0
    return torch.cat([angles.sin(), angles.cos()], -1)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c, pad):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(c).unsqueeze(1).chunk(6, -1)
        h = self.norm1(x) * (1 + scale1) + shift1
        x = x + gate1 * self.attn(h, h, h, key_padding_mask=pad, need_weights=False)[0]
        h = self.norm2(x) * (1 + scale2) + shift2
        return x + gate2 * self.mlp(h)


class MotionGenerator(nn.Module):
    def __init__(self, d=384, heads=6, layers=8, maxlen=720):
        super().__init__()
        self.config = dict(d=d, heads=heads, layers=layers, maxlen=maxlen)
        self.maxlen = maxlen
        self.inp = nn.Linear(N_DOF + N_PLAN + 1, d)
        self.pos = nn.Parameter(torch.randn(1, maxlen, d) * 0.02)
        self.temb = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, N_DOF)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, plan, has, pad):
        """x (B,T,9) noisy motion; t (B,) flow time (1 = noise); plan (B,T,8); has (B,1,1); pad (B,T), True = padding."""
        cond = torch.cat([plan * has, has.expand(-1, x.shape[1], 1)], -1)
        h = self.inp(torch.cat([x, cond], -1)) + self.pos[:, :x.shape[1]]
        c = self.temb(timestep_embedding(t))
        for block in self.blocks:
            h = block(h, c, pad)
        return self.out(self.norm(h))
