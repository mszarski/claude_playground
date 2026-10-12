"""Denoiser for the 256-d CSR latent.

GenCAD's original ResNetDiffusion concatenates [x_t, cond], projects to 256 and adds a timestep embedding once at
the input. Here the (timestep + condition) vector modulates *every* residual block through AdaLN (scale, shift,
gate), which is the standard way to make a conditioning signal hard for a deep MLP to ignore. With
`cond_mode="concat"` the network falls back to GenCAD-style input concatenation for ablations.
"""
import math

import torch
from torch import nn


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class AdaLNBlock(nn.Module):
    def __init__(self, d_main, d_hidden, d_cond, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_main, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(d_main, d_hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_hidden, d_main)
        )
        self.mod = nn.Linear(d_cond, 3 * d_main)
        # adaLN-zero: every block starts as identity, which keeps a 10+ block MLP stable at init
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, c):
        scale, shift, gate = self.mod(c).chunk(3, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        return x + gate * self.ff(h)


class LatentDenoiser(nn.Module):
    """f(x_t, t, c) -> prediction (x0 by default) for x in R^{z_dim}."""

    def __init__(self, z_dim=256, d_cond=512, d_main=1024, d_hidden=2048, n_blocks=10, dropout=0.1,
                 cond_mode="adaln"):
        super().__init__()
        assert cond_mode in ("adaln", "concat")
        self.cond_mode = cond_mode
        self.z_dim = z_dim
        self.d_t = d_main
        self.time_mlp = nn.Sequential(nn.Linear(d_main, d_main), nn.SiLU(), nn.Linear(d_main, d_main))
        self.cond_proj = nn.Linear(d_cond, d_main)
        in_dim = z_dim + (d_cond if cond_mode == "concat" else 0)
        self.inp = nn.Linear(in_dim, d_main)
        self.blocks = nn.ModuleList([AdaLNBlock(d_main, d_hidden, d_main, dropout) for _ in range(n_blocks)])
        self.out_norm = nn.LayerNorm(d_main)
        self.out = nn.Linear(d_main, z_dim)

    def forward(self, x, t, cond):
        temb = self.time_mlp(timestep_embedding(t, self.d_t))
        if self.cond_mode == "concat":
            h = self.inp(torch.cat([x, cond], dim=-1)) + temb
            c = temb
        else:
            h = self.inp(x)
            c = temb + self.cond_proj(cond)
        c = nn.functional.silu(c)
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(self.out_norm(h))
