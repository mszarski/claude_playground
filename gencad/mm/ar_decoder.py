"""Autoregressive CAD decoder over GenCAD's frozen CSR latent space.

GenCAD's CSR decoder is non-autoregressive: its transformer input is a constant positional embedding, so every
token of the 60-row program is an independent argmax given the latent z. When z sits between two plausible
shapes, consecutive rows can come from different ones (a rectangle whose corners come from two widths decodes
as a trapezoid). This decoder writes the program one row at a time, each row conditioned on z and on the rows
already written, so later coordinates agree with earlier ones.

Row = (command, 16 args) exactly as in cadlib (args in [-1, 255], -1 = unused). The input embedding of a row
sums a command embedding and one embedding per arg slot; the output heads predict the next row's command and,
for the args that command uses (cadlib CMD_ARGS_MASK), their values. Training targets are the GT programs paired
with the uid-aligned CSR latents from scripts/export_gencad_embeddings.py; the encoder is never touched, so every
prior trained on those latents works unchanged.

Use it anywhere a CSR model is passed to mm.gencad_io.decode_latents: that function calls .generate() when the
model has one.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cadlib.macro import CMD_ARGS_MASK, EOS_IDX, MAX_TOTAL_LEN, N_ARGS, SOL_IDX

N_CMD = 6           # Line, Arc, Circle, EOS, SOL, Ext
N_VAL = 257         # arg value + 1: 0 = unused (-1), 1..256 = 0..255


class ARDecoder(nn.Module):
    def __init__(self, z_dim=256, d_model=384, n_layers=6, n_heads=8, d_ff=1536, n_mem=4, dropout=0.1,
                 max_len=MAX_TOTAL_LEN):
        super().__init__()
        self.config = dict(z_dim=z_dim, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff,
                           n_mem=n_mem, dropout=dropout, max_len=max_len)
        self.max_len, self.n_mem, self.d_model = max_len, n_mem, d_model
        self.cmd_emb = nn.Embedding(N_CMD + 1, d_model)               # + BOS
        self.arg_emb = nn.EmbeddingBag(N_ARGS * N_VAL, d_model, mode="sum")  # one table slice per arg slot
        self.pos_emb = nn.Parameter(torch.randn(max_len + 1, d_model) * 0.02)
        self.z_proj = nn.Linear(z_dim, n_mem * d_model)               # z -> n_mem memory tokens
        layer = nn.TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerDecoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.cmd_head = nn.Linear(d_model, N_CMD)
        self.arg_head = nn.Linear(d_model, N_ARGS * N_VAL)
        self.register_buffer("arg_mask", torch.tensor(CMD_ARGS_MASK, dtype=torch.bool))  # (N_CMD, N_ARGS)

    # ------------------------------------------------------------------ core
    def _embed_rows(self, rows):
        """rows (B, L, 17) int with args in [-1, 255] -> (B, L, d)."""
        B, L = rows.shape[:2]
        cmd = rows[..., 0].clamp(min=0)
        args = (rows[..., 1:] + 1).clamp(0, N_VAL - 1)                # (B, L, 16)
        offsets = torch.arange(N_ARGS, device=rows.device) * N_VAL
        bag = self.arg_emb((args + offsets).reshape(B * L, N_ARGS)).view(B, L, -1)
        return self.cmd_emb(cmd) + bag

    def _forward(self, z, prev_rows):
        """z (B, z_dim); prev_rows (B, L, 17) = rows already written (teacher-forced or generated).
        Returns logits for rows 0..L: cmd (B, L+1, N_CMD), args (B, L+1, N_ARGS, N_VAL)."""
        B = z.shape[0]
        bos = self.cmd_emb.weight[N_CMD].expand(B, 1, -1)
        x = torch.cat([bos, self._embed_rows(prev_rows)], 1) if prev_rows.shape[1] else bos
        x = x + self.pos_emb[: x.shape[1]]
        mem = self.z_proj(z).view(B, self.n_mem, self.d_model)
        L = x.shape[1]
        causal = torch.triu(torch.full((L, L), float("-inf"), device=z.device), 1)
        h = self.norm(self.transformer(x, mem, tgt_mask=causal))
        return self.cmd_head(h), self.arg_head(h).view(B, L, N_ARGS, N_VAL)

    def loss(self, z, vec):
        """z (B, z_dim); vec (B, max_len, 17) GT programs padded with EOS rows. Loss covers every row up to and
        including the first EOS; args only where the GT command uses them (as in DeepCAD)."""
        cmd_logits, arg_logits = self._forward(z, vec[:, :-1])
        cmd = vec[..., 0].long()
        is_eos = cmd == EOS_IDX
        first_eos = torch.where(is_eos.any(1), is_eos.float().argmax(1), torch.full_like(cmd[:, 0], vec.shape[1] - 1))
        valid = torch.arange(vec.shape[1], device=z.device)[None] <= first_eos[:, None]   # (B, L)
        c_loss = F.cross_entropy(cmd_logits[valid], cmd[valid])
        amask = self.arg_mask[cmd] & valid[..., None]                                      # (B, L, 16)
        target = (vec[..., 1:].long() + 1).clamp(0, N_VAL - 1)
        a_loss = F.cross_entropy(arg_logits[amask], target[amask]) if amask.any() else cmd_logits.sum() * 0
        return c_loss + a_loss, {"cmd_loss": c_loss.detach(), "arg_loss": a_loss.detach()}

    @torch.no_grad()
    def generate(self, z):
        """Greedy row-by-row decoding. Returns a list of (L_i, 17) int arrays, cut before the first EOS (None if
        no EOS within max_len), matching mm.gencad_io.decode_latents."""
        B, device = z.shape[0], z.device
        rows = torch.zeros(B, 0, 1 + N_ARGS, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(self.max_len):
            cmd_logits, arg_logits = self._forward(z, rows)
            cmd = cmd_logits[:, -1].argmax(-1)                                         # (B,)
            args = arg_logits[:, -1].argmax(-1) - 1                                    # (B, 16) in [-1, 255]
            args = torch.where(self.arg_mask[cmd], args, torch.full_like(args, -1))
            cmd = torch.where(done, torch.full_like(cmd, EOS_IDX), cmd)
            rows = torch.cat([rows, torch.cat([cmd[:, None], args], 1)[:, None]], 1)
            done |= cmd == EOS_IDX
            if done.all():
                break
        out = []
        for r in rows.cpu().numpy():
            cmds = r[:, 0].tolist()
            out.append(r[: cmds.index(EOS_IDX)].astype(np.int64) if EOS_IDX in cmds else None)
        return out

    # ------------------------------------------------------------------ io
    def save(self, path, **extra):
        torch.save({"config": self.config, "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        ck = torch.load(path, map_location=map_location, weights_only=False)
        m = cls(**ck["config"])
        m.load_state_dict(ck.get("ema_state_dict") or ck["state_dict"])
        return m, ck


def pad_program(vec, max_len=MAX_TOTAL_LEN):
    """(L, 17) program -> (max_len, 17), padded with EOS rows (and truncated if longer)."""
    from cadlib.macro import EOS_VEC
    vec = np.asarray(vec)[:max_len]
    return np.concatenate([vec, EOS_VEC[None].repeat(max_len - len(vec), 0)], 0)


__all__ = ["ARDecoder", "pad_program", "SOL_IDX"]
