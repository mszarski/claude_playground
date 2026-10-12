"""Fuse a variable subset of modality embeddings into one conditioning vector.

Each available modality embedding becomes a token (linear projection + modality-type embedding). A missing or
dropped modality is replaced by that modality's learned null token, so "text only", "image only", "both" and
"unconditional" are all the same forward pass with a different presence mask. That is what makes per-modality
classifier-free guidance possible at sampling time.
"""
import torch
from torch import nn


class ConditionFuser(nn.Module):
    def __init__(self, modality_dims: dict, d_model=512, n_layers=2, n_heads=8, dropout=0.1,
                 p_drop: dict | None = None, p_drop_all=0.1):
        """
        modality_dims: {name: embedding_dim}, e.g. {"image": 768, "text": 768}
        p_drop: per-modality probability of dropping that modality during training (default 0.15 each)
        p_drop_all: extra probability of dropping every modality (trains the unconditional branch)
        """
        super().__init__()
        self.names = list(modality_dims)
        self.d_model = d_model
        self.proj = nn.ModuleDict({
            m: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_model)) for m, d in modality_dims.items()
        })
        self.null = nn.ParameterDict({m: nn.Parameter(torch.randn(d_model) * 0.02) for m in self.names})
        self.type_emb = nn.Parameter(torch.randn(len(self.names) + 1, d_model) * 0.02)  # +1 for the CLS token
        self.cls = nn.Parameter(torch.randn(d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d_model)
        self.p_drop = {m: 0.15 for m in self.names} if p_drop is None else dict(p_drop)
        self.p_drop_all = p_drop_all

    def sample_keep_mask(self, present: dict, batch: int, device):
        """Training-time modality dropout. Returns {name: bool tensor (B,)} of modalities to keep."""
        drop_all = torch.rand(batch, device=device) < self.p_drop_all
        keep = {}
        for m in self.names:
            p = present.get(m, torch.zeros(batch, dtype=torch.bool, device=device))
            dropped = torch.rand(batch, device=device) < self.p_drop.get(m, 0.0)
            keep[m] = p & ~dropped & ~drop_all
        return keep

    def forward(self, embeds: dict, keep: dict | None = None, batch: int | None = None):
        """
        embeds: {name: (B, d_name)} for the modalities that exist in this batch (others may be absent)
        keep:   {name: (B,) bool}; False -> use the null token. Defaults to "keep all provided".
        returns (B, d_model)
        """
        if batch is None:
            batch = next(iter(embeds.values())).shape[0]
        device = self.cls.device
        tokens = []
        for i, m in enumerate(self.names):
            null = self.null[m].expand(batch, -1)
            if m in embeds:
                tok = self.proj[m](embeds[m].to(device=device, dtype=self.cls.dtype))
                k = keep[m] if keep is not None and m in keep else torch.ones(batch, dtype=torch.bool, device=device)
                tok = torch.where(k[:, None], tok, null)
            else:
                tok = null
            tokens.append(tok + self.type_emb[i])
        cls = (self.cls + self.type_emb[-1]).expand(batch, -1)
        x = torch.stack([cls] + tokens, dim=1)  # (B, 1 + M, d)
        x = self.encoder(x)
        return self.out_norm(x[:, 0])
