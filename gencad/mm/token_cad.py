"""Image (+ optional text) -> CAD program, decoded directly from image patch tokens and prompt tokens.

Generalises mm/text_cad.py: the autoregressive program decoder (mm.ar_decoder.ARDecoder) cross-attends to
  - image tokens: SigLIP2's per-patch hidden states of a render (196 for patch16-224), not its pooled vector;
  - text tokens:  per-token hidden states of a frozen Qwen3 model cut to its first layers (as in TextCAD).
Each source has its own projection plus a learned type embedding. A sample may have an image, a prompt, or both;
a missing source is masked out of the memory. Both encoders are frozen (end-to-end tuning of the pooled image
encoder did not help, see CLAUDE.md), so only the projections and the decoder train.
"""
from __future__ import annotations

import torch
from torch import nn

from .ar_decoder import ARDecoder


class TokenCAD(nn.Module):
    def __init__(self, image_model_id="google/siglip2-base-patch16-224", text_model_id="Qwen/Qwen3-Embedding-0.6B",
                 text_layers=8, max_text_len=512, d_model=384, n_layers=6, n_heads=8, d_ff=1536, dropout=0.1,
                 image_model=None, text_model=None, tokenizer=None):
        super().__init__()
        self.config = dict(image_model_id=image_model_id, text_model_id=text_model_id, text_layers=text_layers,
                           max_text_len=max_text_len, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                           d_ff=d_ff, dropout=dropout)
        if image_model is None:
            from transformers import AutoModel
            image_model = AutoModel.from_pretrained(image_model_id).vision_model
        if text_model is None:
            from transformers import AutoModel
            text_model = AutoModel.from_pretrained(text_model_id, torch_dtype=torch.bfloat16)
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(text_model_id)
        if text_layers and hasattr(text_model, "layers"):
            text_model.layers = text_model.layers[:text_layers]
            text_model.config.num_hidden_layers = len(text_model.layers)
            if getattr(text_model.config, "layer_types", None):
                text_model.config.layer_types = text_model.config.layer_types[:text_layers]
        self.vision = image_model.eval().requires_grad_(False)
        self.text = text_model.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        vdim, tdim = image_model.config.hidden_size, text_model.config.hidden_size
        self.img_proj = nn.Sequential(nn.LayerNorm(vdim), nn.Linear(vdim, d_model))
        self.txt_proj = nn.Sequential(nn.LayerNorm(tdim), nn.Linear(tdim, d_model))
        self.type_emb = nn.Parameter(torch.randn(2, d_model) * 0.02)          # 0 = image token, 1 = text token
        self.core = ARDecoder(d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, dropout=dropout)

    FROZEN = ("vision.", "text.")

    def train(self, mode=True):
        super().train(mode)
        self.vision.eval()
        self.text.eval()
        return self

    @property
    def device(self):
        return self.type_emb.device

    def encode(self, pixels=None, prompts=None):
        """pixels (B, 3, 224, 224) normalised like SigLIP2, or None; prompts: list of str/None (None = no prompt
        for that sample), or None. Returns memory (B, M, d) and padding mask (B, M), True = pad."""
        mems, pads = [], []
        if pixels is not None:
            with torch.no_grad():
                h = self.vision(pixel_values=pixels.to(self.device)).last_hidden_state      # (B, P, vdim)
            mems.append(self.img_proj(h.float()) + self.type_emb[0])
            pads.append(torch.zeros(h.shape[:2], dtype=torch.bool, device=self.device))
        if prompts is not None and any(p for p in prompts):
            texts = [p or "" for p in prompts]
            tok = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_text_len,
                                 return_tensors="pt").to(self.device)
            with torch.no_grad():
                h = self.text(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]).last_hidden_state
            pad = ~tok["attention_mask"].bool()
            pad[torch.tensor([not p for p in prompts], device=self.device)] = True          # sample has no prompt
            mems.append(self.txt_proj(h.float()) + self.type_emb[1])
            pads.append(pad)
        assert mems, "need an image and/or a prompt"
        mem, pad = torch.cat(mems, 1), torch.cat(pads, 1)
        assert not pad.all(1).any(), "every sample needs at least one image or prompt token"
        return mem, pad

    def loss(self, vec, pixels=None, prompts=None):
        mem, pad = self.encode(pixels, prompts)
        return self.core.program_loss(mem, vec, pad)

    @torch.no_grad()
    def generate(self, pixels=None, prompts=None):
        mem, pad = self.encode(pixels, prompts)
        return self.core.generate_from_memory(mem, pad)

    def trainable_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith(self.FROZEN)}

    def save(self, path, **extra):
        torch.save({"config": self.config, "state_dict": self.trainable_state_dict(), **extra}, path)

    @classmethod
    def load(cls, path, map_location="cpu", **kw):
        ck = torch.load(path, map_location=map_location, weights_only=False)
        m = cls(**ck["config"], **kw)
        missing, unexpected = m.load_state_dict(ck.get("ema_state_dict") or ck["state_dict"], strict=False)
        assert not unexpected and all(k.startswith(cls.FROZEN) for k in missing), (missing, unexpected)
        return m, ck
