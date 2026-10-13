"""Text -> CAD program, decoded directly from the prompt's tokens (no pooled embedding, no latent prior).

Every prior in this repo squeezes a prompt into one pooled embedding and maps it to one 256-d CSR latent. Exact
dimensions in Text2CAD's prompts ("a line from (0.6, 0.0) to (0.6, 0.375)") do not survive that. Here the
autoregressive program decoder (mm.ar_decoder.ARDecoder) cross-attends to per-token hidden states of a frozen,
truncated Qwen3 text model, so it can copy numbers from the prompt into the program, Text2CAD-style.

Only the projection and the decoder are trained; the text model is loaded from the Hub (or passed in) and frozen.
"""
from __future__ import annotations

import torch
from torch import nn

from .ar_decoder import ARDecoder


class TextCAD(nn.Module):
    def __init__(self, text_model_id="Qwen/Qwen3-Embedding-0.6B", text_layers=8, max_text_len=512,
                 d_model=384, n_layers=6, n_heads=8, d_ff=1536, dropout=0.1, text_model=None, tokenizer=None):
        super().__init__()
        self.config = dict(text_model_id=text_model_id, text_layers=text_layers, max_text_len=max_text_len,
                           d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        if text_model is None:
            from transformers import AutoModel
            text_model = AutoModel.from_pretrained(text_model_id, torch_dtype=torch.bfloat16)
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(text_model_id)
        if text_layers and hasattr(text_model, "layers"):
            # the decoder needs token-level detail (digits, units) more than deep semantics: keep the first layers
            text_model.layers = text_model.layers[:text_layers]
            text_model.config.num_hidden_layers = len(text_model.layers)
            if hasattr(text_model.config, "layer_types") and text_model.config.layer_types:
                text_model.config.layer_types = text_model.config.layer_types[:text_layers]
        self.text = text_model.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        dim = text_model.config.hidden_size
        self.mem_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, d_model))
        self.core = ARDecoder(d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, dropout=dropout)

    def train(self, mode=True):
        super().train(mode)
        self.text.eval()                    # the frozen text model never goes into train mode
        return self

    def encode(self, prompts):
        """prompts: list of str -> memory (B, T, d), padding mask (B, T) True = pad."""
        device = self.mem_proj[1].weight.device
        tok = self.tokenizer(list(prompts), padding=True, truncation=True, max_length=self.max_text_len,
                             return_tensors="pt").to(device)
        with torch.no_grad():
            h = self.text(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]).last_hidden_state
        return self.mem_proj(h.float()), ~tok["attention_mask"].bool()

    def loss(self, prompts, vec):
        mem, pad = self.encode(prompts)
        return self.core.program_loss(mem, vec, pad)

    @torch.no_grad()
    def generate(self, prompts):
        mem, pad = self.encode(prompts)
        return self.core.generate_from_memory(mem, pad)

    def trainable_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("text.")}

    def save(self, path, **extra):
        torch.save({"config": self.config, "state_dict": self.trainable_state_dict(), **extra}, path)

    @classmethod
    def load(cls, path, map_location="cpu", **kw):
        ck = torch.load(path, map_location=map_location, weights_only=False)
        m = cls(**ck["config"], **kw)
        sd = ck.get("ema_state_dict") or ck["state_dict"]
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith("text.") for k in missing), (missing, unexpected)
        return m, ck
