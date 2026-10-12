"""Frozen pretrained condition encoders ("the other model" GenCAD gets hooked up to).

Embeddings are computed once and cached to HDF5 (scripts/embed_conditions.py), so the prior trains on
fixed vectors and the encoders can be swapped without touching the prior code.

Defaults and why:
- image: SigLIP2 image tower. Strong general-purpose visual features; works on photos and sketches, not only
  on GenCAD's own renders.
- text:  Qwen3-Embedding-0.6B. Text2CAD's expert-level prompts are long and full of dimensions; CLIP/SigLIP
  text towers truncate at 64-77 tokens and would throw most of that away. A decoder-LLM embedding model reads
  the whole prompt.
Short-prompt alternative: --text_backend siglip puts text and image in one shared space.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image


def _as_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    for k in ("pooler_output", "image_embeds", "text_embeds"):
        v = getattr(out, k, None)
        if v is not None:
            return v
    return out[0]


def load_rgb(path_or_img, background=(255, 255, 255)):
    """Open an image and composite transparency onto a white background (GenCAD renders are RGBA)."""
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, background + (255,))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


class ImageEncoder(torch.nn.Module):
    def __init__(self, backend="siglip", model_id="google/siglip2-base-patch16-224", device="cpu", dtype=None,
                 model=None, processor=None):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        self.backend = backend
        self.model = model if model is not None else AutoModel.from_pretrained(model_id, torch_dtype=dtype)
        self.processor = processor if processor is not None else AutoImageProcessor.from_pretrained(model_id)
        self.model.eval().requires_grad_(False).to(device)
        self.device = device

    @property
    def dim(self):
        cfg = self.model.config
        if self.backend in ("siglip", "clip"):
            vc = getattr(cfg, "vision_config", cfg)
            return getattr(cfg, "projection_dim", None) or vc.hidden_size
        return cfg.hidden_size * (2 if self.backend == "dinov2" else 1)

    @torch.no_grad()
    def forward(self, images):
        images = [load_rgb(im) for im in images]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        if self.backend in ("siglip", "clip"):
            emb = _as_tensor(self.model.get_image_features(pixel_values=inputs["pixel_values"].to(self.model.dtype)))
        elif self.backend == "dinov2":
            h = self.model(pixel_values=inputs["pixel_values"].to(self.model.dtype)).last_hidden_state
            emb = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=-1)
        else:
            raise ValueError(self.backend)
        return F.normalize(emb.float(), dim=-1)


class TextEncoder(torch.nn.Module):
    def __init__(self, backend="hf-embed", model_id="Qwen/Qwen3-Embedding-0.6B", device="cpu", dtype=None,
                 max_length=512, pooling="last", instruction=None, model=None, tokenizer=None):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.backend, self.max_length, self.pooling = backend, max_length, pooling
        self.instruction = instruction
        if backend == "hf-embed" and instruction is None:
            self.instruction = ("Instruct: Given a description of a mechanical part, "
                                "retrieve its CAD construction sequence\nQuery: ")
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(
            model_id, padding_side="left" if pooling == "last" else "right")
        self.model = model if model is not None else AutoModel.from_pretrained(model_id, torch_dtype=dtype)
        self.model.eval().requires_grad_(False).to(device)
        self.device = device
        if backend in ("siglip", "clip"):
            self.max_length = min(max_length, 64 if backend == "siglip" else 77)

    @property
    def dim(self):
        cfg = self.model.config
        if self.backend in ("siglip", "clip"):
            return getattr(cfg, "projection_dim", None) or getattr(cfg, "text_config", cfg).hidden_size
        return cfg.hidden_size

    @torch.no_grad()
    def forward(self, texts):
        texts = [t if isinstance(t, str) else "" for t in texts]
        if self.backend in ("siglip", "clip"):
            pad = "max_length" if self.backend == "siglip" else True
            inputs = self.tokenizer(texts, padding=pad, truncation=True, max_length=self.max_length,
                                    return_tensors="pt").to(self.device)
            emb = _as_tensor(self.model.get_text_features(**inputs))
        elif self.backend == "hf-embed":
            texts = [(self.instruction or "") + t for t in texts]
            inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                                    return_tensors="pt").to(self.device)
            h = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"]
            if self.pooling == "last":
                if self.tokenizer.padding_side == "left":
                    emb = h[:, -1]
                else:
                    idx = mask.sum(1) - 1
                    emb = h[torch.arange(h.shape[0], device=h.device), idx]
            else:
                emb = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        else:
            raise ValueError(self.backend)
        return F.normalize(emb.float(), dim=-1)
