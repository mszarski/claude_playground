"""Embed raw conditions (render images, text prompts) with frozen pretrained encoders and cache to HDF5.

  python scripts/embed_conditions.py image --data data --out data/emb/cond_image.h5 \
      --backend siglip --model google/siglip2-base-patch16-224
  python scripts/embed_conditions.py text --text_index data/text_index.json --out data/emb/cond_text.h5 \
      --backend hf-embed --model Qwen/Qwen3-Embedding-0.6B

uids come from data/emb/cad_latents.h5 if it exists (so rows line up with the prior's targets),
otherwise from filtered_data.json.
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm.cond_encoders import ImageEncoder, TextEncoder  # noqa: E402
from mm.data import _read_uids, write_store  # noqa: E402


def get_uids(a):
    if a.cad_latents and os.path.exists(a.cad_latents):
        with h5py.File(a.cad_latents, "r") as f:
            return _read_uids(f["uids"])
    with open(os.path.join(a.data, "filtered_data.json")) as f:
        s = json.load(f)
    return [u for k in ("train", "validation", "test") for u in s[k]]


def run_batched(items, fn, bs, desc):
    """items: list of (i, k, payload). Returns list of (i, k, emb)."""
    out = []
    for s in tqdm(range(0, len(items), bs), desc=desc):
        chunk = items[s: s + bs]
        embs = fn([c[2] for c in chunk]).cpu().numpy()
        out += [(c[0], c[1], e) for c, e in zip(chunk, embs)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modality", choices=["image", "text"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--cad_latents", default="data/emb/cad_latents.h5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--img_dir", default="images")
    ap.add_argument("--n_views", type=int, default=5)
    ap.add_argument("--text_index", default="data/text_index.json")
    ap.add_argument("--levels", nargs="*", default=None, help="text levels to keep, in order")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    uids = get_uids(a)[: a.limit]
    print(f"[embed] {a.modality}: {len(uids)} uids on {device}")

    if a.modality == "image":
        backend = a.backend or "siglip"
        model_id = a.model or "google/siglip2-base-patch16-224"
        enc = ImageEncoder(backend, model_id, device=device, dtype=dtype)
        with open(os.path.join(a.data, "image_ids.json")) as f:
            img_ids = json.load(f)
        items = []
        for i, u in enumerate(uids):
            for k, view in enumerate(img_ids.get(u, [])[: a.n_views]):
                p = os.path.join(a.data, a.img_dir, f"{u}_{view}.png")
                if os.path.exists(p):
                    items.append((i, k, p))
        K, variants = a.n_views, list(range(a.n_views))
    else:
        backend = a.backend or "hf-embed"
        model_id = a.model or "Qwen/Qwen3-Embedding-0.6B"
        enc = TextEncoder(backend, model_id, device=device, dtype=dtype, max_length=a.max_length)
        with open(a.text_index) as f:
            index = json.load(f)
        levels = a.levels or sorted({lv for e in index.values() for lv in e})
        items = [(i, k, index[u][lv]) for i, u in enumerate(uids) if u in index
                 for k, lv in enumerate(levels) if lv in index[u]]
        K, variants = len(levels), levels

    print(f"[embed] {len(items)} {a.modality} items, backend={backend} model={model_id} dim={enc.dim}")
    emb = np.zeros((len(uids), K, enc.dim), np.float16)
    valid = np.zeros((len(uids), K), bool)
    for i, k, e in run_batched(items, enc, a.bs, a.modality):
        emb[i, k], valid[i, k] = e, True
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    write_store(a.out, uids, emb, valid, backend=backend, model=model_id, variants=variants)
    print(f"[embed] wrote {a.out} {emb.shape}; uids with >=1 item: {valid.any(1).mean():.1%}")


if __name__ == "__main__":
    main()
