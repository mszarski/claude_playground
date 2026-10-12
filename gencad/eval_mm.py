"""Evaluate priors on the test split: latent metrics + decoded-sequence accuracy.

Rows reported:
  csr_recon            decode(GT latent)            -- upper bound set by the CSR autoencoder
  gencad_dp            original GenCAD prior on CCIP image latents (if --gencad_dp_ckpt), i.e. the replication
  mm:<combo>           the multimodal prior under every conditioning combination

  python eval_mm.py --emb_dir data/emb --csr_ckpt data/ckpt/ae_ckpt_epoch1000.pth \
      --prior results/mm_v1/last.pt --gencad_dp_ckpt data/ckpt/sketch_cond_diffusion_ckpt_epoch1000000.pt \
      --n 2000 --out results/mm_v1/eval.json
"""
import argparse
import itertools
import json
import os

import h5py
import numpy as np
import torch

from mm.data import MultiModalLatentDataset, split_batch
from mm.diffusion import MultiCondDiffusionPrior
from mm.gencad_io import decode_latents, load_csr
from mm.metrics import latent_metrics, seq_metrics


def gt_vecs(data_root, uids):
    out = []
    for u in uids:
        with h5py.File(os.path.join(data_root, "cad_vec", u + ".h5"), "r") as f:
            out.append(f["vec"][:])
    return out


def decode_batched(csr, z, bs=256):
    out = []
    for i in range(0, len(z), bs):
        out += decode_latents(csr, z[i: i + bs])
    return out


def row(name, z_pred, z_gt, csr, gts, rows=None):
    rows = torch.arange(len(z_gt)) if rows is None else rows
    m = latent_metrics(z_pred, z_gt[rows])
    m.update(seq_metrics(decode_batched(csr, z_pred), [gts[i] for i in rows.tolist()]))
    print(f"{name:28s} " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()),
          flush=True)
    return m


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="data/emb")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--csr_ckpt", required=True)
    ap.add_argument("--prior", default=None)
    ap.add_argument("--gencad_dp_ckpt", default=None)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--guidance", type=float, nargs="*", default=[1.0, 2.0, 3.0])
    ap.add_argument("--text_level", default=None,
                    help="evaluate text with this prompt level (e.g. abstract, expert); default: a random level per uid")
    ap.add_argument("--out", default="eval.json")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split_json = os.path.join(a.data, "filtered_data.json")
    csr, _ = load_csr(a.csr_ckpt, device)
    report = {}

    prior, mods = None, ["ccip_image"]
    if a.prior:
        prior, _ = MultiCondDiffusionPrior.load(a.prior)
        prior = prior.to(device).eval()
        mods = sorted(set(prior.modalities) | ({"ccip_image"} if a.gencad_dp_ckpt else set()))
    paths = {m: os.path.join(a.emb_dir, f"cond_{m}.h5") for m in mods
             if os.path.exists(os.path.join(a.emb_dir, f"cond_{m}.h5"))}
    variants = {m: 0 for m in paths if m != "text"}
    if a.text_level and "text" in paths:
        with h5py.File(paths["text"], "r") as f:
            levels = json.loads(f.attrs["variants"])
        variants["text"] = levels.index(a.text_level)
    ds = MultiModalLatentDataset(os.path.join(a.emb_dir, "cad_latents.h5"), paths, a.split, split_json,
                                 variants=variants, max_items=a.n)
    items = [ds[i] for i in range(len(ds))]
    batch = torch.utils.data.default_collate([{k: v for k, v in b.items() if k != "uid"} for b in items])
    z, embeds, present = split_batch(batch, device)
    gts = gt_vecs(a.data, [b["uid"] for b in items])
    print(f"[eval] {len(items)} {a.split} samples, modalities available: {list(embeds)}")

    report["csr_recon"] = row("csr_recon", z, z, csr, gts)

    if a.gencad_dp_ckpt and "ccip_image" in embeds:
        from model import GaussianDiffusion1D, ResNetDiffusion
        net = ResNetDiffusion(d_in=256, n_blocks=10, d_main=2048, d_hidden=2048, dropout_first=0.1,
                              dropout_second=0.1, d_out=256)
        dp = GaussianDiffusion1D(net, z_dim=256, timesteps=500, objective="pred_x0", auto_normalize=False)
        dp.load_state_dict(torch.load(a.gencad_dp_ckpt, map_location="cpu", weights_only=False)["model"])
        dp = dp.to(device).eval()
        rows = torch.nonzero(present["ccip_image"].cpu()).flatten()
        torch.manual_seed(0)
        zs = torch.cat([dp.sample(cond=embeds["ccip_image"][rows[i: i + 512]]) for i in range(0, len(rows), 512)])
        report["gencad_dp"] = row("gencad_dp", zs, z, csr, gts, rows)

    if prior is not None:
        pm = [m for m in prior.modalities if m in embeds]
        for r in range(1, len(pm) + 1):
            for combo in itertools.combinations(pm, r):
                rows = torch.nonzero(torch.stack([present[m] for m in combo]).all(0).cpu()).flatten()
                if len(rows) < 8:
                    continue
                for w in a.guidance:
                    g = torch.Generator(device=device).manual_seed(0)
                    zs = prior.sample({m: embeds[m][rows] for m in combo}, guidance={m: w for m in combo},
                                      steps=a.steps, generator=g)
                    name = f"mm:{'+'.join(combo)}@w{w:g}"
                    report[name] = row(name, zs, z, csr, gts, rows)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[eval] wrote {a.out}")


if __name__ == "__main__":
    main()
