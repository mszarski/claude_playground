"""Side-by-side visual check on random test CAD models: ground truth vs generations from each condition.

For every picked uid it decodes and renders (iso view, shaded mesh):
  gt            the ground-truth CAD sequence
  gencad        released GenCAD diffusion prior on its CCIP sketch latent       (needs --gencad_dp_ckpt)
  <name>        the multimodal prior under each condition in --conds, using the precomputed embeddings in
                --emb_dir (image = render view 0; text:<level> = that Text2CAD prompt level)
plus free-form --prompts through the prior's text encoder. Writes <out>/grid_<k>.png, <out>/prompts.png,
<out>/cells/*.{png,stl}, and <out>/index.json (uids, prompts, per-cell status). Needs pythonocc-core.

  python scripts/visual_check.py --prior results/mm_v1/last.pt --csr_ckpt data/ckpt/ae_ckpt_epoch1000.pth \
      --gencad_dp_ckpt data/ckpt/sketch_cond_diffusion_ckpt_epoch1000000.pt --n_uids 12 --out results/visual
"""
import argparse
import json
import os
import random
import shutil
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm.data import CadLatentStore, EmbeddingStore  # noqa: E402
from mm.diffusion import MultiCondDiffusionPrior  # noqa: E402
from mm.gencad_io import decode_latents, export_shape, load_csr, vec_to_shape  # noqa: E402

DEFAULT_CONDS = ["image", "text:beginner", "text:expert", "image+text:expert"]
DEFAULT_PROMPTS = [
    "A rectangular plate with four holes near the corners.",
    "A cylinder with a smaller concentric hole through the middle.",
    "An L-shaped bracket made of two perpendicular rectangular plates.",
    "A hexagonal nut.",
    "A flat washer.",
    "A stepped shaft made of three cylinders of decreasing diameter.",
]


def render_mesh(stl_path, png_path, size=3.0):
    """Shaded iso render of an STL with matplotlib (no display / GL needed)."""
    import trimesh
    m = trimesh.load(stl_path, force="mesh")
    v, f = m.vertices, m.faces
    v = (v - v.mean(0)) / max(np.ptp(v, 0).max(), 1e-9)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([0.4, -0.5, 0.8]) / np.linalg.norm([0.4, -0.5, 0.8])
    shade = 0.35 + 0.65 * np.abs(n @ light)
    colors = np.clip(np.outer(shade, [0.36, 0.55, 0.85]), 0, 1)
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(tri, facecolors=colors, edgecolor="none"))
    ax.set_xlim(-0.55, 0.55), ax.set_ylim(-0.55, 0.55), ax.set_zlim(-0.55, 0.55)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=-60)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(png_path, dpi=80)
    plt.close(fig)


def vec_cell(vec, stem):
    """CAD vector -> STL + PNG. Returns a status string: ok / no_eos / not_solid / render_error."""
    if vec is None:
        return "no_eos"
    shape = vec_to_shape(vec)
    if shape is None:
        return "not_solid"
    try:
        export_shape(shape, stem + ".stl")
        render_mesh(stem + ".stl", stem + ".png")
        return "ok"
    except Exception as e:  # noqa: BLE001
        print(f"[render] {stem}: {e}")
        return "render_error"


def placeholder(text, path, size=3.0):
    fig = plt.figure(figsize=(size, size))
    fig.text(0.5, 0.5, text, ha="center", va="center", fontsize=14, color="#a33")
    fig.savefig(path, dpi=80)
    plt.close(fig)


def grid(rows, cols, cell_png, out_png, row_labels=None):
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.4 * len(cols), 2.4 * len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        for c, col in enumerate(cols):
            ax = axes[r][c]
            ax.imshow(plt.imread(cell_png(row, col)))
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(col, fontsize=10)
            if c == 0 and row_labels:
                ax.set_ylabel(row_labels[r], fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True)
    ap.add_argument("--csr_ckpt", required=True)
    ap.add_argument("--gencad_dp_ckpt", default=None)
    ap.add_argument("--emb_dir", default="data/emb")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_uids", type=int, default=12)
    ap.add_argument("--rows_per_grid", type=int, default=6)
    ap.add_argument("--conds", nargs="*", default=DEFAULT_CONDS)
    ap.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    ap.add_argument("--n_prompt_samples", type=int, default=3)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/visual")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cells = os.path.join(a.out, "cells")
    os.makedirs(cells, exist_ok=True)
    prior, ck = MultiCondDiffusionPrior.load(a.prior)
    prior = prior.to(device).eval()
    csr, _ = load_csr(a.csr_ckpt, device)

    # pick uids that have every condition we want to show
    cad = CadLatentStore(os.path.join(a.emb_dir, "cad_latents.h5"))
    stores = {m: EmbeddingStore(os.path.join(a.emb_dir, f"cond_{m}.h5"))
              for m in ["image", "text", "ccip_image"] if os.path.exists(os.path.join(a.emb_dir, f"cond_{m}.h5"))}
    levels = json.loads(h5py.File(os.path.join(a.emb_dir, "cond_text.h5"), "r").attrs["variants"]) \
        if "text" in stores else []
    with open(os.path.join(a.data, "filtered_data.json")) as f:
        split = json.load(f)[a.split]
    with open(os.path.join(a.data, "image_ids.json")) as f:
        img_ids = json.load(f)
    with open(os.path.join(a.data, "text_index.json")) as f:
        text_index = json.load(f)

    def has_all(u):
        if u not in cad.index or not img_ids.get(u) or u not in text_index:
            return False
        return all(u in s.index and s.valid[s.index[u]].any() for s in stores.values())

    pool = [u for u in split if has_all(u)]
    random.Random(a.seed).shuffle(pool)
    uids = pool[: a.n_uids]
    print(f"[visual] {len(uids)} uids from {len(pool)} eligible {a.split} uids")

    def cond_embeds(cond):
        mods = cond.split(":")[0].split("+")
        lv = cond.split(":")[1] if ":" in cond else None
        out = {}
        for m in mods:
            variant = levels.index(lv) if m == "text" else 0
            out[m] = torch.stack([stores[m].get(u, variant)[0] for u in uids]).to(device)
        return out

    def sample(embeds, n=None):
        g = torch.Generator(device=device).manual_seed(a.seed)
        z = prior.sample(embeds, guidance={m: a.guidance for m in embeds}, steps=a.steps, generator=g)
        with torch.no_grad():
            return decode_latents(csr, z)

    status = {u: {} for u in uids}
    cols = ["input render", "gt"]
    # input render + ground truth
    for k, u in enumerate(uids):
        shutil.copy(os.path.join(a.data, "images", f"{u}_{img_ids[u][0]}.png"), os.path.join(cells, f"{k}_input.png"))
        with h5py.File(os.path.join(a.data, "cad_vec", u + ".h5"), "r") as f:
            status[u]["gt"] = vec_cell(f["vec"][:], os.path.join(cells, f"{k}_gt"))
    # released GenCAD prior on its own CCIP latents
    if a.gencad_dp_ckpt and "ccip_image" in stores:
        from model import GaussianDiffusion1D, ResNetDiffusion
        net = ResNetDiffusion(d_in=256, n_blocks=10, d_main=2048, d_hidden=2048, dropout_first=0.1,
                              dropout_second=0.1, d_out=256)
        dp = GaussianDiffusion1D(net, z_dim=256, timesteps=500, objective="pred_x0", auto_normalize=False)
        dp.load_state_dict(torch.load(a.gencad_dp_ckpt, map_location="cpu", weights_only=False)["model"])
        dp = dp.to(device).eval()
        torch.manual_seed(a.seed)
        zs = dp.sample(cond=torch.stack([stores["ccip_image"].get(u, 0)[0] for u in uids]).to(device))
        with torch.no_grad():
            vecs = decode_latents(csr, zs)
        for k, (u, v) in enumerate(zip(uids, vecs)):
            status[u]["gencad"] = vec_cell(v, os.path.join(cells, f"{k}_gencad"))
        cols.append("gencad")
    # the multimodal prior under each condition
    for cond in a.conds:
        if any(m not in prior.modalities or m not in stores for m in cond.split(":")[0].split("+")):
            print(f"[visual] skip {cond}: not in prior/embeddings")
            continue
        for k, (u, v) in enumerate(zip(uids, sample(cond_embeds(cond)))):
            status[u][cond] = vec_cell(v, os.path.join(cells, f"{k}_{cond.replace(':', '-').replace('+', '_')}"))
        cols.append(cond)

    def cell_png(k, col):
        name = {"input render": "input"}.get(col, col.replace(":", "-").replace("+", "_"))
        p = os.path.join(cells, f"{k}_{name}.png")
        if not os.path.exists(p):
            placeholder(status[uids[k]].get(col, "missing"), p)
        return p

    for g0 in range(0, len(uids), a.rows_per_grid):
        rows = list(range(g0, min(g0 + a.rows_per_grid, len(uids))))
        grid(rows, cols, cell_png, os.path.join(a.out, f"grid_{g0 // a.rows_per_grid}.png"),
             row_labels=[uids[k] for k in rows])

    # free-form prompts through the prior's own text encoder
    prompt_status = []
    if a.prompts and "text" in prior.modalities:
        from mm.cond_encoders import TextEncoder
        info = ck.get("encoders", {}).get("text", {})
        enc = TextEncoder(info.get("backend", "hf-embed"), info.get("model", "Qwen/Qwen3-Embedding-0.6B"),
                          device=device)
        emb = enc(a.prompts).float().repeat_interleave(a.n_prompt_samples, 0)
        g = torch.Generator(device=device).manual_seed(a.seed)
        z = prior.sample({"text": emb}, guidance={"text": a.guidance}, steps=a.steps, generator=g)
        with torch.no_grad():
            vecs = decode_latents(csr, z)
        for k, v in enumerate(vecs):
            i, j = divmod(k, a.n_prompt_samples)
            prompt_status.append({"prompt": i, "sample": j, "status": vec_cell(v, os.path.join(cells, f"p{i}_{j}"))})

        def prompt_png(i, j):
            p = os.path.join(cells, f"p{i}_{j}.png")
            if not os.path.exists(p):
                placeholder(prompt_status[i * a.n_prompt_samples + j]["status"], p)
            return p

        grid(list(range(len(a.prompts))), list(range(a.n_prompt_samples)), prompt_png,
             os.path.join(a.out, "prompts.png"), row_labels=[p[:40] for p in a.prompts])

    index = {"uids": uids, "columns": cols, "status": status, "guidance": a.guidance,
             "texts": {u: {lv: text_index[u].get(lv) for lv in ["abstract", "beginner", "expert"]} for u in uids},
             "prompts": a.prompts, "prompt_status": prompt_status}
    with open(os.path.join(a.out, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    ok = {c: sum(status[u].get(c) == "ok" for u in uids) for c in cols[1:]}
    print(f"[visual] valid solids per column (of {len(uids)}): {ok}")
    if prompt_status:
        print(f"[visual] prompts: {sum(p['status'] == 'ok' for p in prompt_status)}/{len(prompt_status)} valid solids")


if __name__ == "__main__":
    main()
