"""Two sanity checks for eval_mm.py's numbers, on the same test uids.

1. Train/test duplication: how many test programs also occur in the training set, exactly or within the
   prog_match tolerance (same command sequence, every used parameter within +-tol quantisation steps).
2. Nearest-neighbour baselines: answer each test condition with the GT program of the closest *training* model in
   embedding space (no generation at all). Conditions: image (view 0 vs every training view), text (expert
   level), ccip_image (GenCAD's CCIP latent), and an oracle that retrieves by the GT CSR latent itself.

With --prior / --gencad_dp_ckpt it also samples those priors and reports every method separately on test uids
that have a training duplicate and on those that do not, which shows how much of each score is memorisation.

  python scripts/baselines.py --emb_dir data/emb --data data --csr_ckpt data/ckpt/ae_ckpt_epoch1000.pth \
      --prior results/mm_v1/last.pt --gencad_dp_ckpt data/ckpt/sketch_cond_diffusion_ckpt_epoch1000000.pt \
      --out results/baselines/report.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadlib.macro import CMD_ARGS_MASK, EOS_IDX  # noqa: E402
from mm.data import MultiModalLatentDataset, PooledViewStore, cond_file  # noqa: E402
from mm.metrics import latent_metrics, seq_metrics  # noqa: E402


def body(v):
    """Program rows before the first EOS."""
    c = v[:, 0].tolist()
    return v[: c.index(EOS_IDX)] if EOS_IDX in c else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="data/emb")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--tol", type=int, default=3)
    ap.add_argument("--csr_ckpt", default=None)
    ap.add_argument("--prior", default=None, help="multimodal prior to sample (image, text, image+text at --guidance)")
    ap.add_argument("--gencad_dp_ckpt", default=None)
    ap.add_argument("--text_cad", default=None, help="mm.text_cad.TextCAD checkpoint: decode programs from prompts")
    ap.add_argument("--text_cad_levels", nargs="*", default=["expert", "intermediate", "beginner", "abstract"])
    ap.add_argument("--token_cad", default=None, help="mm.token_cad.TokenCAD checkpoint: decode from image/text tokens")
    ap.add_argument("--ar_decoder", default=None, help="also report the autoregressive decoder's GT-latent ceiling")
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--n_views", type=int, nargs="*", default=[1, 5], help="view counts for a multi-view prior")
    ap.add_argument("--override", action="append", default=[], help="modality=path to an alternative cond_*.h5")
    ap.add_argument("--skip_knn", action="store_true", help="only score the priors (duplicates are always computed)")
    ap.add_argument("--out", default="results/baselines/report.json")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split_json = os.path.join(a.data, "filtered_data.json")
    with open(split_json) as f:
        splits = json.load(f)

    # --- test uids: exactly eval_mm.py's selection (first n of the split that have any condition)
    paths = {m: os.path.join(a.emb_dir, f"cond_{m}.h5") for m in ["ccip_image", "image", "text"]}
    for spec in a.override:   # e.g. image=results/e2e_v1/emb/cond_image.h5 (fine-tuned encoder's embeddings)
        m, p = spec.split("=", 1)
        paths[m] = p
    paths = {m: p for m, p in paths.items() if os.path.exists(p)}
    ds = MultiModalLatentDataset(os.path.join(a.emb_dir, "cad_latents.h5"), paths, a.split, split_json,
                                 variants={m: 0 for m in paths if m != "text"}, max_items=a.n)
    test = ds.uids
    cad = ds.cad
    with h5py.File(paths["text"], "r") as f:
        expert = json.loads(f.attrs["variants"]).index("expert")

    # --- programs: train from the cache written by train_ar_decoder.py, test from cad_vec
    cache = np.load(os.path.join(a.emb_dir, "programs.npz"))
    cidx = {u: i for i, u in enumerate(cache["uids"].astype(str))}
    train = [u for u in splits["train"] if u in cidx and u in cad.index]
    tr_prog = cache["vec"][[cidx[u] for u in train]].astype(np.int64)
    gts = []
    for u in test:
        with h5py.File(os.path.join(a.data, "cad_vec", u + ".h5"), "r") as f:
            gts.append(f["vec"][:].astype(np.int64))
    print(f"[data] {len(test)} {a.split} uids, {len(train)} train programs")

    # --- 1. duplicates
    by_cmds = defaultdict(list)
    tr_bodies = [body(v) for v in tr_prog]
    for i, b in enumerate(tr_bodies):
        by_cmds[b[:, 0].tobytes()].append(i)
    exact, near = np.zeros(len(test), bool), np.zeros(len(test), bool)
    for k, g in enumerate(tqdm(gts, desc="dups")):
        gb = body(g)
        cand = by_cmds.get(gb[:, 0].tobytes(), [])
        if not cand:
            continue
        P = np.stack([tr_bodies[i] for i in cand])                          # (C, L, 17)
        m = CMD_ARGS_MASK[gb[:, 0]].astype(bool)                            # (L, 16)
        d = np.abs(P[:, :, 1:] - gb[None, :, 1:])
        exact[k] = (d[:, m] == 0).all(1).any() if m.any() else True
        near[k] = (d[:, m] < a.tol).all(1).any() if m.any() else True
    report = {"n": len(test), "dup_exact": float(exact.mean()), "dup_near": float(near.mean())}
    print(f"[dups] test programs also in train: exact {exact.mean():.3f}, within +-{a.tol} {near.mean():.3f}")

    def score(name, vecs, z_pred=None):
        groups = {"all": np.ones(len(test), bool), "dup": near, "novel": ~near}
        row = {}
        for gname, mask in groups.items():
            idx = np.nonzero(mask)[0]
            m = seq_metrics([vecs[i] for i in idx], [gts[i] for i in idx])
            row[gname] = {k: float(v) for k, v in m.items()} | {"n": int(len(idx))}
        if z_pred is not None:
            row["all"].update({k: float(v) for k, v in latent_metrics(z_pred.cpu(), z_gt.cpu()).items()
                               if k in ("cos", "R@1", "R@10")})
        report[name] = row
        print(f"{name:26s} " + "  ".join(
            f"{g}: prog_match={row[g]['prog_match']:.3f} acc_cmd={row[g]['acc_cmd']:.3f} "
            f"acc_param={row[g]['acc_param']:.3f} (n={row[g]['n']})" for g in groups)
            + (f"  R@1={row['all']['R@1']:.3f}" if "R@1" in row["all"] else ""), flush=True)

    # --- 2. nearest-neighbour retrieval over the training set
    z_gt = torch.stack([cad.z[cad.index[u]] for u in test]).to(device)
    z_tr = torch.stack([cad.z[cad.index[u]] for u in train]).to(device)

    def knn(name, q, keys, key_owner):
        """q (Q, d) queries; keys (K, d) training embeddings; key_owner (K,) index into train."""
        q, keys = F.normalize(q.float(), dim=-1), F.normalize(keys.float(), dim=-1)
        best = torch.cat([key_owner[(q[s: s + 256] @ keys.t()).argmax(1)] for s in range(0, len(q), 256)])
        score(name, [body(tr_prog[i]) for i in best.tolist()], z_tr[best])  # no EOS row, like decoded programs

    def train_views(store, variants=None):
        rows = torch.tensor([store.index[u] for u in train])
        emb, val = store.emb[rows], store.valid[rows]                     # (T, K, d), (T, K)
        if variants is not None:
            emb, val = emb[:, variants], val[:, variants]
        owner = torch.arange(len(train))[:, None].expand_as(val)
        return emb[val].to(device), owner[val].to(device)

    knn_runs = [] if a.skip_knn else [("image", 0), ("ccip_image", 0), ("text", expert)]
    if not a.skip_knn:
        knn("knn_oracle_latent", z_gt, z_tr, torch.arange(len(train), device=device))
    for m, variant in knn_runs:
        if m not in ds.stores:
            continue
        store = ds.stores[m]
        keep = [i for i, u in enumerate(test) if store.get(u, variant)[1]]
        if len(keep) < len(test):
            print(f"[knn] {m}: {len(test) - len(keep)} test uids lack this condition; scored as misses")
        q = torch.stack([store.get(u, variant)[0] for u in test]).to(device)
        keys, owner = train_views(store, [variant] if m == "text" else None)
        knn(f"knn_{m}", q, keys, owner)

    # --- the generative models, split the same way
    if a.csr_ckpt:
        from mm.gencad_io import decode_latents, load_csr
        csr, _ = load_csr(a.csr_ckpt, device)

        def decode(z, dec=csr):
            out = []
            for s in range(0, len(z), 256):
                out += decode_latents(dec, z[s: s + 256])
            return out

        # ceiling: decode each test part's own GT latent (the best latent any prior could produce)
        score("csr_recon", decode(z_gt), z_gt)
        if a.ar_decoder:
            from mm.ar_decoder import ARDecoder
            score("ar_recon", decode(z_gt, ARDecoder.load(a.ar_decoder)[0].to(device).eval()), z_gt)

        if a.gencad_dp_ckpt:
            from model import GaussianDiffusion1D, ResNetDiffusion
            net = ResNetDiffusion(d_in=256, n_blocks=10, d_main=2048, d_hidden=2048, dropout_first=0.1,
                                  dropout_second=0.1, d_out=256)
            dp = GaussianDiffusion1D(net, z_dim=256, timesteps=500, objective="pred_x0", auto_normalize=False)
            dp.load_state_dict(torch.load(a.gencad_dp_ckpt, map_location="cpu", weights_only=False)["model"])
            dp = dp.to(device).eval()
            torch.manual_seed(0)
            c = torch.stack([ds.stores["ccip_image"].get(u, 0)[0] for u in test]).to(device)
            z = torch.cat([dp.sample(cond=c[s: s + 512]) for s in range(0, len(c), 512)])
            score("gencad", decode(z), z)
        if a.prior:
            from mm.diffusion import MultiCondDiffusionPrior
            prior = MultiCondDiffusionPrior.load(a.prior)[0].to(device).eval()
            text = torch.stack([ds.stores["text"].get(u, expert)[0] for u in test]).to(device)
            if "image_mv" in prior.modalities:   # multi-view prior: score it at each requested number of views
                mv = PooledViewStore(cond_file(a.emb_dir, "image_mv"))
                images = {f"image_mv@{k}v": torch.stack([mv.get(u, k - 1)[0] for u in test]).to(device)
                          for k in a.n_views}
            else:
                images = {"image": torch.stack([ds.stores["image"].get(u, 0)[0] for u in test]).to(device)}
            combos = [{"text": text}] + [{"image": e} for e in images.values()] + \
                     [{"image": e, "text": text} for e in images.values()]
            names = ["text"] + list(images) + [f"{n}+text" for n in images]
            img_mod = "image_mv" if "image_mv" in prior.modalities else "image"
            for name, combo in zip(names, combos):
                emb = {(img_mod if m == "image" else m): e for m, e in combo.items()}
                g = torch.Generator(device=device).manual_seed(0)
                z = prior.sample(emb, guidance={m: a.guidance for m in emb}, steps=100, generator=g)
                score(f"prior:{name}", decode(z), z)

    if a.text_cad:   # direct text -> program model, no latent: score each prompt level (missing prompt = invalid)
        from mm.text_cad import TextCAD
        tc = TextCAD.load(a.text_cad)[0].to(device).eval()
        with open(os.path.join(a.data, "text_index.json")) as f:
            text_index = json.load(f)
        for lv in a.text_cad_levels:
            have = [i for i, u in enumerate(test) if text_index.get(u, {}).get(lv)]
            vecs = [None] * len(test)
            for s in range(0, len(have), 128):
                chunk = have[s: s + 128]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    out = tc.generate([text_index[test[i]][lv] for i in chunk])
                for i, v in zip(chunk, out):
                    vecs[i] = v
            print(f"[text_cad] {lv}: {len(have)}/{len(test)} test uids have this prompt level")
            score(f"text_cad:{lv}", vecs)

    if a.token_cad:  # image (+ text) tokens -> program: render view 0, alone / with the expert prompt / prompt only
        from mm.cond_encoders import load_rgb
        from mm.token_cad import TokenCAD
        from train_e2e import TF
        tk = TokenCAD.load(a.token_cad)[0].to(device).eval()
        with open(os.path.join(a.data, "text_index.json")) as f:
            text_index = json.load(f)
        with open(os.path.join(a.data, "image_ids.json")) as f:
            img_ids = json.load(f)
        modes = {"image": (True, False), "image+expert": (True, True), "expert": (False, True)}
        out = {m: [] for m in modes}
        for s in range(0, len(test), 128):
            chunk = test[s: s + 128]
            pixels = torch.stack([TF(load_rgb(os.path.join(a.data, "images", f"{u}_{img_ids[u][0]}.png")))
                                  for u in chunk]).to(device)
            prompts = [text_index.get(u, {}).get("expert") for u in chunk]
            for m, (use_img, use_txt) in modes.items():
                if not use_img and not any(prompts):
                    out[m] += [None] * len(chunk)
                    continue
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    if use_img:
                        out[m] += tk.generate(pixels=pixels, prompts=prompts if use_txt else None)
                    else:   # prompt only: samples without a prompt count as invalid
                        keep = [i for i, p in enumerate(prompts) if p]
                        got = tk.generate(prompts=[prompts[i] for i in keep])
                        row = [None] * len(chunk)
                        for i, v in zip(keep, got):
                            row[i] = v
                        out[m] += row
        for m in modes:
            score(f"token_cad:{m}", out[m])

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()
