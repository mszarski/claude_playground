"""How often does each prior turn a plain rectangular plate into something else?

Takes the test models whose ground truth is the simplest CAD program there is (SOL, four axis-aligned lines, one
extrude) and checks what each prior generates for them: same program structure? still axis-aligned? how far off
is the worst corner? Also decodes the ground-truth latent itself (csr_recon) to separate decoder error from prior
error, and dumps every sample for --uid (default: the trapezoid from the visual check) over --n_seeds seeds.

  python scripts/diagnose_rectangles.py --emb_dir data/emb --data data --csr_ckpt data/ckpt/ae_ckpt_epoch1000.pth \
      --gencad_dp_ckpt data/ckpt/sketch_cond_diffusion_ckpt_epoch1000000.pt \
      --prior mm_v1=results/mm_v1/last.pt --prior mm_ccip=results/mm_ccip/last.pt --out results/rect/report.json
"""
import argparse
import json
import os
import sys
from collections import Counter

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadlib.macro import ALL_COMMANDS, EOS_IDX, EXT_IDX, LINE_IDX, SOL_IDX  # noqa: E402
from mm.data import CadLatentStore, EmbeddingStore  # noqa: E402
from mm.diffusion import MultiCondDiffusionPrior  # noqa: E402
from mm.gencad_io import decode_latents, load_csr  # noqa: E402
from mm.rerank import best_of_n  # noqa: E402


def rect_points(vec):
    """Corner points (4, 2) if vec is exactly SOL + 4 lines + 1 extrude (trailing EOS ignored), else None."""
    if vec is None:
        return None
    vec = vec[[int(c) != EOS_IDX for c in vec[:, 0]]]  # stored GT keeps its EOS row, decoded sequences don't
    if [int(c) for c in vec[:, 0]] != [SOL_IDX] + [LINE_IDX] * 4 + [EXT_IDX]:
        return None
    return vec[1:5, 1:3].astype(int)


def axis_error(pts):
    """Largest off-axis offset over the 4 edges, in quantised sketch units (0 for a perfect rectangle)."""
    p = np.concatenate([pts, pts[:1]])
    return int(max(min(abs(b[0] - a[0]), abs(b[1] - a[1])) for a, b in zip(p[:-1], p[1:])))


def classify(vec, gt_pts, tol):
    if vec is None:
        return "no_eos", None
    pts = rect_points(vec)
    if pts is None:
        cmds = Counter(int(c) for c in vec[:, 0])
        return f"other_program({cmds.get(LINE_IDX, 0)}L/{cmds.get(EXT_IDX, 0)}E)", None
    err = axis_error(pts)
    return ("rectangle" if err <= tol else "skewed_quad"), err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="data/emb")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--csr_ckpt", required=True)
    ap.add_argument("--gencad_dp_ckpt", default=None)
    ap.add_argument("--prior", action="append", default=[], help="name=path, repeatable")
    ap.add_argument("--max_uids", type=int, default=1000)
    ap.add_argument("--tol", type=int, default=2, help="off-axis tolerance in quantised units (of 256)")
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--uid", default="0068/00682829")
    ap.add_argument("--n_seeds", type=int, default=16)
    ap.add_argument("--best_of", type=int, default=0, help="also run each prior condition with best-of-N reranking")
    ap.add_argument("--samplers", nargs="*", default=["0:100"],
                    help="eta:steps pairs for the priors' DDIM sampler (eta 0 = deterministic, 1 = DDPM-like); "
                         "anything but the first gets an '@eta<e>s<steps>' suffix")
    ap.add_argument("--out", default="results/rect/report.json")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    csr, _ = load_csr(a.csr_ckpt, device)

    cad = CadLatentStore(os.path.join(a.emb_dir, "cad_latents.h5"))
    stores = {m: EmbeddingStore(os.path.join(a.emb_dir, f"cond_{m}.h5")) for m in ["image", "text", "ccip_image"]}
    with h5py.File(os.path.join(a.emb_dir, "cond_text.h5"), "r") as f:
        expert = json.loads(f.attrs["variants"]).index("expert")
    with open(os.path.join(a.data, "filtered_data.json")) as f:
        split = json.load(f)[a.split]

    def gt_vec(u):
        with h5py.File(os.path.join(a.data, "cad_vec", u + ".h5"), "r") as f:
            return f["vec"][:]

    def has_all(u):
        return u in cad.index and all(u in s.index and s.valid[s.index[u]].any() for s in stores.values())

    if a.uid in cad.index:
        v = gt_vec(a.uid)
        print(f"[rect] GT program of {a.uid}: " + " ".join(ALL_COMMANDS[int(c)] for c in v[:, 0]))
        print(v.tolist())
    pattern = Counter()
    uids, gts = [], []
    for u in split:
        if not has_all(u):
            continue
        g = gt_vec(u)
        pattern[" ".join(ALL_COMMANDS[int(c)] for c in g[:, 0])] += 1
        p = rect_points(g)
        if p is not None and axis_error(p) == 0:
            uids.append(u)
            gts.append(p)
        if len(uids) >= a.max_uids:
            break
    if a.uid not in uids and has_all(a.uid) and rect_points(gt_vec(a.uid)) is not None:
        uids.append(a.uid)
        gts.append(rect_points(gt_vec(a.uid)))
    print(f"[rect] {len(uids)} {a.split} uids whose GT is one axis-aligned rectangle extrude")
    print("[rect] most common GT programs:", pattern.most_common(5))

    def cond(u_list, m, variant=0):
        return torch.stack([stores[m].get(u, variant)[0] for u in u_list]).to(device)

    # every "method" maps a uid list + seed to decoded CAD programs
    methods = {"csr_recon": lambda us, seed: decode_latents(
        csr, torch.stack([cad.z[cad.index[u]] for u in us]).to(device))}
    if a.gencad_dp_ckpt:
        from model import GaussianDiffusion1D, ResNetDiffusion
        net = ResNetDiffusion(d_in=256, n_blocks=10, d_main=2048, d_hidden=2048, dropout_first=0.1,
                              dropout_second=0.1, d_out=256)
        dp = GaussianDiffusion1D(net, z_dim=256, timesteps=500, objective="pred_x0", auto_normalize=False)
        dp.load_state_dict(torch.load(a.gencad_dp_ckpt, map_location="cpu", weights_only=False)["model"])
        dp = dp.to(device).eval()

        def gencad(us, seed):
            torch.manual_seed(seed)
            return decode_latents(csr, dp.sample(cond=cond(us, "ccip_image")))
        methods["gencad"] = gencad
    for spec in a.prior:
        name, path = spec.split("=", 1)
        prior, _ = MultiCondDiffusionPrior.load(path)
        prior = prior.to(device).eval()
        combos = {"ccip_image": ["ccip_image"], "image": ["image"], "image+text:expert": ["image", "text"],
                  "text:expert": ["text"]}
        for label, mods in combos.items():
            if not all(m in prior.modalities for m in mods):
                continue

            for k, spec in enumerate(a.samplers):
                eta, steps = float(spec.split(":")[0]), int(spec.split(":")[1])

                def run(us, seed, prior=prior, mods=mods, n=1, eta=eta, steps=steps):
                    g = torch.Generator(device=device).manual_seed(seed)
                    emb = {m: cond(us, m, expert if m == "text" else 0) for m in mods}
                    kw = dict(guidance={m: a.guidance for m in mods}, steps=steps, eta=eta, generator=g)
                    if n > 1:
                        return best_of_n(prior, csr, emb, n=n, **kw)[0]
                    return decode_latents(csr, prior.sample(emb, **kw))
                key = f"{name}:{label}" + ("" if k == 0 else f"@eta{eta:g}s{steps}")
                methods[key] = run
                if a.best_of > 1 and k == 0:
                    methods[f"{key}@bo{a.best_of}"] = lambda us, seed, run=run: run(us, seed, n=a.best_of)

    report = {"n_uids": len(uids), "tol": a.tol, "methods": {}, "uid": a.uid, "uid_samples": {}}
    with torch.no_grad():
        for name, fn in methods.items():
            vecs = []
            for s in range(0, len(uids), 512):
                vecs += fn(uids[s: s + 512], 0)
            cls = [classify(v, g, a.tol) for v, g in zip(vecs, gts)]
            counts = Counter(c for c, _ in cls)
            errs = [e for c, e in cls if c == "skewed_quad"]
            row = {k: v / len(uids) for k, v in sorted(counts.items())}
            row["median_skew_units"] = float(np.median(errs)) if errs else 0.0
            report["methods"][name] = row
            print(f"{name:26s} rectangle={row.get('rectangle', 0):.3f} skewed_quad={row.get('skewed_quad', 0):.3f} "
                  f"other={1 - row.get('rectangle', 0) - row.get('skewed_quad', 0):.3f} "
                  f"median_skew={row['median_skew_units']:.0f}", flush=True)
            # the visual-check uid, several seeds
            if a.uid in uids:
                rows = []
                for seed in range(a.n_seeds):
                    v = fn([a.uid], seed)[0]
                    c, e = classify(v, None, a.tol)
                    rows.append({"seed": seed, "class": c, "skew": e,
                                 "points": None if rect_points(v) is None else rect_points(v).tolist()})
                    if name == "csr_recon":
                        break
                report["uid_samples"][name] = rows
    if a.uid in uids:
        report["uid_gt_points"] = gts[uids.index(a.uid)].tolist()
        print(f"[uid {a.uid}] GT corners {report['uid_gt_points']}")
        for name, rows in report["uid_samples"].items():
            print(f"  {name:26s} " + " ".join(r["class"][:4] + (f"({r['skew']})" if r["skew"] else "") for r in rows))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[rect] wrote {a.out}")


if __name__ == "__main__":
    main()
