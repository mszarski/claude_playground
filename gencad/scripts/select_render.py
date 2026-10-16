"""Pick one program per part from several candidates by how well its render matches the input renders.

Candidates are CadQuery programs for the same test uids from several predictions files (greedy outputs of different
models, Claude, samples from train_vlm_cq.py --n_samples). Each candidate is executed, scored against the GT with the
cq_iou.py protocol (for reporting only), and rendered with the input cameras (scripts/render_views.py: same
orthographic directions, same bounding-sphere framing, so a correct part reproduces the input pixels). The selector
sees only what a user would give the model: the input views (all 4 for view4 modes, view 0 for view1 modes).

Selector scores per candidate (higher = better), averaged over the input views:
  sil  - IoU of the foreground masks (pixel < 250)
  pix  - minus mean |difference| of the grey images (shading and drawn edges, so holes and steps count too)
Invalid candidates (no solid) are never picked unless every candidate is invalid.

  python scripts/select_render.py --data data --out results/select/view4 --views 4 \
      --cand results/vlm_cq_v1_test/preds.pkl:vlm_cq:view4 results/claude_baseline/preds.pkl:claude:view4 \
             'results/vlm_cq_samples/preds.pkl:vlm_cq:view4@s*'
Writes <out>/rows.pkl (per uid, per candidate: valid, gt, own, sil, pix) and <out>/report.json; prints the table.
"""
import argparse
import fnmatch
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_input(stem, views, size):
    from PIL import Image
    return [np.asarray(Image.open(f"{stem}_{k}.png").convert("L").resize((size, size), Image.BILINEAR), np.float32)
            for k in views]


def work(args):
    """One (uid, candidate): GT IoU (cq_iou protocol) + render agreement with the input views."""
    gt_vec, code, stem, views, size = args
    from mm.cadquery_io import run_cadquery
    from scripts.cq_iou import score
    from scripts.render_views import DIRS, render, tessellate_cq
    r = score((gt_vec, code, "cadquery"))
    if r is None:
        return None                                   # GT does not build: part not scored
    if not r["valid"]:
        return {**r, "sil": -1.0, "pix": -1e9}
    shape = run_cadquery(code)
    tris, fid = tessellate_cq(shape)
    sil, pix = [], []
    for k, ref in zip(views, load_input(stem, views, size)):
        img = np.asarray(render(tris, fid, DIRS[k], size=size), np.float32)
        a, b = img < 250, ref < 250
        sil.append((a & b).sum() / max((a | b).sum(), 1))
        pix.append(-np.abs(img - ref).mean())
    return {**r, "sil": float(np.mean(sil)), "pix": float(np.mean(pix))}


def summary(rows, dup):
    out = {}
    for g, mask in {"all": np.ones(len(rows), bool), "dup": dup, "novel": ~dup}.items():
        r = [x for x, k in zip(rows, mask) if k and x is not None]
        out[g] = {"n": len(r), "IR": 100 * float(np.mean([not x["valid"] for x in r])),
                  "IoU_gt": 100 * float(np.mean([x["gt"] for x in r])),
                  "IoU_own": 100 * float(np.mean([x["own"] for x in r]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--cand", nargs="+", required=True,
                    help="preds.pkl:method[=alias] (fnmatch pattern allowed in the method, e.g. 'vlm_cq:view4@s*')")
    ap.add_argument("--views", type=int, choices=[1, 4], required=True, help="input views the selector may use")
    ap.add_argument("--size", type=int, default=128, help="render side for the comparison")
    ap.add_argument("--n", type=int, default=None, help="first n uids only")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--task_timeout", type=float, default=180.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import h5py
    from pebble import ProcessPool

    names, codes, uids, dup = [], [], None, None
    for spec in a.cand:
        spec, alias = spec.split("=", 1) if "=" in spec else (spec, None)
        path, pat = spec.split(":", 1)
        with open(path, "rb") as f:
            P = pickle.load(f)
        if uids is None:
            uids, dup = list(P["uids"]), np.asarray(P["dup"], bool)
        assert list(P["uids"]) == uids, f"{path} scores different uids"
        hit = sorted(m for m in P["preds"] if fnmatch.fnmatch(m, pat))
        assert hit, f"no method matches {pat} in {path} ({sorted(P['preds'])})"
        for m in hit:
            name = m.replace(pat.split("@")[0], alias, 1) if alias else m
            names.append(name if name not in names else f"{name}#{len(names)}")
            codes.append(P["preds"][m])
    n = a.n or len(uids)
    uids, dup = uids[:n], dup[:n]
    print(f"[select] {n} uids x {len(names)} candidates: {names}", flush=True)
    views = [0, 1, 2, 3] if a.views == 4 else [0]

    rows = [[None] * len(names) for _ in range(n)]
    failed = 0
    t0 = time.time()
    with ProcessPool(max_workers=a.workers) as pool:
        futs = {}
        for i, u in enumerate(uids):
            with h5py.File(os.path.join(a.data, "cad_vec", u + ".h5"), "r") as f:
                g = f["vec"][:].astype(np.int64)
            stem = os.path.join(a.data, "renders", u)
            for c in range(len(names)):
                futs[(i, c)] = pool.schedule(work, args=((g, codes[c][i], stem, views, a.size),),
                                             timeout=a.task_timeout)
        for k, ((i, c), fu) in enumerate(futs.items()):
            try:
                rows[i][c] = fu.result()
            except Exception:  # noqa: BLE001 - timeout / crashed worker: count it, treat as invalid
                failed += 1
                rows[i][c] = {"valid": False, "gt": 0.0, "own": 0.0, "sil": -1.0, "pix": -1e9, "failed": True}
            if k % 2000 == 0:
                print(f"[select] {k}/{len(futs)} {time.time() - t0:.0f}s", flush=True)

    keep = [i for i in range(n) if all(r is not None for r in rows[i])]   # GT builds
    report = {"n": n, "candidates": names, "views": views, "size": a.size, "timeout_or_crash": failed,
              "single": {}, "select": {}}

    def pick(i, cand, key):
        rs = [rows[i][c] for c in cand]
        return rs[int(np.argmax([r[key] for r in rs]))]

    def table(label, picked):
        s = summary([picked[i] if i in keep_set else None for i in range(n)], dup)
        print(f"{label:44s} " + "  ".join(f"{g}: own {v['IoU_own']:.1f} gt {v['IoU_gt']:.1f} IR {v['IR']:.1f}"
                                         for g, v in s.items()), flush=True)
        return s

    keep_set = set(keep)
    for c, m in enumerate(names):
        report["single"][m] = table(m, {i: rows[i][c] for i in keep})
    groups = {"all": list(range(len(names)))}
    samples = [c for c, m in enumerate(names) if "@s" in m]
    greedy = [c for c, m in enumerate(names) if "@s" not in m]
    if samples and greedy:
        groups["greedy_only"] = greedy
        groups["first_greedy+samples"] = [greedy[0]] + samples
    for gname, cand in groups.items():
        for key in ("sil", "pix"):
            report["select"][f"{gname}:{key}"] = table(f"select {gname} by {key}", {i: pick(i, cand, key) for i in keep})
        report["select"][f"{gname}:oracle"] = table(f"oracle {gname} (best own IoU)",
                                                    {i: pick(i, cand, "own") for i in keep})
    # how often each candidate is picked by the main selector
    counts = np.bincount([int(np.argmax([rows[i][c]["sil"] for c in groups["all"]])) for i in keep],
                         minlength=len(names))
    report["picked_by_sil"] = dict(zip(names, counts.tolist()))
    print(f"[select] picked by sil: {report['picked_by_sil']}; timeout/crash {failed}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "rows.pkl"), "wb") as f:
        pickle.dump({"uids": uids, "dup": dup.tolist(), "names": names, "rows": rows}, f)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()
