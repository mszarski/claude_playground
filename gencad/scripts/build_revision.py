"""Training data for the build-inspect-fix revision turn, from scripts/sample_rft.py chunks (model attempts on DeepCAD
train parts, each scored against the GT).

For each part: the greedy attempt, plus one flawed sample (IoU < --bad or failed) when there is one. Each attempt is
rebuilt here: if it fails, the error text is recorded; if it builds, it is rendered with all 4 fixed cameras (same as
the input renders). Target: the attempt itself when it is already right (IoU >= --keep: teaches "leave it"), else the
GT program (DeepCAD converted to CadQuery, fidelity >= 0.95), else the best sample if >= --keep_best.

  python scripts/build_revision.py --chunks results/rft_v1 --out data/revision --deepcad_cq data/deepcad_cq.parquet
Writes <out>/revision.parquet (uid, views, prompt, attempt, attempt_iou, error, attempt_stem, target, kind) and
<out>/attempts/<id>_{0..3}.png.
"""
import argparse
import glob
import json
import os
import pickle
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def error_of(code):
    """Build the program; -> (shape or None, error text or None)."""
    from mm.cad_helpers import namespace
    ns = namespace()
    try:
        exec(code, ns)
        r = ns.get("r")
        if r is None:
            return None, "the program does not assign r"
        shape = r.val() if hasattr(r, "val") else r
        if shape is None or not hasattr(shape, "Volume"):
            return None, "r holds no solid"
        if not shape.isValid():
            return None, "the result is not a valid solid"
        if shape.Volume() <= 1e-9:
            return None, "the result has no volume"
        return shape, None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {' '.join(str(e).split())[:200]}"


def work(args):
    rid, code, out = args
    from scripts.render_views import DIRS, render, tessellate_cq
    shape, err = error_of(code)
    if err:
        return rid, err, None
    try:
        tris, fid = tessellate_cq(shape)
        stem = os.path.join(out, "attempts", rid)
        for j, d in enumerate(DIRS):
            render(tris, fid, d).save(f"{stem}_{j}.png", optimize=True)
        return rid, None, stem
    except Exception as e:  # noqa: BLE001
        return rid, f"render failed: {type(e).__name__}", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--deepcad_cq", default="data/deepcad_cq.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bad", type=float, default=0.9)
    ap.add_argument("--keep", type=float, default=0.98)
    ap.add_argument("--keep_best", type=float, default=0.95)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import pandas as pd
    from pebble import ProcessPool
    rng = random.Random(a.seed)
    dc = pd.read_parquet(a.deepcad_cq)
    gt_code = {u: c for u, c, ok, i in zip(dc.uid, dc.code, dc.ok, dc.iou) if ok and 0.95 <= i <= 1.001}
    recs = []
    for fn in sorted(glob.glob(os.path.join(a.chunks, "chunk_*.pkl"))):
        recs += pickle.load(open(fn, "rb"))
    rows = []
    for r in recs:
        if r["scores"][0] is None:
            continue
        codes = [r["greedy"]] + list(r["samples"])
        v = [(s["gt"] if (c and s and s["valid"]) else -1.0) for c, s in zip(codes, r["scores"])]
        picks = [0]
        bad = [k for k in range(1, len(codes)) if codes[k] and v[k] < a.bad]
        if bad:
            picks.append(rng.choice(bad))
        best = max(range(len(codes)), key=lambda k: v[k])
        for k in picks:
            if v[k] >= a.keep:
                target, kind = codes[k], "keep"
            elif r["uid"] in gt_code:
                target, kind = gt_code[r["uid"]], "fix_gt"
            elif v[best] >= a.keep_best:
                target, kind = codes[best], "fix_best"
            else:
                continue
            rows.append({"rid": f"{r['uid'].replace('/', '_')}_{k}", "uid": r["uid"], "mode": r["mode"],
                         "views": json.dumps(r["views"]), "prompt": r["prompt"] or "", "attempt": codes[k],
                         "attempt_iou": v[k], "target": target, "kind": kind})
    print(f"[rev] {len(rows)} examples from {len(recs)} parts: "
          f"{pd.Series([x['kind'] for x in rows]).value_counts().to_dict()}", flush=True)
    os.makedirs(os.path.join(a.out, "attempts"), exist_ok=True)
    res, t0 = {}, time.time()
    with ProcessPool(max_workers=a.workers) as pool:
        futs = [pool.schedule(work, args=((x["rid"], x["attempt"], a.out),), timeout=60) for x in rows]
        for i, fu in enumerate(futs):
            try:
                rid, err, stem = fu.result()
            except Exception as e:  # noqa: BLE001
                rid, err, stem = rows[i]["rid"], f"build timed out ({type(e).__name__})", None
            res[rid] = (err, stem)
            if i % 5000 == 0:
                print(f"[rev] {i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    for x in rows:
        x["error"], x["attempt_stem"] = res[x["rid"]]
        if x["attempt_stem"]:
            x["attempt_stem"] = os.path.relpath(x["attempt_stem"], a.out)
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(a.out, "revision.parquet"))
    print(f"[rev] wrote {len(df)}: failed attempts {int(df.error.notna().sum())}, rendered "
          f"{int(df.attempt_stem.notna().sum())}; kinds {df.kind.value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
