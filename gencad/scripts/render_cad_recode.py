"""Render CAD-Recode programs (scripts/fetch_cad_recode.py parquet) with the same 4 cameras, shading and edges as
our DeepCAD renders (scripts/render_views.py), so one model can train on both. Each program is executed with
CadQuery; programs that do not build a non-empty valid solid are skipped.

Sharded so several CPU jobs can split the ~1M programs: shard i of n takes rows i, i+n, i+2n, ...
Output: <out>/<path without .py>_<k>.png (k = camera 0..3) and <out>/status_<i>.json {path: status}.

  python scripts/render_cad_recode.py --parquet data/cad_recode.parquet --out data/recode_renders --shard 0 --n_shards 4
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm.sysutil import usable_cpus  # noqa: E402
from scripts.render_views import DIRS, render, tessellate_cq  # noqa: E402


def render_one(args):
    path, code, out = args
    from mm.cadquery_io import run_cadquery
    stem = os.path.join(out, path[:-3])
    paths = [f"{stem}_{k}.png" for k in range(len(DIRS))]
    if all(os.path.exists(p) for p in paths):
        return "skip"
    shape = run_cadquery(code)
    if shape is None:
        return "no_solid"
    tris, fid = tessellate_cq(shape)
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    for d, p in zip(DIRS, paths):
        render(tris, fid, d).save(p, optimize=True)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/cad_recode.parquet")
    ap.add_argument("--out", default="data/recode_renders")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=usable_cpus())
    ap.add_argument("--task_timeout", type=float, default=60.0)
    a = ap.parse_args()
    from pebble import ProcessPool
    df = pd.read_parquet(a.parquet).iloc[a.shard:: a.n_shards]
    if a.limit:
        df = df.iloc[: a.limit]
    print(f"[recode] shard {a.shard}/{a.n_shards}: {len(df)} programs", flush=True)
    status = {}
    with ProcessPool(max_workers=a.workers) as pool:
        futs = [(p, pool.schedule(render_one, args=((p, c, a.out),), timeout=a.task_timeout))
                for p, c in zip(df.path, df.code)]
        for k, (p, fu) in enumerate(futs):
            try:
                status[p] = fu.result()
            except Exception as e:  # noqa: BLE001 - timeout / crashed worker
                status[p] = f"error {repr(e)[:80]}"
            if k % 10000 == 0:
                print(f"[recode] {k}/{len(futs)}", flush=True)
    counts = {}
    for s in status.values():
        counts[s.split()[0]] = counts.get(s.split()[0], 0) + 1
    print(f"[recode] {counts}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"status_{a.shard}.json"), "w") as f:
        json.dump(status, f)


if __name__ == "__main__":
    main()
