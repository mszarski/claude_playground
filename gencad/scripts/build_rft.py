"""Rejection-sampling fine-tuning targets from scripts/sample_rft.py chunks: per part, the best program among greedy and
samples by GT IoU (GT normalisation, the test metric), kept if it reaches --min_iou. Ties prefer the greedy program,
then the shortest. Writes a parquet for train_vlm_cq.py --rft (uid, code, views as JSON, prompt, scores) and prints
how many targets differ from greedy.

  python scripts/build_rft.py --chunks results/rft_v1 --out results/rft_v1/rft.parquet
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="sample_rft.py output dir")
    ap.add_argument("--min_iou", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = []
    for fn in sorted(glob.glob(os.path.join(a.chunks, "chunk_*.pkl")), key=lambda f: int(f.split("_")[-1][:-4])):
        with open(fn, "rb") as f:
            recs += pickle.load(f)
    rows, n_gt_fail = [], 0
    for r in recs:
        if r["scores"][0] is None:
            n_gt_fail += 1
            continue
        codes = [r["greedy"]] + list(r["samples"])
        cand = [(s["gt"], -k, -len(c), k) for k, (c, s) in enumerate(zip(codes, r["scores"]))
                if c and s and s["valid"]]
        if not cand:
            continue
        gt_iou, _, _, k = max(cand)
        if gt_iou < a.min_iou:
            continue
        rows.append({"uid": r["uid"], "mode": r["mode"], "views": json.dumps(r["views"]), "prompt": r["prompt"] or "",
                     "code": codes[k], "best_gt": gt_iou, "greedy_gt": r["scores"][0]["gt"] if r["scores"][0]["valid"]
                     else 0.0, "from_sample": k > 0})
    df = pd.DataFrame(rows)
    df.to_parquet(a.out)
    imp = df.best_gt - df.greedy_gt
    stats = {"parts": len(recs), "gt_does_not_build": n_gt_fail, "targets": len(df),
             "target_is_a_sample": int(df.from_sample.sum()), "improves_greedy_by_0.02": int((imp > 0.02).sum()),
             "improves_greedy_by_0.10": int((imp > 0.10).sum()), "fixes_invalid_greedy": int((df.greedy_gt == 0).sum()),
             "mean_target_iou": round(float(df.best_gt.mean()), 4), "by_mode": df["mode"].value_counts().to_dict()}
    print(json.dumps(stats, indent=1))
    with open(os.path.splitext(a.out)[0] + "_stats.json", "w") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
