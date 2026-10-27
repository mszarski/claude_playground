#!/usr/bin/env bash
# usage: final_chain.sh <prefix: final2> <orient prefix: or2> <compare prefix: final>
# raw score + guard, then (after orient jobs) merge oriented turns, guard, paired comparisons.
set -u
cd /home/user/claude_playground/gencad
P=$1; O=$2; C=$3
for s in real rich dc; do mkdir -p results/${P}_$s && hf buckets cp hf://buckets/mszarski/gencad-work/results/${P}_$s/revise.pkl results/${P}_$s/ >/dev/null; done
python scripts/rich_iou.py --preds results/${P}_real/revise.pkl --out results/${P}_real/rich_iou.json --rows results/${P}_real/rich_rows.pkl 2>&1 | grep -E 'all:'
python scripts/cq_iou.py --preds results/${P}_dc/revise.pkl --out results/${P}_dc/cq_iou_local.json --rows results/${P}_dc/rows.pkl 2>&1 | grep -E 'all:' | cut -c1-120
python scripts/rich_iou.py --preds results/${P}_rich/revise.pkl --out results/${P}_rich/rich_iou.json --rows results/${P}_rich/rich_rows.pkl 2>&1 | grep -E 'all:'
# wait for the orient jobs of this prefix
for R in ${O}_real_view4 ${O}_real_view1 ${O}_rich_view4 ${O}_rich_view4_r1 ${O}_rich_view1 ${O}_rich_view1_r1 ${O}_dc_view4 ${O}_dc_view1 ${O}_dc_view1_expert; do
  mkdir -p results/$R
  until hf buckets cp hf://buckets/mszarski/gencad-work/results/$R/orient_rows.pkl results/$R/ >/dev/null 2>&1; do sleep 300; done
  hf buckets cp hf://buckets/mszarski/gencad-work/results/$R/orient.pkl results/$R/ >/dev/null
done
python3 - "$P" "$O" <<'EOF'
import os, pickle, sys
P, O = sys.argv[1], sys.argv[2]
sets = {"real": ["real_view4", "real_view1"], "rich": ["rich_view4", "rich_view4_r1", "rich_view1", "rich_view1_r1"],
        "dc": ["dc_view4", "dc_view1", "dc_view1_expert"]}
for name, runs in sets.items():
    Pk = W = None
    for r in runs:
        p = pickle.load(open(f"results/{O}_{r}/orient.pkl", "rb")); w = pickle.load(open(f"results/{O}_{r}/orient_rows.pkl", "rb"))
        if Pk is None:
            Pk, W = {**p, "preds": {}}, {**w, "rows": {}}
        ren = lambda k: k.replace("+orient", "").replace("vlm_cq:", "vlm_cq_o:")
        Pk["preds"].update({ren(k): v for k, v in p["preds"].items()}); W["rows"].update({ren(k): v for k, v in w["rows"].items()})
    os.makedirs(f"results/{P}_{name}_o", exist_ok=True)
    pickle.dump(Pk, open(f"results/{P}_{name}_o/og.pkl", "wb")); pickle.dump(W, open(f"results/{P}_{name}_o/og_rows.pkl", "wb"))
EOF
for n in real dc rich; do python scripts/revise_guard.py --preds results/${P}_${n}_o/og.pkl --rows results/${P}_${n}_o/og_rows.pkl --out results/${P}_${n}_o/guard.json 2>&1 | grep -E '\[guard\]'; done
python3 - "$P" "$C" <<'EOF'
import pickle, sys, numpy as np
P, C = sys.argv[1], sys.argv[2]
def pair(ag, bg, idx, lab):
    ok = [i for i in idx if ag(i) is not None and bg(i) is not None]
    a = np.array([ag(i) for i in ok]); b = np.array([bg(i) for i in ok]); d = b - a
    print(f"{lab:46s} n={len(ok)} ref {100*a.mean():.1f} new {100*b.mean():.1f} d {100*d.mean():+.2f} se {100*d.std()/len(d)**.5:.2f}")
for ref_name, ref in [("old fullres", None), (C, C)]:
    for s, key, n, oldrows in [("real", "own", 600, "results/real_old/rich_rows.pkl"), ("dc", "gt", 2000, "results/vlm_grpo_fullres_test/rows.pkl")]:
        G = pickle.load(open(f"results/{P}_{s}_o/guard_picks.pkl", "rb"))["picked"]
        if ref is None:
            A = pickle.load(open(oldrows, "rb"))["rows"]
            for v in (["view4", "view1"] if s == "real" else ["view4", "view1", "view1+expert"]):
                pair(lambda i: A["vlm_cq:" + v][i][key] if A["vlm_cq:" + v][i] else None, lambda i: G[f"vlm_cq_o:{v}@r1+guard"].get(i), range(n), f"{s} {v} pipeline vs {ref_name}")
        else:
            GA = pickle.load(open(f"results/{C}_{s}_o/guard_picks.pkl", "rb"))["picked"]
            for k in G:
                pair(lambda i: GA[k].get(i), lambda i: G[k].get(i), range(n), f"{s} {k} vs {C}")
    G = pickle.load(open(f"results/{P}_rich_o/guard_picks.pkl", "rb"))["picked"]
    if ref is not None:
        GA = pickle.load(open(f"results/{C}_rich_o/guard_picks.pkl", "rb"))["picked"]
        for k in G:
            pair(lambda i: GA[k].get(i), lambda i: G[k].get(i), range(1989), f"rich {k} vs {C}")
EOF
