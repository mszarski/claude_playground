#!/usr/bin/env python3
"""Reward audit: does the label-free RL reward agree with ground truth?

Takes saved model outputs on a held-out split that HAS reference answers, hides the references, and scores each
output with the goal reward used in RL (design_edit.goal_reward). Then compares with the true exact-match label:

    false accept = output is wrong, but gets the maximum reward   -> a blind spot RL can exploit
    false reject = output is right, but gets less than the maximum -> reward noise that slows RL down

    python scripts/audit_reward.py data/test_novel_big.jsonl out1.json out2.json --per-kind 40 \
        --version 2 --refs teacher1.json teacher2.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from design_edit.geometry import Renderer  # noqa: E402
from design_edit.goal_reward import goal_reward  # noqa: E402
from design_edit.schema import PatchError, apply_patch, parse, serialize  # noqa: E402

_tl = threading.local()


def renderer() -> Renderer:
    if not hasattr(_tl, "r"):
        _tl.r = Renderer().__enter__()
    return _tl.r


def teacher_refs(rows: list[dict], files: list[str]) -> list[list[str]]:
    """Per row: the designs produced by each teacher's (valid, non-empty) answer."""
    refs: list[list[str]] = [[] for _ in rows]
    for f in files:
        outs = json.load(open(f))["rows"]
        for i, (r, o) in enumerate(zip(rows, outs)):
            try:
                after = serialize(apply_patch(parse(r["before"]), o["output"]))
            except (PatchError, ValueError, KeyError, IndexError):
                continue
            if after != r["before"] and after not in refs[i]:
                refs[i].append(after)
    return refs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("outputs", nargs="+", help="eval JSONs (rows with kind/output/exact) on the same data")
    ap.add_argument("--per-kind", type=int, default=40, help="max wrong and max right outputs sampled per kind")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--refs", nargs="*", default=[], help="teacher eval JSONs on the same data (version 2)")
    ap.add_argument("--judge", default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.data)]
    refs = teacher_refs(rows, args.refs) if args.refs else [[] for _ in rows]
    pool: dict[tuple[str, bool], dict[tuple[int, str], None]] = defaultdict(dict)
    for f in args.outputs:
        for i, o in enumerate(json.load(open(f))["rows"][:len(rows)]):
            if o.get("valid", 1):
                pool[(rows[i]["kind"], o["exact"] >= 1)][(i, o["output"].strip())] = None  # dedupe identical outputs
    rng = random.Random(args.seed)
    cases = []
    for (kind, right), items in sorted(pool.items()):
        keys = list(items)
        rng.shuffle(keys)
        cases += [(kind, right, i, out) for i, out in keys[: args.per_kind]]

    def run(case):
        kind, right, i, out = case
        res = goal_reward(rows[i]["before"], rows[i]["instruction"], out, judge_model=args.judge, renderer=renderer(),
                          version=args.version, refs=refs[i])
        return {"kind": kind, "right": right, "row": i, "output": out, "reward": res["reward"],
                "ref_f1": res["ref_f1"], "verdict": res["verdict"], "penalty": res["penalty"], "n_refs": len(refs[i])}

    with ThreadPoolExecutor(args.workers) as ex:
        results = [r for r in ex.map(run, cases) if r["verdict"] is not None]

    by = defaultdict(lambda: {True: [], False: []})
    for r in results:
        by[r["kind"]][r["right"]].append(r["reward"])
    print(f"reward version {args.version}, refs from {len(args.refs)} teacher(s); accept = reward >= 0.999")
    print(f"{'kind':<20}{'wrong n':>8}{'FALSE ACCEPT':>14}{'mean rew':>10}{'right n':>9}{'false reject':>14}{'mean rew':>10}")
    tot = {True: [], False: []}
    for kind in sorted(by):
        w, r = by[kind][False], by[kind][True]
        tot[False] += w
        tot[True] += r
        fa = f"{sum(x >= 0.999 for x in w) / len(w):.2f}" if w else "-"
        fr = f"{sum(x < 0.999 for x in r) / len(r):.2f}" if r else "-"
        mw = f"{sum(w) / len(w):.2f}" if w else "-"
        mr = f"{sum(r) / len(r):.2f}" if r else "-"
        print(f"{kind:<20}{len(w):>8}{fa:>14}{mw:>10}{len(r):>9}{fr:>14}{mr:>10}")
    w, r = tot[False], tot[True]
    print(f"{'OVERALL':<20}{len(w):>8}{sum(x >= 0.999 for x in w) / max(1, len(w)):>14.2f}{sum(w) / max(1, len(w)):>10.2f}"
          f"{len(r):>9}{sum(x < 0.999 for x in r) / max(1, len(r)):>14.2f}{sum(r) / max(1, len(r)):>10.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
