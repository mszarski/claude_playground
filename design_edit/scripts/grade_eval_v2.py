#!/usr/bin/env python3
"""Grade model outputs on eval v2 (build_eval_v2.py) against each item's criteria with an LLM judge.

Input: the eval_test.json that eval_model.py / eval_baseline.py write for the eval-v2 dataset (rows in dataset
order, each with "output"). An item passes when the patch applies, introduces nothing the design language lacks,
and the judge passes every criterion. Verdicts are cached per (item, resulting design), so identical answers from
different models are judged once.

    python scripts/grade_eval_v2.py --dataset user/design-edits-eval-v2 --outputs a.json b.json --names A B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_eval_v2 import JUDGE, grade  # noqa: E402

from design_edit.leak import dsl_violations  # noqa: E402
from design_edit.schema import PatchError, apply_patch, parse, serialize, strip_fences  # noqa: E402


def grade_row(row: dict, output: str, cache: dict, judge: str) -> dict:
    before = parse(row["before"])
    patch = strip_fences(output).strip()
    if not patch:
        return {"pass": False, "frac": 0.0, "why": "empty"}
    try:
        after = apply_patch(before, patch)
    except (PatchError, ValueError, KeyError, IndexError) as e:
        return {"pass": False, "frac": 0.0, "why": f"invalid: {str(e)[:80]}"}
    if dsl_violations(before, after):
        return {"pass": False, "frac": 0.0, "why": "not in the design language"}
    if serialize(after) == serialize(before):
        return {"pass": False, "frac": 0.0, "why": "no change"}
    key = hashlib.sha1((row["instruction"] + row["before"] + serialize(after)).encode()).hexdigest()
    if key not in cache:
        crit = json.loads(row["criteria"])
        v = None
        for _ in range(2):  # the judge occasionally returns unparseable output
            v = grade(row["design_system"], before, row["instruction"], crit, after, model=judge)
            if v:
                break
        cache[key] = v
    v = cache[key]
    if not v:
        return {"pass": None, "frac": None, "why": "judge unavailable"}
    return {"pass": all(v), "frac": sum(v) / len(v), "why": "" if all(v) else
            "failed: " + "; ".join(c for c, ok in zip(json.loads(row["criteria"]), v) if not ok)[:300]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outputs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--judge", default=JUDGE)
    ap.add_argument("--cache", default="eval_v2_judge_cache.json")
    ap.add_argument("--save", default=None, help="write per-row grades here (JSON)")
    args = ap.parse_args()
    from datasets import load_dataset

    rows = list(load_dataset(args.dataset, split="test"))
    cache = json.loads(Path(args.cache).read_text()) if os.path.exists(args.cache) else {}
    names = args.names or [Path(p).stem for p in args.outputs]
    report = {}
    for name, path in zip(names, args.outputs):
        outs = [r["output"] for r in json.load(open(path))["rows"]]
        assert len(outs) == len(rows), f"{path}: {len(outs)} outputs for {len(rows)} items"
        with ThreadPoolExecutor(16) as pool:
            grades = list(pool.map(lambda i: grade_row(rows[i], outs[i], cache, args.judge), range(len(rows))))
        Path(args.cache).write_text(json.dumps(cache))
        by = defaultdict(list)
        for r, g in zip(rows, grades):
            if g["pass"] is not None:
                by[r["kind"]].append(g)
                by["ALL"].append(g)
        report[name] = {k: {"pass": round(sum(g["pass"] for g in v) / len(v), 3),
                            "criteria": round(sum(g["frac"] for g in v) / len(v), 3), "n": len(v)}
                        for k, v in sorted(by.items())}
        if args.save:
            Path(args.save.replace(".json", f"_{name}.json")).write_text(json.dumps(grades, indent=1))
        print(name, json.dumps(report[name]["ALL"]), flush=True)
    kinds = sorted({k for r in report.values() for k in r})
    print(f"\n{'model':28}" + "".join(f"{k[3:] if k.startswith('v2_') else k:>13}" for k in kinds))
    for name, r in report.items():
        print(f"{name:28}" + "".join(f"{r.get(k, {}).get('pass', float('nan')):>13.3f}" for k in kinds))


if __name__ == "__main__":
    main()
