#!/usr/bin/env python3
"""Round 12 data: reasoning SFT on both trace sets, plus an RL split of prompts the base model only sometimes solves.

* train: correct thinking traces (filter_traces.py) from the synthetic broad-p2 set and from the realistic
  design-system set (train_real, graded by its compiled checks so valid alternatives count).
* <dst>-rl: realistic items with deterministic checks whose 4 base samples were right 1-3 times (DAPO-style dynamic
  sampling, done once up front), plus a smaller share of 0/4 items: rollouts there are mostly 0 reward, but a model
  improved by SFT solves some.
* other splits are copied from the source dataset (v1 tests).

    python scripts/build_v12.py --sft a.jsonl b.jsonl --real train_real_checks.jsonl --real-traces traces.jsonl \
        --src user/design-edits-v9-broad-p2 --dst user/design-edits-v12-think
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from design_edit.checks import grade_checks  # noqa: E402
from design_edit.schema import strip_fences  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", nargs="+", default=[], help="filter_traces.py outputs")
    ap.add_argument("--real", required=True, help="realistic items with checks (compile_checks.py output)")
    ap.add_argument("--real-traces", required=True, help="sample_traces.py output on --real (same order)")
    ap.add_argument("--src", default=None)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--hard-frac", type=float, default=0.25, help="share of 0/k prompts in the rl split")
    ap.add_argument("--rl-only", action="store_true",
                    help="re-pick RL prompts only (e.g. from a later model's samples): push them as <dst>")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sft = [json.loads(line) for p in args.sft for line in open(p)]
    real = [json.loads(line) for line in open(args.real)]
    mixed, hard = [], []
    for line in open(args.real_traces):
        t = json.loads(line)
        r = real[t["index"]]
        if not r["checks"]:
            continue
        n_ok = sum(grade_checks(r["before"], strip_fences(s["text"]), r["checks"], r["allowed"])["pass"]
                   for s in t["samples"])
        (mixed if 0 < n_ok < len(t["samples"]) else hard if n_ok == 0 else []).append(r)
    rng = random.Random(0)
    rng.shuffle(hard)
    rl = mixed + hard[:int(len(mixed) * args.hard_frac / (1 - args.hard_frac))]
    rng.shuffle(rl)
    print(f"sft rows {len(sft)} | rl: {len(mixed)} mixed + {len(rl) - len(mixed)} hard = {len(rl)} "
          f"(of {len(hard)} hard available)")
    if args.dry_run:
        return

    from datasets import Dataset, DatasetDict, load_dataset
    from huggingface_hub import HfApi

    rl_cols = ["prompt", "completion", "before", "after", "relax", "kind", "instruction", "checks", "allowed",
               "criteria", "design_system"]
    rl_ds = Dataset.from_list([{c: r.get(c, "") for c in rl_cols} for r in rl])
    root = Path(__file__).resolve().parents[1]
    if args.rl_only:
        DatasetDict({"train": rl_ds}).push_to_hub(args.dst, private=True)
        for d in ("design_edit", "scripts"):
            HfApi().upload_folder(repo_id=args.dst, repo_type="dataset", folder_path=str(root / d),
                                  path_in_repo=f"code/{d}", ignore_patterns=["__pycache__/*", "*.pyc"])
        print("pushed", args.dst)
        return
    src = load_dataset(args.src)
    feats = src["train"].features
    train = Dataset.from_list([{c: r.get(c, "") for c in feats} for r in sft], features=feats).shuffle(seed=0)
    # Splits of one repo must share features, so the RL prompts (with checks) go to <dst>-rl.
    DatasetDict({"train": train, **{k: v for k, v in src.items() if k != "train"}}).push_to_hub(args.dst, private=True)
    DatasetDict({"train": rl_ds}).push_to_hub(args.dst + "-rl", private=True)
    for repo in (args.dst, args.dst + "-rl"):
        for d in ("design_edit", "scripts"):
            HfApi().upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(root / d),
                                  path_in_repo=f"code/{d}", ignore_patterns=["__pycache__/*", "*.pyc"])
    print("pushed", args.dst, args.dst + "-rl")


if __name__ == "__main__":
    main()
