#!/usr/bin/env python3
"""Generate the synthetic edit dataset and optionally push it to the Hub.

Each row has:
  prompt      chat messages (system + user: serialized design + request)
  completion  [assistant message with the reference patch]   <- SFT target
  before / after / relax                                     <- GRPO reward inputs
  kind / instruction                                         <- for analysis

Train and test use disjoint design seeds, so test designs are never seen in training.
``--holdout-kinds`` additionally keeps whole edit types out of train, to measure generalization.

    python scripts/build_dataset.py --n-train 20000 --n-test 500 --out data/
    python scripts/build_dataset.py --n-train 20000 --push mszarski/design-edits
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from design_edit import edits  # noqa: E402
from design_edit.generator import generate_design  # noqa: E402
from design_edit.prompts import prompt_messages  # noqa: E402
from design_edit.schema import diff, format_patch, serialize  # noqa: E402

FAMILY_WEIGHTS = {"forward": 0.55, "repair": 0.25, "compose": 0.20}


def sample_example(rng: random.Random, seed_lo: int, seed_hi: int, kinds: dict[str, list[str]]) -> dict | None:
    family = rng.choices(list(FAMILY_WEIGHTS), weights=list(FAMILY_WEIGHTS.values()))[0]
    if not kinds[family]:
        return None
    kind = rng.choice(kinds[family])
    doc = generate_design(rng.randint(seed_lo, seed_hi), max_sections=rng.choice([1, 2, 2, 3]))
    ex = edits.ALL_EDITS[kind](doc, rng)
    if ex is None:
        return None
    patch = format_patch(diff(ex.before, ex.after))
    if not patch:
        return None
    return {
        "prompt": prompt_messages(ex.before, ex.instruction),
        "completion": [{"role": "assistant", "content": patch}],
        "before": serialize(ex.before),
        "after": serialize(ex.after),
        "relax": json.dumps(ex.relax),
        "kind": ex.kind,
        "instruction": ex.instruction,
    }


def build(n: int, seed: int, seed_range: tuple[int, int], allowed: set[str]) -> list[dict]:
    rng = random.Random(seed)
    kinds = {
        "forward": [k for k in edits.FORWARD if k in allowed],
        "repair": [k for k in edits.REPAIR if k in allowed],
        "compose": ["compose"] if "compose" in allowed else [],
    }
    rows, seen = [], set()
    while len(rows) < n:
        row = sample_example(rng, *seed_range, kinds)
        if row is None:
            continue
        key = (row["before"], row["instruction"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-kinds", default="", help="comma-separated edit kinds to keep out of train")
    ap.add_argument("--out", default="data")
    ap.add_argument("--push", default=None, help="Hub dataset repo id, e.g. user/design-edits")
    ap.add_argument("--public", action="store_true", help="make the pushed dataset public (default private)")
    args = ap.parse_args()

    holdout = {k for k in args.holdout_kinds.split(",") if k}
    unknown = holdout - set(edits.ALL_EDITS)
    if unknown:
        ap.error(f"unknown kinds: {sorted(unknown)}; choose from {sorted(edits.ALL_EDITS)}")
    all_kinds = set(edits.ALL_EDITS)

    train = build(args.n_train, args.seed, (0, 999_999), all_kinds - holdout)
    test = build(args.n_test, args.seed + 1, (1_000_000, 1_099_999), all_kinds)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("test", test)):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} rows -> {out / f'{name}.jsonl'}")
        print("  ", dict(Counter(r["kind"] for r in rows).most_common()))

    if args.push:
        from datasets import Dataset, DatasetDict

        DatasetDict({"train": Dataset.from_list(train), "test": Dataset.from_list(test)}).push_to_hub(
            args.push, private=not args.public)
        print(f"pushed to https://huggingface.co/datasets/{args.push}")


if __name__ == "__main__":
    main()
