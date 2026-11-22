#!/usr/bin/env python3
"""Build the round-9 training sets (three arms, same test splits as the source dataset).

* clean        source train minus every row whose effect matches a held-out test type (design_edit.leak)
* broad        clean with its teacher-written goal rows (healed_goal) replaced by the same number of typed rows,
               picked round-robin across edit types so every type contributes about equally
* broad-think  exactly the rows of broad, each target preceded by a brief <think> rationale

Rows without a rationale are left out of broad as well, so broad and broad-think differ only in the rationale.

    python scripts/build_v9.py --clean clean.jsonl --clean-r clean_r.jsonl --typed typed_*.jsonl \
        --typed-r typed_r.jsonl --src user/design-edits-v8 --dst user/design-edits-v9
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict


def load(paths: list[str]) -> list[dict]:
    return [json.loads(line) for p in paths for line in open(p)]


def key(r: dict) -> tuple[str, str]:
    return r["before"], r["instruction"]


def pick_typed(typed: list[dict], n: int, seed: int = 0) -> list[dict]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in typed:
        by_type[r["kind"]].append(r)
    rng = random.Random(seed)
    for rows in by_type.values():
        rng.shuffle(rows)
    out, i = [], 0
    while len(out) < n and any(i < len(v) for v in by_type.values()):
        out += [v[i] for v in by_type.values() if i < len(v)][: n - len(out)]
        i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--clean-r", required=True)
    ap.add_argument("--typed", nargs="+", required=True)
    ap.add_argument("--typed-r", required=True, help="typed rows with rationales (add_rationales.py on --typed)")
    ap.add_argument("--src", required=True, help="dataset whose non-train splits are copied")
    ap.add_argument("--dst", required=True, help="prefix; pushes <dst>-clean, <dst>-broad, <dst>-broad-think")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    clean = load([args.clean])
    rat = {key(r): r["completion"] for r in load([args.clean_r]) + load([args.typed_r])}
    typed = list({key(r): r for r in load(args.typed) if key(r) in rat}.values())
    keep = [r for r in clean if r["kind"] != "healed_goal"]
    n_goal = len(clean) - len(keep)
    chosen = pick_typed(typed, n_goal)
    broad = [r for r in keep + chosen if key(r) in rat]
    think = [{**r, "completion": rat[key(r)]} for r in broad]
    kinds = {r["kind"] for r in chosen}
    print(f"clean {len(clean)} | replaced {n_goal} healed_goal rows with {len(chosen)} typed rows from "
          f"{len(kinds)} types (of {len(typed)} available) | broad/broad-think {len(broad)} rows")
    if args.dry_run:
        return

    from datasets import Dataset, DatasetDict, load_dataset

    src = load_dataset(args.src)
    cols = list(src["train"].features)  # match the source schema so the copied splits fit in one DatasetDict

    def ds(rows):
        return Dataset.from_list([{c: r.get(c, "[]" if c in ("relax", "alts", "refs") else r.get("instruction", "")
                                    if c == "orig_instruction" else "") for c in cols} for r in rows],
                                 features=src["train"].features)

    for name, rows in (("clean", clean), ("broad", broad), ("broad-think", think)):
        out = {"train": ds(rows).shuffle(seed=0)}
        out.update({k: v for k, v in src.items() if k != "train"})
        DatasetDict(out).push_to_hub(f"{args.dst}-{name}", private=True)
        print("pushed", f"{args.dst}-{name}", len(rows))


if __name__ == "__main__":
    main()
