#!/usr/bin/env python3
"""Repair teacher edits that fill an outlined button but leave the outline on.

The teacher often writes "set n8 fill=#DC2626; set n9 color=#FFFFFF" for "make it look like the primary button".
The outlined (secondary) button keeps its grey stroke, which no filled button in these designs has. A student trained
on 258 such rows stopped removing the outline at all (`promote_button` 0.59 -> 0.09 exact).

Rule: a node that had a stroke, got a new dark/saturated fill, and whose stroke the edit did not touch loses the stroke,
provided the design's other dark-filled nodes have none. Labels are regenerated with `diff`, so they stay canonical.

    python scripts/repair_outline.py --src user/design-edits-v7 --dst user/design-edits-v8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from design_edit.rewards import _luminance  # noqa: E402
from design_edit.schema import diff, format_patch, parse, serialize  # noqa: E402


def _walk(n):
    yield n
    for c in n.get("children", []):
        yield from _walk(c)


def _dark(v) -> bool:
    try:
        return isinstance(v, str) and v.startswith("#") and _luminance(v) < 0.5
    except (ValueError, TypeError):
        return False


def repair_doc(before: dict, after: dict) -> int:
    """Remove left-over outlines in ``after`` (in place). Returns how many nodes were fixed."""
    old = {n["id"]: n["props"] for n in _walk(before)}
    filled = [p for p in old.values() if _dark(p.get("fill"))]
    if not filled or any(p.get("stroke") for p in filled):
        return 0  # no "filled means no outline" convention in this design
    fixed = 0
    for n in _walk(after):
        o, p = old.get(n["id"]), n["props"]
        if (o and o.get("stroke") and p.get("stroke") == o["stroke"] and p.get("fill") != o.get("fill")
                and _dark(p.get("fill")) and not _dark(o.get("fill"))):
            del p["stroke"]
            p.pop("stroke_w", None)
            fixed += 1
    return fixed


def repair_row(row: dict, kinds: set[str]) -> tuple[dict, int]:
    n = 0
    before = parse(row["before"])
    if row["kind"] in kinds and row["after"] != row["before"]:
        after = parse(row["after"])
        n = repair_doc(before, after)
        if n:
            row = {**row, "after": serialize(after),
                   "completion": [{"role": "assistant", "content": format_patch(diff(before, after))}]}
    refs = json.loads(row.get("refs") or "[]")
    if refs:
        out = []
        for r in refs:
            doc = parse(r)
            n += repair_doc(before, doc)
            out.append(serialize(doc))
        row = {**row, "refs": json.dumps(list(dict.fromkeys(out)))}
    return row, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--splits", default="train,rl_v2", help="splits to repair; the rest are copied unchanged")
    ap.add_argument("--kinds", default="healed_goal", help="train kinds to repair (teacher-written rows)")
    args = ap.parse_args()
    from datasets import DatasetDict, load_dataset

    ds = load_dataset(args.src)
    kinds, out = set(args.kinds.split(",")), {}
    for name, split in ds.items():
        if name not in args.splits.split(","):
            out[name] = split
            continue
        counts = []

        def fix(row):
            new, n = repair_row(row, kinds)
            counts.append(n)
            return new

        out[name] = split.map(fix, load_from_cache_file=False)
        print(f"{name}: repaired {sum(1 for c in counts if c)} of {len(split)} rows ({sum(counts)} nodes)")
    DatasetDict(out).push_to_hub(args.dst, private=True)
    print("pushed", args.dst)


if __name__ == "__main__":
    main()
