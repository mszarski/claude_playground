#!/usr/bin/env python3
"""Add a brief rationale before each SFT target: "<think>\\n{why}\\n</think>\\n{patch}".

A teacher sees the design, the request and the reference patch, and explains in one or two sentences which
elements the request refers to and which conventions visible in the design decide the values. The student learns
to do that short reasoning before answering (fast mode stays fast: ~40 extra tokens). strip_fences drops the
<think> block before scoring, so rewards and evals need no change.

    python scripts/add_rationales.py --src in.jsonl --dst out.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from design_edit.judge import chat  # noqa: E402

PROMPT = """Here is a UI design (one node per line), an edit request, and the correct patch.

Design:
{design}

Request: {request}

Correct patch:
{patch}

In at most 35 words, write the reasoning a designer would do BEFORE writing this patch: which node(s) the request
refers to, and which conventions or values visible in the design decide the change (e.g. "the other primary buttons
use #2563EB and no stroke"). Don't restate the patch line by line, don't mention that a patch was given, and don't
hedge. Output only the reasoning."""


def rationale(row: dict, model: str, token: str) -> str:
    patch = row["completion"][0]["content"]
    out = chat(model, [{"role": "user", "content": PROMPT.format(design=row["before"], request=row["instruction"],
                                                                  patch=patch)}], max_tokens=120, temperature=0.3,
               token=token).strip()
    return " ".join(out.split()) if out and len(out.split()) <= 60 else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    token = os.environ["HF_TOKEN"]
    rows = [json.loads(line) for line in open(args.src)]
    with ThreadPoolExecutor(args.workers) as pool:
        whys = list(pool.map(lambda r: rationale(r, args.model, token), rows))
    kept = 0
    with open(args.dst, "w") as f:
        for r, why in zip(rows, whys):
            if not why:
                continue  # rows without a usable rationale are dropped, so every target has the same format
            patch = r["completion"][0]["content"]
            r = {**r, "completion": [{"role": "assistant", "content": f"<think>\n{why}\n</think>\n{patch}"}]}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept += 1
    print(f"kept {kept}/{len(rows)} -> {args.dst}")


if __name__ == "__main__":
    main()
