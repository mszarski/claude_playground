#!/usr/bin/env python3
"""Rewrite template requests the way real people type them, using an LLM via the HF router.

The LLM sees the design so it can refer to elements naturally ("the CTA", "the big price"),
but every concrete value in the original request (quoted text, numbers, hex colors) must
survive verbatim. Rows that fail that check are retried, then dropped.
Only ``instruction`` and the user message change; the reference edit is untouched.

    python scripts/paraphrase.py data/test.jsonl data/test_paraphrased.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_baseline import chat  # noqa: E402

from design_edit.prompts import SYSTEM_PROMPT  # noqa: E402

STYLES = [
    "terse and lowercase, like a quick Slack message to a teammate",
    "a polite, complete sentence from a product manager",
    "a designer using jargon (CTA, hero, copy, padding, hierarchy, nudge, bump)",
    "casual and a bit rambling, with one filler phrase",
    "a short imperative note left as a comment on the design",
    "a non-native English speaker, simple words, slightly unusual phrasing",
]

PROMPT = """Rewrite a UI edit request in a different voice. Style: {style}.

Rules:
- Same meaning: the same element(s) must be changed in the same way. Do not add or drop any part of the request.
- You may refer to elements differently (by role, position, color or content) as long as it is still unambiguous in this design.
- Keep every quoted text, number and hex color exactly as written (you may change the quote marks).
- Output only the rewritten request, nothing else.

Design:
{design}

Original request: {instruction}"""

_QUOTED = re.compile(r"[“\"]([^”\"]+)[”\"]")
_VALUES = re.compile(r"#[0-9A-Fa-f]{6}\b|\b\d+(?:\.\d+)?\b")


def required_values(instruction: str) -> list[str]:
    quoted = _QUOTED.findall(instruction)
    rest = _QUOTED.sub(" ", instruction)
    return quoted + _VALUES.findall(rest)


def preserves_values(original: str, rewritten: str) -> bool:
    low = rewritten.lower()
    return all(v.lower() in low for v in required_values(original))


VERIFY = """You check whether a rewritten UI edit request still asks for exactly the same change.

Design:
{design}

Original request A: {a}
Rewritten request B: {b}

Step 1. List each separate change request A asks for (element + what changes). A complaint like
"X feels cramped" counts as a request to fix X.
Step 2. For each one, quote the words in B that ask for the same change to the same element,
or write MISSING.
Step 3. List anything B asks for that A does not, or write NONE.
Step 4. Last line: VERDICT: SAME if nothing is missing, changed or added; otherwise VERDICT: DIFFERENT."""


def equivalent(row: dict, rewritten: str, model: str, token: str, base_url: str) -> bool:
    msg = VERIFY.format(design=row["before"], a=row["instruction"], b=rewritten)
    out = chat(base_url, token, model, [{"role": "user", "content": msg}], max_tokens=600, temperature=0.0)
    verdict = re.findall(r"VERDICT:\s*(SAME|DIFFERENT)", out.upper())
    return bool(verdict) and verdict[-1] == "SAME"


def rewrite(row: dict, model: str, token: str, base_url: str, rng: random.Random, tries: int = 3) -> dict | None:
    for _ in range(tries):
        msg = PROMPT.format(style=rng.choice(STYLES), design=row["before"], instruction=row["instruction"])
        out = chat(base_url, token, model, [{"role": "user", "content": msg}], max_tokens=200, temperature=0.9)
        out = out.strip().strip('"').strip()
        if out and not out.startswith("// request failed") and len(out) < 600 and preserves_values(row["instruction"], out) \
                and equivalent(row, out, model, token, base_url):
            new = dict(row)
            new["orig_instruction"] = row["instruction"]
            new["instruction"] = out
            new["prompt"] = [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": f"Design:\n{row['before']}\n\nRequest: {out}"}]
            return new
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    ap.add_argument("--base-url", default="https://router.huggingface.co/v1")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    token = os.environ["HF_TOKEN"]

    rows = [json.loads(line) for line in open(args.inp)][: args.n]
    rngs = [random.Random(args.seed * 100_003 + i) for i in range(len(rows))]
    with ThreadPoolExecutor(args.workers) as pool:
        out = list(pool.map(lambda ir: rewrite(ir[1], args.model, token, args.base_url, rngs[ir[0]]), enumerate(rows)))
    kept = [r for r in out if r is not None]
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"kept {len(kept)}/{len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
