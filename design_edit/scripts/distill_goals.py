#!/usr/bin/env python3
"""Distill goal-level edits from a reasoning teacher, filtered by programmatic checks and an LLM judge.

1. Proposer LLM writes one goal-level request per design ("make the pricing feel less cluttered",
   "give the card more visual hierarchy"), in many styles. Requests resembling the held-out goals
   (edits.NOVEL / edits.GOAL) are rejected by keyword, so those test splits stay held out.
2. Teacher (a reasoning model) writes the patch.
3. Keep only patches that parse, apply, change something, add no design-lint violations,
   and that a judge LLM confirms accomplish the request.

Output rows match the dataset schema (reference after = the teacher's result).

    python scripts/distill_goals.py --n 3000 --out data/distilled.jsonl
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

from design_edit.generator import generate_design  # noqa: E402
from design_edit.prompts import SYSTEM_PROMPT, prompt_messages  # noqa: E402
from design_edit.rewards import lint  # noqa: E402
from design_edit.schema import PatchError, apply_patch, diff, format_patch, serialize, strip_fences  # noqa: E402

# Requests touching held-out goals are rejected (test_novel / test_goal must stay unseen).
HELD_OUT = re.compile(
    r"disabl|inactive|grey(ed)? out|gray(ed)? out|unavailable|highlight|featured|toggle|switch(ed)? (on|off)|"
    r"turn(ed)? (on|off)|enable|spread|space between|evenly|distribut|primary|call to action|\bcta\b|promot|"
    r"\bbold\b|darker|darken|border|outline|wide|width|duplicate|copy", re.I)
# Plan-card emphasis is the held-out highlight_plan goal.
PLAN_EMPH = re.compile(
    r"(\bplans?\b|\b(free|starter|pro|team|enterprise)\b).*(stand out|emphasi|prominen|attention|draw the eye|focal|"
    r"visually distinct|pop)|(stand out|emphasi|prominen|attention|focal|visually distinct|pop).*"
    r"(\bplans?\b|\b(free|starter|pro|team|enterprise)\b)", re.I)

ANGLES = [
    "visual hierarchy", "readability", "consistency with the rest of the design", "a calmer, less cluttered feel",
    "emphasis on one element", "de-emphasizing a secondary element", "brand color usage", "tighter or looser rhythm",
    "matching one element's style to another", "making a section feel more grouped", "a more modern look",
    "accessibility of text", "reordering content by importance", "simplifying a component",
]

PROPOSE = """You are a product designer reviewing this UI design (one node per line, indented by depth):

{design}

Write ONE short edit request a designer might give, focused on: {angle}.
It must be a goal (what should improve), not a list of property changes, and must refer to specific elements by
name or content so it is unambiguous. It must be achievable by editing existing properties or reordering/removing
nodes. Do not mention disabled states, highlighting, toggles, borders, widths, bold text, primary/CTA styling,
spreading items out, or duplicating. Output only the request."""

JUDGE = """A designer asked for an edit to a UI design. Decide whether the applied change accomplishes the request
well, without unrelated changes.

Request: {request}

Design before:
{before}

Changes applied:
{patch}

Answer with one word: GOOD or BAD."""


def propose(doc_text: str, rng: random.Random, model: str, token: str, url: str) -> str | None:
    out = chat(url, token, model, [{"role": "user", "content": PROPOSE.format(design=doc_text, angle=rng.choice(ANGLES))}],
               max_tokens=120, temperature=1.0).strip().strip('"')
    if not out or out.startswith("// request failed") or len(out) > 300 or HELD_OUT.search(out) or PLAN_EMPH.search(out):
        return None
    return out


def solve(doc, request: str, model: str, token: str, url: str) -> str | None:
    out = chat(url, token, model, prompt_messages(doc, request), max_tokens=8192, temperature=0.0)
    return None if out.startswith("// request failed") else strip_fences(out).strip()


def judge(before_text: str, request: str, patch: str, model: str, token: str, url: str) -> bool:
    out = chat(url, token, model, [{"role": "user", "content": JUDGE.format(request=request, before=before_text, patch=patch)}],
               max_tokens=5, temperature=0.0)
    return out.strip().upper().startswith("GOOD")


def one(seed: int, args, token: str) -> tuple[str, dict | None]:
    rng = random.Random(seed)
    lo = 3_000_000 if args.propose_only else 2_000_000  # disjoint designs from the distilled SFT set
    doc = generate_design(rng.randint(lo, lo + 999_999), max_sections=rng.choice([1, 2, 2, 3]))
    text = serialize(doc)
    req = propose(text, rng, args.proposer, token, args.base_url)
    if req is None:
        return "rejected_request", None
    if args.propose_only:  # unlabeled goal prompt for RL with a judge reward
        return "kept", {"prompt": prompt_messages(doc, req), "completion": [{"role": "assistant", "content": ""}],
                        "before": text, "after": text, "relax": "[]", "kind": "goal_prompt",
                        "instruction": req, "orig_instruction": req, "alts": "[]"}
    patch = solve(doc, req, args.teacher, token, args.base_url)
    if not patch:
        return "no_patch", None
    try:
        after = apply_patch(doc, patch)
    except (PatchError, ValueError, KeyError, IndexError):
        return "invalid_patch", None
    canon = format_patch(diff(doc, after)) if after["id"] == doc["id"] else ""
    if not canon:
        return "no_change", None
    if set(lint(after)) - set(lint(doc)):
        return "adds_lint", None
    if not judge(text, req, canon, args.judge, token, args.base_url):
        return "judge_rejected", None
    return "kept", {
        "prompt": prompt_messages(doc, req),
        "completion": [{"role": "assistant", "content": canon}],
        "before": text, "after": serialize(after), "relax": "[]", "kind": "distilled_goal",
        "instruction": req, "orig_instruction": req, "alts": "[]",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="designs to try")
    ap.add_argument("--out", required=True)
    ap.add_argument("--proposer", default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    ap.add_argument("--teacher", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--judge", default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    ap.add_argument("--base-url", default="https://router.huggingface.co/v1")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--propose-only", action="store_true", help="only write goal prompts (no teacher, no labels)")
    args = ap.parse_args()
    token = os.environ["HF_TOKEN"]

    from collections import Counter

    stats: Counter = Counter()
    kept = 0
    with ThreadPoolExecutor(args.workers) as pool, open(args.out, "w") as f:
        for i, (status, row) in enumerate(pool.map(lambda s: one(s, args, token),
                                                   range(args.seed * 1_000_000, args.seed * 1_000_000 + args.n))):
            stats[status] += 1
            if row is not None:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                kept += 1
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{args.n} kept={kept} {dict(stats)}", flush=True)
    print(f"done: kept {kept}/{args.n} -> {args.out}\n{dict(stats)}")


if __name__ == "__main__":
    main()
