#!/usr/bin/env python3
"""Repair eval-v2 grading criteria (eval v2.1).

An audit of the items every strong model failed found grading defects rather than model errors:
* alternative solutions written as separate criteria ("One acceptable solution ...", "Another acceptable solution
  ...") are AND-ed by the judge, so every answer fails one of them;
* criteria demand the id or child index of a node the edit creates, which is an arbitrary choice;
* exact values where the request is vague ("a bit wider" -> "w=280 or more").

Stage 1 (every item): an LLM rewrites the criteria under those rules. The rewrite is kept only if the reference
answer still passes every new criterion (judge); otherwise the item keeps its original criteria.
Stage 2 (--adjudicate, items all given models fail under the new criteria): the LLM sees the request, criteria,
reference and the models' answers and decides keep / fix (new criteria, again validated on the reference) / drop
(ambiguous request or a reference a designer would not accept). A "fix" whose criteria the reference fails is a drop.

    python scripts/repair_eval_v2.py --data eval_v2_checks.jsonl --out eval_v21.jsonl
    python scripts/repair_eval_v2.py --data eval_v21.jsonl --out eval_v21b.jsonl --adjudicate \
        --grades g_a.json g_b.json --outputs a.json b.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_eval_v2 import JUDGE, grade, parse_json  # noqa: E402

from design_edit.judge import chat  # noqa: E402
from design_edit.schema import apply_patch, parse, strip_fences  # noqa: E402

WRITER = "deepseek-ai/DeepSeek-V3.2"

RULES = """Rules for good criteria:
1. Each criterion must hold for EVERY valid answer. Alternative solutions never get separate criteria: merge them
   into one, e.g. "The indicator is clearly more prominent, e.g. a filled background with a contrasting label, or a
   bolder/larger label".
2. Never require the id of a node the edit creates, nor a numeric child index. Describe position relative to
   existing nodes ("inside the Input section, right after the email input").
3. Exact values only where the request or the design system pins them down (a named token, a matching sibling).
   For vague requests ("a bit wider", "stand out more") require the direction and a noticeable amount, not a number.
4. Keep every requirement that the request really implies, and keep "Nothing unrelated to the request changes."
5. Same language and specificity otherwise; don't add new requirements."""

REWRITE = """You maintain an evaluation set of UI-design edit requests. Rewrite one item's grading criteria so that
they are fair.

{rules}

Design system:
{ds}

Design before:
{before}

Request: {request}

Current criteria:
{criteria}

A reference answer (one valid edit, patch format):
{patch}

Reply with JSON only: {{"criteria": ["...", ...], "changed": true|false}}"""

ADJUDICATE = """You maintain an evaluation set of UI-design edit requests. Every strong model failed this item,
which often means the item, not the models, is wrong. Decide:
* "keep": the criteria are fair and the models' answers really are wrong;
* "fix": some answers are reasonable readings of the request that the criteria unfairly reject; give fair criteria;
* "drop": the request is ambiguous enough that designers would disagree, or the reference itself is a poor edit.

{rules}

Design system:
{ds}

Design before:
{before}

Request: {request}

Criteria:
{criteria}

Reference answer:
{patch}

Model answers (and the criteria they failed):
{answers}

Reply with JSON only: {{"verdict": "keep"|"fix"|"drop", "reason": "...", "criteria": [...]}}
("criteria" only for "fix")."""


def ref_patch(row: dict) -> str:
    return row["completion"][0]["content"] if isinstance(row["completion"], list) else row["completion"]


def ref_passes(row: dict, criteria: list[str]) -> bool:
    before = parse(row["before"])
    after = apply_patch(before, strip_fences(ref_patch(row)))
    for _ in range(2):
        v = grade(row["design_system"], before, row["instruction"], criteria, after, model=JUDGE)
        if v:
            return all(v)
    return False


def rewrite_one(row: dict) -> dict:
    try:
        crit = json.loads(row["criteria"])
        out = parse_json(chat(WRITER, [{"role": "user", "content": REWRITE.format(
            rules=RULES, ds=row["design_system"], before=row["before"], request=row["instruction"],
            criteria="\n".join(f"- {c}" for c in crit), patch=ref_patch(row))}], max_tokens=2000))
        if not isinstance(out, dict) or not out.get("changed") or not out.get("criteria"):
            return {**row, "repair": "unchanged"}
        new = [str(c) for c in out["criteria"]]
        if not ref_passes(row, new):
            return {**row, "repair": "rejected (reference fails)"}
        return {**row, "criteria_orig": row["criteria"], "criteria": json.dumps(new), "repair": "rewritten"}
    except Exception as e:  # noqa: BLE001  one bad reply must not kill the run
        return {**row, "repair": f"error {type(e).__name__}"}


def adjudicate_one(row: dict, answers: list[tuple[str, str]]) -> dict:
    try:
        crit = json.loads(row["criteria"])
        ans = "\n\n".join(f"[{n}] failed: {why or '-'}\n{patch}" for n, patch, why in answers)
        out = parse_json(chat(WRITER, [{"role": "user", "content": ADJUDICATE.format(
            rules=RULES, ds=row["design_system"], before=row["before"], request=row["instruction"],
            criteria="\n".join(f"- {c}" for c in crit), patch=ref_patch(row), answers=ans)}], max_tokens=3000))
        if not isinstance(out, dict) or out.get("verdict") not in ("keep", "fix", "drop"):
            return {**row, "adjudication": "error"}
        why = str(out.get("reason", ""))[:300]
        if out["verdict"] == "fix":
            new = [str(c) for c in out.get("criteria") or []]
            if not new or not ref_passes(row, new):
                # the fair reading rejects the reference itself: the item can't be graded reliably
                return {**row, "adjudication": "drop", "adj_reason": "reference fails the fair criteria: " + why}
            return {**row, "criteria_pre_adj": row["criteria"], "criteria": json.dumps(new),
                    "adjudication": "fix", "adj_reason": why}
        return {**row, "adjudication": out["verdict"], "adj_reason": why}
    except Exception as e:  # noqa: BLE001
        return {**row, "adjudication": f"error {type(e).__name__}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adjudicate", action="store_true")
    ap.add_argument("--grades", nargs="*", default=[], help="per-row grades (grade_eval_v2.py --save) per model")
    ap.add_argument("--outputs", nargs="*", default=[], help="eval_test.json per model, same order as --grades")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    assert os.environ.get("HF_TOKEN")
    rows = [json.loads(line) for line in open(args.data)]
    with ThreadPoolExecutor(args.workers) as pool:
        if not args.adjudicate:
            out = list(pool.map(rewrite_one, rows))
            stats = [r["repair"] for r in out]
        else:
            G = [json.load(open(p)) for p in args.grades]
            O = [[r["output"] for r in json.load(open(p))["rows"]] for p in args.outputs]
            todo = [i for i in range(len(rows)) if not any(g[i].get("pass") for g in G)]
            res = list(pool.map(lambda i: adjudicate_one(rows[i], [
                (f"model {k + 1}", strip_fences(O[k][i]).strip()[:1500], G[k][i].get("why", ""))
                for k in range(len(G))]), todo))
            out = list(rows)
            for i, r in zip(todo, res):
                out[i] = r
            stats = [r["adjudication"] for r in res]
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in out))
    from collections import Counter

    print(Counter(stats), "->", args.out)


if __name__ == "__main__":
    main()
