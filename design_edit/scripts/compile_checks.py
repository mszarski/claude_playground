#!/usr/bin/env python3
"""Compile each eval-v2-style item's natural-language criteria into deterministic checks (design_edit.checks).

An LLM writes {"checks": [...], "allowed": [...]}; the result is kept only if the reference patch passes every
check with a perfect score and the unedited design fails at least one check (so "do nothing" can't pass).
On failure the LLM gets one retry with the grader's feedback; items that still fail keep judge-only grading.

    python scripts/compile_checks.py --data eval_v2.jsonl --out eval_v2_checks.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from design_edit.checks import grade_checks  # noqa: E402
from design_edit.judge import chat  # noqa: E402
from design_edit.schema import parse  # noqa: E402

SPEC = """Check language (JSON objects; all node ids refer to the design):
  {"node": "n41", "prop": "opacity", "eq": 0.5}            ops: eq, ne, in (list), lt, gt, le, ge, contains (text),
  {"node": "n8", "prop": "stroke", "absent": true}          absent/present (property removed / set),
  {"node": "n3", "prop": "size", "gt": 18}                  changed/unchanged (vs. before, value true)
  {"node": "n9", "parent": "n4"}                            node now sits inside n4
  {"deleted": "n12"}   {"exists": "n12"}
  {"new_node": {"parent": "n21", "type": "text", "text_contains": "required", "props": {"color": "#E11D48",
                "size": {"le": 13}}}}                       some newly inserted node (anywhere inside parent) matches
  {"order": ["n5", "n7"]}                                   siblings in this order
  {"same": ["n26", "n24"], "props": ["fill", "stroke"]}     these nodes end with equal values for these props
  {"count": {"type": "frame", "parent": "n4", "name_contains": "row"}, "eq": 3}
  {"judge": "<criterion text>"}                             only for genuinely subjective criteria
Plus "allowed": every existing node id the edit may change (props, deletion, children order, moving). Include
nodes whose change is clearly acceptable for the request (e.g. a button's label as well as its frame). New nodes
are always allowed."""

PROMPT = """Translate grading criteria for a UI-design edit into machine checks.

{spec}

Design system:
{ds}

Design before:
{before}

Request: {request}

Criteria (written by the item author):
{criteria}

A correct reference edit (one of possibly several valid answers):
{patch}

Write checks that every valid answer passes, not just this reference: when the criteria allow alternatives, use
ranges or "in" lists, or leave a subjective criterion as {{"judge": ...}}. Encode "Nothing unrelated changes" through
"allowed" only. At least one check must fail on the unedited design.
Reply with JSON only: {{"checks": [...], "allowed": [...]}}{feedback}"""

WRITER = "deepseek-ai/DeepSeek-V3.2"


def compile_one(row: dict) -> dict:
    try:
        return _compile_one(row)
    except Exception as e:  # noqa: BLE001  one malformed LLM reply must not kill a 5k-item run
        print(f"compile error: {type(e).__name__}: {e}", flush=True)
        return {**row, "checks": "", "allowed": ""}


def _compile_one(row: dict) -> dict:
    patch = row["completion"][0]["content"] if isinstance(row["completion"], list) else row["completion"]
    feedback = ""
    for _ in range(2):
        msg = PROMPT.format(spec=SPEC, ds=row["design_system"], before=row["before"], request=row["instruction"],
                            criteria="\n".join(f"- {c}" for c in json.loads(row["criteria"])), patch=patch,
                            feedback=feedback)
        out = chat(WRITER, [{"role": "user", "content": msg}], max_tokens=1500)
        m = re.search(r"\{.*\}", out, re.S)
        try:
            spec = json.loads(m.group(0)) if m else None
            checks, allowed = spec["checks"], spec["allowed"]
            assert isinstance(checks, list) and isinstance(allowed, list) and checks
        except (json.JSONDecodeError, KeyError, TypeError, AssertionError):
            feedback = "\n\nYour previous reply was not valid JSON of the required shape."
            continue
        ref = grade_checks(row["before"], patch, checks, allowed)
        noop = grade_checks(row["before"], "", checks, allowed)
        before = parse(row["before"])
        from design_edit.checks import eval_check

        noop_fails = any(eval_check(c, before, before) is False for c in checks)
        if ref["pass"] and ref["score"] > 0.999 and noop_fails:
            return {**row, "checks": json.dumps(checks), "allowed": json.dumps(allowed)}
        feedback = (f"\n\nYour previous checks were rejected: reference passes={ref['pass']} "
                    f"(failed: {ref['why'] or 'none'}), the unedited design fails some check={noop_fails}. Fix them.")
        del noop
    return {**row, "checks": "", "allowed": ""}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    assert os.environ.get("HF_TOKEN")
    rows = [json.loads(line) for line in open(args.data)]
    with ThreadPoolExecutor(args.workers) as pool:
        out = list(pool.map(compile_one, rows))
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in out))
    ok = sum(1 for r in out if r["checks"])
    print(f"compiled {ok}/{len(out)} items into deterministic checks -> {args.out}")


if __name__ == "__main__":
    main()
