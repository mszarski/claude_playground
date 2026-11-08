#!/usr/bin/env python3
"""Grade saved outputs on goal-level requests with designer-written rubrics and an LLM judge.

Goal requests ("make X look disabled") have many valid answers, so reference matching under-counts.
Each goal kind gets a short rubric of what a designer would accept. A judge from a different model
family than the teacher (default gpt-oss-120b) grades the applied change against it.

    python scripts/judge_goal.py outputs.json data/test_goal.jsonl
    python scripts/judge_goal.py hf://mszarski/design-edit-sft-v5/eval_test_goal.json data/test_goal.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_baseline import chat  # noqa: E402
from rescore import load_outputs  # noqa: E402

from design_edit.schema import PatchError, apply_patch, diff, format_patch, parse  # noqa: E402

RUBRICS = {
    "disable_button": """- The button clearly reads as unavailable/inactive: visibly reduced prominence (e.g. lower opacity, or a muted
  grey fill and/or grey label, or a muted outline style).
- The label must stay visible: it may be low-contrast, but not the same color as the fill.
- It must not look like an ordinary active secondary button (a white button with dark text is NOT disabled).
- Nothing other than that button (and its label) changes.
- Minimal: no extra changes beyond what the request needs, including on the edited element itself (e.g. no added
  bold, size, or unrelated color changes that the request didn't ask for).""",
    "highlight_plan": """- The target plan card is emphasized the same way as the referenced featured plan (e.g. the same accent-colored
  border; optionally also its call-to-action styled like the featured plan's).
- All text stays readable (no dark text on a saturated fill).
- Other plans do not change.
- Minimal: no extra changes beyond what the request needs, including on the edited element itself (e.g. no added
  bold, size, or unrelated color changes that the request didn't ask for).""",
    "toggle_on": """- The named toggle now looks ON, using the same on-state color as the design's other on toggles.
- Nothing else changes.
- Minimal: no extra changes beyond what the request needs, including on the edited element itself (e.g. no added
  bold, size, or unrelated color changes that the request didn't ask for).""",
}

JUDGE = """You are a senior product designer grading an edit to a UI design.

Request: {request}

Rubric (all points must hold):
{rubric}

Design before (one node per line, indented by depth):
{before}

Change applied (patch):
{patch}

Think briefly, then end with a final line exactly: VERDICT: PASS or VERDICT: FAIL"""


def judge_row(row: dict, output: str, model: str, token: str, url: str) -> bool:
    rubric = RUBRICS.get(row["kind"])
    if rubric is None:
        raise ValueError(f"no rubric for {row['kind']}")
    before = parse(row["before"])
    try:
        after = apply_patch(before, output)
    except (PatchError, ValueError, KeyError, IndexError):
        return False
    canon = format_patch(diff(before, after)) if after["id"] == before["id"] else ""
    if not canon:
        return False
    msg = JUDGE.format(request=row["instruction"], rubric=rubric, before=row["before"], patch=canon)
    # Reasoning judges can exhaust the token budget and return no verdict: retry rather than count it as FAIL.
    for _ in range(4):
        out = chat(url, token, model, [{"role": "user", "content": msg}], max_tokens=12000, temperature=0.0)
        v = re.findall(r"VERDICT:\s*(PASS|FAIL)", out.upper())
        if v:
            return v[-1] == "PASS"
    print(f"warning: judge gave no verdict, counting as FAIL: {row['instruction']}", file=sys.stderr)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs")
    ap.add_argument("data")
    ap.add_argument("--judge", default="openai/gpt-oss-120b")
    ap.add_argument("--base-url", default="https://router.huggingface.co/v1")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    token = os.environ["HF_TOKEN"]

    outs = load_outputs(args.outputs)
    rows = [json.loads(line) for line in open(args.data)]
    assert len(outs) == len(rows), "row count mismatch"
    with ThreadPoolExecutor(args.workers) as pool:
        verdicts = list(pool.map(lambda ro: judge_row(ro[0], ro[1]["output"], args.judge, token, args.base_url),
                                 zip(rows, outs)))
    by = defaultdict(list)
    for r, v in zip(rows, verdicts):
        by[r["kind"]].append(v)
    print(f"{'kind':<20}{'n':>5}{'judged pass':>13}")
    print(f"{'OVERALL':<20}{len(verdicts):>5}{sum(verdicts) / len(verdicts):>13.3f}")
    for k, v in sorted(by.items()):
        print(f"{k:<20}{len(v):>5}{sum(v) / len(v):>13.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"verdicts": verdicts}, indent=1))


if __name__ == "__main__":
    main()
