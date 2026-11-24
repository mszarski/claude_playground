#!/usr/bin/env python3
"""Eval v2: realistic screens, a design system in the prompt, and criteria-graded requests.

Why: the v1 test sets score exact match against one generated answer, and the held-out "unseen" kinds follow the
generator's own conventions, which the prompt never states. Eval v2 instead
  * has a strong LLM write varied app screens in the design language (20 screen types, not 5 generator templates),
  * gives each screen a design system (design_edit.design_system) so component conventions are in context,
  * pairs each request with 2-5 checkable criteria written with it, graded by an LLM judge (grade_eval_v2.py),
    so any edit that meets them counts,
  * keeps an item only if the writer's own reference patch applies cleanly and passes its criteria.

    python scripts/build_eval_v2.py --n-designs 110 --out data/eval_v2.jsonl [--push user/design-edits-eval-v2]
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

from design_edit.design_system import render_design_system, sample_design_system, user_message_ds  # noqa: E402
from design_edit.judge import chat  # noqa: E402
from design_edit.leak import dsl_violations  # noqa: E402
from design_edit.prompts import SYSTEM_PROMPT  # noqa: E402
from design_edit.schema import PatchError, apply_patch, parse, serialize, strip_fences, walk  # noqa: E402

SCREENS = ["analytics dashboard with stat cards", "checkout form with order summary", "sign-in screen",
           "account settings page with sections", "product detail page", "onboarding step with progress",
           "email composer", "chat conversation", "kanban board column", "pricing plans", "notification list",
           "file browser list", "event details modal", "search results list", "empty state", "team members table",
           "billing and invoices page", "mobile bottom navigation with a feed", "support ticket form",
           "project overview with tabs"]

KINDS = {
    "state": "a component state change defined by the design system (disable or enable a button, make a button "
             "primary/secondary/destructive, select a tab/option/plan, mark an input invalid with an error)",
    "direct": "a direct edit of one or two properties or texts (wording, size, spacing, a color token)",
    "goal": "a goal-level request a designer might say (improve hierarchy, make something stand out, declutter, "
            "fix readability) where several solutions are acceptable",
    "structure": "a structural edit (add, remove, reorder or regroup elements, e.g. add helper text under a field "
                 "or a secondary button next to a primary one), following the design system",
    "consistency": "a consistency fix (make one element match another, or bring an off-system value back to the "
                   "design system)",
}

DESIGN_PROMPT = """{system}

That is the system prompt of a model that edits designs written in the design language above.
Write a realistic {screen} screen in this design language, as a designer would build it.

Design system to follow:
{ds}

Rules: 15-45 nodes; the root is frame#n1 "<screen name>" with w and layout; ids n1, n2, ... in document order;
use only the properties named in the language (name, text, layout, gap, pad, align, justify, size, weight, color,
fill, stroke, radius, opacity, w, h); colors as 6-digit hex; give every node a meaningful name (e.g. "Save button",
"Email input", "Email label"). Buttons are frames containing a text label. Include a few components in different
states where it fits (a primary and a secondary button, inputs, a selected tab). Leave one or two small
imperfections a designer might fix (an off-scale size or spacing, an inconsistent color), but nothing broken.
Reply with the design only."""

TASK_PROMPT = """You are writing an evaluation set for a model that edits UI designs. The model sees the design
system, the design and one request, and must reply with a patch in this language:

{system}

Design system:
{ds}

Design:
{design}

Write {k} requests, one of each kind below, as a user of a design tool would phrase them (short, natural, no ids,
no hex codes unless a real user would say them):
{kinds}

For each request also write:
* "criteria": 2-5 short statements that must ALL hold for the edited design to be correct. They must be strict
  enough that an insufficient or token edit fails (e.g. for "make X stand out", a 1px size bump must fail).
  Make them checkable
  from the edited design (name the nodes by id and name, give exact design-system values when the design system
  defines the answer). Where several solutions are acceptable, say so in the criterion (e.g. "the Upgrade button
  is visually more prominent than the other buttons, e.g. primary styling or larger size"). Always end with one
  criterion of the form "Nothing unrelated to the request changes."
* "patch": one correct, minimal patch.

Reply with JSON only: [{{"kind": "...", "request": "...", "criteria": ["...", ...], "patch": "..."}}, ...]"""

GRADE_PROMPT = """You are a senior product designer grading an edit to a UI design against criteria.

Design system:
{ds}

Design before:
{before}

Request: {request}

Changes the edit made (every change, including invisible ones):
{changes}

Design after:
{after}

Criteria:
{criteria}

For each criterion decide PASS or FAIL from the design after the edit. Be strict: a criterion passes only if the
edited design clearly satisfies it; an edit that technically touches the right node but would not convince a
designer fails. Reply with JSON only, e.g.
{{"1": "PASS", "2": "FAIL"}}"""

WRITER = "deepseek-ai/DeepSeek-V3.2"
JUDGE = "openai/gpt-oss-120b"  # a different family from the writer, and a reasoning model


def parse_json(text: str):
    m = re.search(r"[\[{].*[\]}]", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def check_design(text: str):
    text = strip_fences(text).strip()
    try:
        doc = parse(text)
    except Exception:  # noqa: BLE001
        return None, "parse error"
    nodes = [n for n, _ in walk(doc)]
    if not 12 <= len(nodes) <= 60:
        return None, f"{len(nodes)} nodes"
    if len({n["id"] for n in nodes}) != len(nodes):
        return None, "duplicate ids"
    empty = {"id": "n0", "type": "frame", "props": {}, "children": []}
    bad = dsl_violations(empty, doc)
    return (None, bad[0]) if bad else (doc, "")


def grade(ds_text: str, before, request: str, criteria: list[str], after, model: str = JUDGE) -> list[bool] | None:
    from design_edit.goal_reward import describe_changes

    msg = GRADE_PROMPT.format(ds=ds_text, before=serialize(before), request=request,
                              changes=describe_changes(before, after) or "(no changes)", after=serialize(after),
                              criteria="\n".join(f"{i + 1}. {c}" for i, c in enumerate(criteria)))
    out = parse_json(chat(model, [{"role": "user", "content": msg}], max_tokens=6000))
    if not isinstance(out, dict):
        return None
    return [str(out.get(str(i + 1), "")).upper().startswith("PASS") for i in range(len(criteria))]


def build_one(i: int, token: str, seed_base: int = 9_000, k: int = 3, empty_check: bool = True) -> list[dict]:
    rng = random.Random(seed_base + i)
    screen = SCREENS[i % len(SCREENS)]
    ds = sample_design_system(seed_base + i)
    ds_text = render_design_system(ds)
    doc = None
    for attempt in range(3):
        out = chat(WRITER, [{"role": "user", "content": DESIGN_PROMPT.format(system=SYSTEM_PROMPT, screen=screen,
                                                                             ds=ds_text)}],
                   max_tokens=4000, temperature=0.7, token=token)
        doc, why = check_design(out)
        if doc is not None:
            break
    if doc is None:
        print(f"design {i}: rejected ({why})", flush=True)
        return []
    kinds = [rng.choice(list(KINDS)) for _ in range(k)] if k > len(KINDS) else rng.sample(list(KINDS), k)
    tasks = parse_json(chat(WRITER, [{"role": "user", "content": TASK_PROMPT.format(
        system=SYSTEM_PROMPT, ds=ds_text, design=serialize(doc), k=len(kinds),
        kinds="\n".join(f"- {k}: {KINDS[k]}" for k in kinds))}], max_tokens=3000, temperature=0.7, token=token))
    rows = []
    stats = {"ref_failed": 0, "empty_passed": 0, "invalid": 0}
    for t in tasks if isinstance(tasks, list) else []:
        try:
            req, crit, patch = t["request"], [str(c) for c in t["criteria"]], strip_fences(t["patch"])
            after = apply_patch(doc, patch)
        except (KeyError, TypeError, PatchError, ValueError):
            stats["invalid"] += 1
            continue
        if not 2 <= len(crit) <= 6 or dsl_violations(doc, after) or serialize(after) == serialize(doc):
            stats["invalid"] += 1
            continue
        verdict = grade(ds_text, doc, req, crit, after)
        if not verdict or not all(verdict):
            stats["ref_failed"] += 1
            continue  # the writer's own answer must pass its criteria
        empty = grade(ds_text, doc, req, crit, doc) if empty_check else [False]
        if not empty or all(empty):
            stats["empty_passed"] += 1
            continue  # ...and the unedited design must fail them
        rows.append({
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": user_message_ds(serialize(doc), ds_text, req)}],
            "completion": [{"role": "assistant", "content": patch}],
            "before": serialize(doc), "after": serialize(after), "kind": f"v2_{t.get('kind', 'other')}",
            "instruction": req, "orig_instruction": req, "relax": "[]", "alts": "[]", "refs": "[]",
            "criteria": json.dumps(crit), "design_system": ds_text, "screen": screen,
        })
    print(f"design {i} ({screen}): {len(rows)} items {stats}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-designs", type=int, default=110)
    ap.add_argument("--out", required=True)
    ap.add_argument("--push", default=None, help="push as the --split split of this dataset repo")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed-base", type=int, default=9_000, help="eval v2 uses 9000+; training data must use others")
    ap.add_argument("--k", type=int, default=3, help="requests per design")
    ap.add_argument("--no-empty-check", action="store_true", help="skip the unedited-design-must-fail check")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    token = os.environ["HF_TOKEN"]
    # Resumable: each design's items are appended as soon as it finishes, and finished designs are skipped.
    import threading

    done_path = Path(args.out + ".done")
    done = {int(x) for x in done_path.read_text().split()} if done_path.exists() else set()
    lock = threading.Lock()

    def run(i: int) -> None:
        rs = build_one(i, token, args.seed_base, args.k, not args.no_empty_check)
        with lock, open(args.out, "a") as f, open(done_path, "a") as d:
            f.write("".join(json.dumps(r) + "\n" for r in rs))
            d.write(f"{i}\n")

    if not done:
        Path(args.out).write_text("")
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(run, [i for i in range(args.n_designs) if i not in done]))
    rows = [json.loads(line) for line in open(args.out)]
    print(f"{len(rows)} items from {args.n_designs} designs -> {args.out}")
    if args.push:
        from datasets import Dataset, DatasetDict
        from huggingface_hub import HfApi

        DatasetDict({args.split: Dataset.from_list(rows)}).push_to_hub(args.push, private=True)
        root = Path(__file__).resolve().parents[1]
        for d in ("design_edit", "scripts"):
            HfApi().upload_folder(repo_id=args.push, repo_type="dataset", folder_path=str(root / d),
                                  path_in_repo=f"code/{d}", ignore_patterns=["__pycache__/*", "*.pyc"])
        print("pushed", args.push)


if __name__ == "__main__":
    main()
