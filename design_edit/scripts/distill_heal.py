#!/usr/bin/env python3
"""Self-healing teacher distillation, following Figma's described pipeline.

For each generated design: a proposer writes an edit request; a teacher model attempts the edit; the
candidate goes through deterministic validators (parses, applies, changes something, no new lint, no new
rendered-geometry issues: clipping / overlap / overflow) and a visual judge (before/after screenshots,
graded on fulfilled / preserved / legible / contained). On failure the teacher gets the specific
problems back and revises, up to --max-attempts. Only candidates that pass everything are kept.

    python scripts/distill_heal.py --n 2500 --out data/healed.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from distill_goals import propose  # noqa: E402  (same proposer + held-out filters)

from design_edit.generator import generate_design  # noqa: E402
from design_edit.geometry import Renderer  # noqa: E402
from design_edit.judge import chat  # noqa: E402
from design_edit.prompts import prompt_messages  # noqa: E402
from design_edit.rewards import lint  # noqa: E402
from design_edit.schema import PatchError, apply_patch, diff, format_patch, serialize, strip_fences  # noqa: E402
from design_edit.visual_judge import DIMENSIONS, judge  # noqa: E402

_local = threading.local()


def renderer() -> Renderer:
    if not hasattr(_local, "r"):
        _local.r = Renderer().__enter__()
    return _local.r


def check(doc, patch: str, request: str, judge_model: str) -> tuple[str | None, list[str], dict]:
    """Return (canonical patch or None, list of problems, info)."""
    try:
        after = apply_patch(doc, patch)
    except (PatchError, ValueError, KeyError, IndexError) as e:
        return None, [f"the patch does not apply: {e}"], {"stage": "apply"}
    canon = format_patch(diff(doc, after)) if after["id"] == doc["id"] else ""
    if not canon:
        return None, ["the patch changes nothing"], {"stage": "empty"}
    problems = []
    new_lint = sorted(set(lint(after)) - set(lint(doc)))
    if new_lint:
        problems.append("it introduces design-rule violations: " + ", ".join(new_lint))
    r = renderer()
    new_geo = sorted(r.new_issues(doc, after))
    if new_geo:
        problems.append("the rendered layout has new problems: " + ", ".join(new_geo)
                        + " (clip = child exceeds its container, overlap = siblings overlap, "
                          "text_overflow = text wider than its box, offscreen = outside the page)")
    if problems:
        return canon, problems, {"stage": "validators"}
    v = judge(r.screenshot(doc), r.screenshot(after), request, canon, judge_model)
    if v is None:
        return canon, ["(visual judge unavailable)"], {"stage": "judge_error"}
    failed = [d for d in DIMENSIONS if not v[d]]
    if failed:
        return canon, [f"a design reviewer looking at the rendered result says it is not {', '.join(failed)}: "
                       f"{v.get('reason', '')}"], {"stage": "visual", "failed": failed}
    return canon, [], {"stage": "pass"}


def one(seed: int, args, token: str) -> tuple[str, int, dict | None]:
    rng = random.Random(seed)
    doc = generate_design(rng.randint(4_000_000, 4_999_999), max_sections=rng.choice([1, 2, 2, 3]))
    text = serialize(doc)
    req = propose(text, rng, args.proposer, token, args.base_url)
    if req is None:
        return "rejected_request", 0, None
    messages = prompt_messages(doc, req)
    for attempt in range(1, args.max_attempts + 1):
        out = chat(args.teacher, messages, max_tokens=8192, temperature=0.6 if attempt > 1 else 0.0, token=token)
        patch = strip_fences(out).strip()
        if not patch:
            return "no_patch", attempt, None
        canon, problems, info = check(doc, patch, req, args.judge)
        if info["stage"] == "judge_error":
            return "judge_error", attempt, None
        if not problems:
            return f"kept_after_{attempt}", attempt, {
                "prompt": prompt_messages(doc, req), "completion": [{"role": "assistant", "content": canon}],
                "before": text, "after": serialize(apply_patch(doc, canon)), "relax": "[]",
                "kind": "healed_goal", "instruction": req, "orig_instruction": req, "alts": "[]"}
        messages = messages + [
            {"role": "assistant", "content": patch},
            {"role": "user", "content": "That patch has problems:\n- " + "\n- ".join(problems)
                                        + "\nReply with a corrected patch for the ORIGINAL design (not a diff on top "
                                          "of your previous attempt), and nothing else."}]
    return f"failed_{info['stage']}", args.max_attempts, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--proposer", default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    ap.add_argument("--teacher", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--judge", default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
    ap.add_argument("--base-url", default="https://router.huggingface.co/v1")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    token = os.environ["HF_TOKEN"]
    stats: Counter = Counter()
    kept = 0
    with ThreadPoolExecutor(args.workers) as pool, open(args.out, "a") as f:
        seeds = range(args.seed * 1_000_000, args.seed * 1_000_000 + args.n)
        for i, (status, attempts, row) in enumerate(pool.map(lambda s: one(s, args, token), seeds)):
            stats[status] += 1
            if row is not None:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                kept += 1
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{args.n} kept={kept} {dict(stats)}", flush=True)
    print(f"done: kept {kept}/{args.n} -> {args.out}\n{dict(stats)}")


if __name__ == "__main__":
    main()
