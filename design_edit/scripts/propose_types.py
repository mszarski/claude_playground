#!/usr/bin/env python3
"""Ask a strong LLM for a broad taxonomy of edit types this design language can express.

Output: a JSON list of {"name", "category", "description", "examples": [request, ...]}. Each type later seeds a
few dozen teacher-written rows (distill_heal.py --types), so the student sees many conventions with a few
examples each, instead of a few conventions with hundreds of examples each.

Types that resemble a held-out test type (edits.NOVEL / edits.GOAL) are dropped by the same keyword filter as
the request proposer; patches are additionally checked on their effect (design_edit.leak) at distillation time.

    python scripts/propose_types.py --out data/edit_types.json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from distill_goals import HELD_OUT, PLAN_EMPH  # noqa: E402

from design_edit.generator import generate_design  # noqa: E402
from design_edit.judge import chat  # noqa: E402
from design_edit.prompts import SYSTEM_PROMPT  # noqa: E402
from design_edit.schema import serialize  # noqa: E402

CATEGORIES = [
    "typography (size, line height, letter case, hierarchy between texts)",
    "color and contrast (text, fills, brand usage, neutrals)",
    "spacing and rhythm (gap, padding, consistency between siblings)",
    "auto-layout and alignment (direction, align, justify, wrapping into rows/columns)",
    "content edits (rewording, shortening, adding or removing a label, placeholder, caption or helper text)",
    "structure (moving, reordering, grouping, ungrouping, deleting nodes, inserting new nodes)",
    "component styling (cards, inputs, navbars, lists, badges, avatars, images)",
    "consistency and cleanup (matching one element to another, removing redundant properties, design-system fixes)",
    "visual weight and emphasis (de-emphasizing, softening, quieting, simplifying)",
    "multi-element edits (apply one change to every element of a kind, swap two elements' styles)",
]

PROMPT = """A design language for UI mockups is described by this system prompt:

---
{system}
---

Example design:
{design}

List {k} distinct, concrete EDIT TYPES in the category: {category}.
An edit type is a reusable kind of change (e.g. "tighten line spacing of body text", "move a helper text above its
field"), achievable with the patch operations above on designs like the example (cards, forms, navbars, pricing
tables, settings lists). Make the types varied and specific; avoid near-duplicates.

Do NOT include anything about: disabled/inactive states, highlighting or featuring an element, toggles or switches,
borders or outlines, widths, bold text, primary/call-to-action button styling, spreading items apart or
space-between, darkening text, or duplicating/copying elements.

Reply with a JSON list only: [{{"name": "snake_case_name", "description": "one sentence", "examples": ["request 1",
"request 2", "request 3"]}}, ...]"""


def parse_json_list(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [t for t in out if isinstance(t, dict) and t.get("name") and isinstance(t.get("examples"), list)]


def leaks(t: dict) -> bool:
    text = " ".join([t["name"].replace("_", " "), t.get("description", ""), *map(str, t["examples"])])
    return bool(HELD_OUT.search(text) or PLAN_EMPH.search(text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    ap.add_argument("--per-category", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=2, help="calls per category (different example designs)")
    args = ap.parse_args()
    token = os.environ["HF_TOKEN"]

    jobs = [(c, r) for c in CATEGORIES for r in range(args.rounds)]

    def run(job):
        cat, r = job
        doc = generate_design(5_000_000 + 7 * CATEGORIES.index(cat) + r, max_sections=3)
        msg = PROMPT.format(system=SYSTEM_PROMPT, design=serialize(doc), k=args.per_category, category=cat)
        types = parse_json_list(chat(args.model, [{"role": "user", "content": msg}], max_tokens=6000,
                                     temperature=0.8, token=token))
        return [{**t, "category": cat.split(" (")[0]} for t in types]

    with ThreadPoolExecutor(8) as pool:
        proposed = [t for ts in pool.map(run, jobs) for t in ts]
    kept, seen, dropped = [], set(), 0
    for t in proposed:
        key = re.sub(r"[^a-z]", "", t["name"].lower())
        if key in seen:
            continue
        seen.add(key)
        if leaks(t):
            dropped += 1
            continue
        kept.append(t)
    Path(args.out).write_text(json.dumps(kept, indent=1, ensure_ascii=False))
    print(f"proposed {len(proposed)}, unique {len(seen)}, dropped as held-out-like {dropped}, kept {len(kept)}")


if __name__ == "__main__":
    main()
