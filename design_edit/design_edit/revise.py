"""Review turn for propose -> apply -> review -> revise (the text analogue of CADGenBench's render-and-review loop).

After the model answers, the harness applies the patch and shows the outcome: every change in plain words, the
resulting design, or the error if the patch didn't apply. The model then keeps or corrects its answer.
"""

from __future__ import annotations

import re

from .schema import PatchError, apply_patch, parse, serialize, strip_fences

REVIEW = """I applied your patch. {outcome}

Check the result against the request{ds}. If it fully and minimally does what was asked, reply with the same patch.
Otherwise reply with a corrected complete patch (relative to the ORIGINAL design). Reply with the patch only."""


def _design_of(prompt: list[dict]) -> str | None:
    m = re.search(r"Design:\n(.*?)\n\nRequest: ", prompt[-1]["content"], re.S)
    return m.group(1) if m else None


def review_message(prompt: list[dict], output: str) -> str:
    from .goal_reward import describe_changes

    text = _design_of(prompt)
    ds = " and the design system" if "Design system:" in prompt[-1]["content"] else ""
    patch = strip_fences(output).strip()
    if text is None:
        return REVIEW.format(outcome="", ds=ds)
    before = parse(text)
    if not patch:
        return REVIEW.format(outcome="Your reply contained no patch, so nothing changed.", ds=ds)
    try:
        after = apply_patch(before, patch)
    except (PatchError, ValueError, KeyError, IndexError) as e:
        return REVIEW.format(outcome=f"It failed to apply: {e}. Nothing changed.", ds=ds)
    changes = describe_changes(before, after) or "(no changes)"
    return REVIEW.format(outcome=f"Changes it made:\n{changes}\n\nResulting design:\n{serialize(after)}", ds=ds)
