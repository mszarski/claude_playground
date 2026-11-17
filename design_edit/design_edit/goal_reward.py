"""Label-free reward for goal-level edit requests, shared by RL training and the reward audit.

version 1 (what the RL runs so far used):
    -1 invalid patch / 0 empty patch; otherwise the visual judge's dimensions (gated on "fulfilled")
    minus penalties: new lint, new rendered-geometry issues, edit size, placement violations.

version 2 (closes blind spots as a class instead of one at a time):
    * the judge also gets a plain-language list of EVERY change, including ones a screenshot can't show
      (removed properties, reordering, sub-pixel borders), plus 2x close-ups of the edited area, and is told that
      subtle-but-correct changes count. The v1 judge rejected most correct "darken", "bold" and "spread out"
      edits as "nothing changed", which rewards conspicuous over-edits;
    * optional teacher references: answers from a stronger reasoning model for the same request. Half of the
      reward is the F1 between the prediction's changes and the closest reference, so *omitted* changes
      (e.g. skipping `unset stroke`) cost reward even though no screenshot shows them.
"""

from __future__ import annotations

from typing import Any

from .rewards import change_facts, fact_f1, lint, placement_violations
from .schema import Node, PatchError, apply_patch, diff, format_patch, index_nodes, parse
from .visual_judge import DIMENSIONS, judge as visual_judge


def describe_changes(before: Node, after: Node) -> str:
    """Every change as one plain line, visible or not."""
    b_idx, a_idx = index_nodes(before), index_nodes(after)

    def name(nid: str) -> str:
        n = (a_idx.get(nid) or b_idx.get(nid))[0]
        return f'{nid} "{n["props"].get("name", n["type"])}"'

    lines = []
    for f in sorted(change_facts(before, after), key=str):
        if f[0] == "prop":
            old = b_idx[f[1]][0]["props"].get(f[2])
            lines.append(f"{name(f[1])}: {f[2]} {old!r} -> {f[3]!r}" if f[3] is not None
                         else f"{name(f[1])}: {f[2]} {old!r} REMOVED")
        elif f[0] == "del":
            lines.append(f"{name(f[1])}: deleted")
        elif f[0] == "ins":
            where = f"after {name(f[2])}" if f[2] else "as the first child"
            lines.append(f"new node inserted in {name(f[1])} {where}: {f[3].splitlines()[0][:120]}")
        elif f[0] == "parent":
            lines.append(f"{name(f[1])}: moved into {name(f[2])}")
        elif f[0] == "order":
            lines.append(f"children of {name(f[1])} reordered to: {', '.join(f[2])}")
    return "\n".join(lines)


def focus_node(before: Node, after: Node) -> str | None:
    """The smallest node (present in both designs) that contains every change; None if that is the whole page."""
    b_idx, a_idx = index_nodes(before), index_nodes(after)
    touched = set()
    for f in change_facts(before, after):
        if f[0] in ("prop", "order", "parent"):
            touched.add(f[1])
        elif f[0] == "ins":
            touched.add(f[1])
        elif f[0] == "del":
            parent = b_idx[f[1]][1]
            if parent is not None:
                touched.add(parent["id"])
    touched = {t for t in touched if t in a_idx and t in b_idx}
    if not touched:
        return None

    def chain(nid: str) -> list[str]:
        out = [nid]
        while a_idx[out[-1]][1] is not None:
            out.append(a_idx[out[-1]][1]["id"])
        return out  # node ... root

    chains = [chain(t) for t in touched]
    common = next((c for c in chains[0] if all(c in ch for ch in chains[1:])), None)
    if common is None or common not in b_idx:
        return None
    # a lone leaf (e.g. one text) needs its container for context
    if not a_idx[common][0]["children"] and a_idx[common][1] is not None:
        common = a_idx[common][1]["id"]
    return None if a_idx[common][1] is None or common not in b_idx else common


def goal_reward(before_text: str, instruction: str, completion: str, *, judge_model: str, renderer,
                free_changes: int = 3, version: int = 1, refs: list[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"reward": 0.0, "valid": False, "verdict": None, "penalty": 0.0, "ref_f1": None,
                           "judge_reachable": True}
    doc = parse(before_text)
    try:
        after_doc = apply_patch(doc, completion)
    except (PatchError, ValueError, KeyError, IndexError):
        out["reward"] = -1.0
        return out
    out["valid"] = True
    canon = format_patch(diff(doc, after_doc)) if after_doc["id"] == doc["id"] else ""
    if not canon:
        return out
    penalty = 0.25 * (len(set(lint(after_doc)) - set(lint(doc))) + len(renderer.new_issues(doc, after_doc)))
    # Edit-size penalty: goal edits rarely need more than a few property changes.
    penalty += 0.1 * max(0, len(change_facts(doc, after_doc)) - free_changes)
    # Structural placement: "below X" / "above X" / "underneath it" must be honoured exactly.
    penalty += min(1.0, 0.5 * len(placement_violations(instruction, doc, after_doc)))
    out["penalty"] = penalty

    patch_for_judge = canon
    if version >= 2:
        patch_for_judge = (canon + "\n\nEvery change this patch makes, including ones the screenshots cannot show:\n"
                           + describe_changes(doc, after_doc))
    closeups = None
    if version >= 2:
        fid = focus_node(doc, after_doc)
        if fid:
            c0, c1 = renderer.crop(doc, fid), renderer.crop(after_doc, fid)
            closeups = (c0, c1) if c0 and c1 else None
    v = visual_judge(renderer.screenshot(doc), renderer.screenshot(after_doc), instruction, patch_for_judge, judge_model,
                     closeups=closeups)
    out["verdict"] = v
    if v is None:
        out["judge_reachable"] = False
        out["reward"] = -penalty
        return out
    # Fulfilment gates the rest: a change that ignores the request earns nothing for being tidy.
    judge_part = (sum(bool(v[d]) for d in DIMENSIONS) / len(DIMENSIONS)) if v["fulfilled"] else 0.0

    if version >= 2 and refs:
        pred = change_facts(doc, after_doc)
        best = 0.0
        for ref in refs:
            gold = change_facts(doc, parse(ref))
            if gold:
                best = max(best, fact_f1(gold, pred, doc, [])[2])
        out["ref_f1"] = best
        out["reward"] = 0.5 * judge_part + 0.5 * best - penalty
    else:
        out["reward"] = judge_part - penalty
    return out
