"""Scoring a predicted patch against a reference edit.

``score`` compares the *effects* of the predicted patch with the effects of the
reference, not the patch text. Two patches that produce the same design score
the same, and a vague request ("bigger") accepts any value that moves the
right way (see ``relax``). The main parts:

* ``valid``  - the patch parses and applies.
* ``f1``     - F1 over change facts (prop changes, deletes, inserts, reorders).
* ``exact``  - the resulting design matches the reference exactly (allowing relaxed props).
* ``lint``   - design-rule violations the patch adds beyond the reference
               (low contrast, off-scale spacing or type).

``reward`` folds these into one scalar for RL.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from . import generator as g
from .schema import Node, PatchError, apply_patch, index_nodes, parse, serialize, walk

Fact = tuple


# ---------------------------------------------------------------------------
# Change facts
# ---------------------------------------------------------------------------

def _strip_ids(node: Node) -> str:
    """Serialized subtree with ids removed: identifies inserted content regardless of the ids the model picked."""
    return "\n".join(re.sub(r"^(\s*\w+)#\S+", r"\1", line) for line in serialize(node).splitlines())


def change_facts(before: Node, after: Node) -> set[Fact]:
    b_idx, a_idx = index_nodes(before), index_nodes(after)
    facts: set[Fact] = set()
    for nid, (node, parent) in b_idx.items():
        if nid not in a_idx and (parent is None or parent["id"] in a_idx):
            facts.add(("del", nid))
    for nid, (node, parent) in a_idx.items():
        if nid not in b_idx:
            if parent is not None and parent["id"] in b_idx:
                # Position = the nearest pre-existing sibling before it (robust to the model's choice of ids).
                prev = None
                for c in parent["children"]:
                    if c["id"] == nid:
                        break
                    if c["id"] in b_idx:
                        prev = c["id"]
                facts.add(("ins", parent["id"], prev, _strip_ids(node)))
            continue
        bnode, bparent = b_idx[nid]
        for k in set(bnode["props"]) | set(node["props"]):
            if bnode["props"].get(k) != node["props"].get(k):
                facts.add(("prop", nid, k, node["props"].get(k)))
        if (bparent and bparent["id"]) != (parent and parent["id"]):
            facts.add(("parent", nid, parent and parent["id"]))
    # Order changes among surviving siblings.
    for nid, (node, _) in a_idx.items():
        if nid in b_idx:
            kept_after = [c["id"] for c in node["children"] if c["id"] in b_idx]
            kept_before = [c["id"] for c in b_idx[nid][0]["children"] if c["id"] in a_idx]
            common = [c for c in kept_after if c in kept_before]
            if common != [c for c in kept_before if c in kept_after]:
                facts.add(("order", nid, tuple(common)))
    return facts


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _relaxed_match(gold: Fact, pred: Fact, before: Node, relax: list[dict[str, Any]]) -> bool:
    if gold == pred:
        return True
    if gold[0] != "prop" or pred[0] != "prop" or gold[1:3] != pred[1:3]:
        return False
    for r in relax:
        if r["id"] == gold[1] and r["prop"] == gold[2]:
            return relaxed_ok(r, index_nodes(before)[gold[1]][0]["props"].get(gold[2]), pred[3])
    return False


def _hsv(hexv: str) -> tuple[float, float, float] | None:
    import colorsys

    h = str(hexv).lstrip("#")
    if len(h) != 6:
        return None
    try:
        return colorsys.rgb_to_hsv(*(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)))
    except ValueError:
        return None


def relaxed_ok(r: dict[str, Any], old: Any, new: Any) -> bool:
    """Does ``new`` satisfy a relaxed requirement? ``cmp`` kinds:

    gt / lt   numeric value moved up / down          ("make it bigger")
    any       any value                              ("change the copy")
    allowed   value in r["values"]                   ("snap to the type scale": either neighbor is fine)
    contrast  contrast vs r["bg"] >= r["min"]        ("fix the contrast": any readable color)
    hue       same color family as r["target"]       ("make it yellow": any yellow)
    darker / lighter  lower / higher luminance than before  ("make the body text darker")
    """
    cmp = r["cmp"]
    if new is None:
        return False
    if cmp == "any":
        return True
    if cmp == "allowed":
        return new in r["values"]
    if cmp == "contrast":
        return isinstance(new, str) and _hsv(new) is not None and contrast(new, r["bg"]) >= r["min"]
    if cmp == "hue":
        a, b = _hsv(new), _hsv(r["target"])
        if a is None or b is None:
            return False
        if b[1] < 0.15:  # target is a neutral (black/gray): accept low-saturation colors of similar value
            return a[1] < 0.2 and abs(a[2] - b[2]) < 0.35
        dh = min(abs(a[0] - b[0]), 1 - abs(a[0] - b[0])) * 360
        return a[1] >= 0.3 and dh <= 25
    if cmp in ("darker", "lighter"):
        if _hsv(new) is None or _hsv(old) is None:
            return False
        return (_luminance(new) < _luminance(old)) == (cmp == "darker")
    new_n, old_n = _num(new), _num(old)
    if new_n is None or old_n is None:
        return False
    return new_n > old_n if cmp == "gt" else new_n < old_n


def fact_f1(gold: set[Fact], pred: set[Fact], before: Node, relax: list[dict[str, Any]]) -> tuple[float, float, float]:
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    matched_pred: set[Fact] = set()
    tp = 0
    for gf in gold:
        hit = next((pf for pf in pred if pf not in matched_pred and _relaxed_match(gf, pf, before, relax)), None)
        if hit is not None:
            matched_pred.add(hit)
            tp += 1
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Design lint
# ---------------------------------------------------------------------------

def _luminance(hexv: str) -> float:
    h = hexv.lstrip("#")
    if len(h) != 6:
        return 1.0
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(fg: str, bg: str) -> float:
    a, b = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def lint(doc: Node) -> list[str]:
    """Design-rule violations: things a designer would flag in review."""
    issues: list[str] = []

    def rec(node: Node, bg: str) -> None:
        p = node["props"]
        fill = p.get("fill", bg) if node["type"] == "frame" else bg
        if node["type"] == "text":
            if isinstance(p.get("color"), str) and contrast(p["color"], bg) < 3.0:
                issues.append(f"contrast:{node['id']}")
            if p.get("size") not in g.TYPE_SCALE:
                issues.append(f"type_scale:{node['id']}")
        for key in ("gap", "pad"):
            v = p.get(key)
            if isinstance(v, (int, float)) and v % 2 != 0:
                issues.append(f"spacing:{node['id']}:{key}")
        for c in node["children"]:
            rec(c, fill if isinstance(fill, str) else bg)

    rec(doc, "#FFFFFF")
    return issues


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class Score:
    valid: bool
    precision: float
    recall: float
    f1: float
    exact: bool
    new_lint: int
    reward: float
    error: str | None = None


def score(before: Node | str, after: Node | str, patch_text: str,
          relax: list[dict[str, Any]] | str | None = None,
          alts: list[dict[str, Any]] | str | None = None) -> Score:
    """Score a patch against the reference edit, or the best-matching alternative.

    ``alts`` lists other equally valid answers, each ``{"after": doc, "relax": [...]}``
    (e.g. "make it look disabled" = lower opacity OR grey fill + grey label).
    """
    before = parse(before) if isinstance(before, str) else before
    relax = json.loads(relax) if isinstance(relax, str) else (relax or [])
    alts = json.loads(alts) if isinstance(alts, str) else (alts or [])
    try:
        pred_doc = apply_patch(before, patch_text)
        pred_facts = change_facts(before, pred_doc)
    except (PatchError, ValueError, KeyError, IndexError) as e:
        return Score(False, 0.0, 0.0, 0.0, False, 0, -1.0, str(e))
    best: Score | None = None
    for ref, ref_relax in [(after, relax)] + [(a["after"], a.get("relax", [])) for a in alts]:
        ref = parse(ref) if isinstance(ref, str) else ref
        gold = change_facts(before, ref)
        p, r, f1 = fact_f1(gold, pred_facts, before, ref_relax)
        exact = f1 == 1.0 and len(pred_facts) == len(gold)
        new_lint = max(0, len(set(lint(pred_doc)) - set(lint(ref))))
        cand = Score(True, p, r, f1, exact, new_lint, f1 + (0.5 if exact else 0.0) - 0.25 * new_lint)
        if best is None or cand.reward > best.reward:
            best = cand
    return best


def score_dict(*args: Any, **kw: Any) -> dict[str, Any]:
    return asdict(score(*args, **kw))


# ---------------------------------------------------------------------------
# Placement: verify spatial instructions ("below X", "above X") structurally
# ---------------------------------------------------------------------------

# Longest alternatives first, whole words only ("under" must not match inside "underneath").
_AFTER = r"\b(?:underneath|beneath|below|under|following|after|to the right of|right of|next to)\b"
_BEFORE = r"\b(?:on top of|to the left of|left of|ahead of|above|before)\b"


def _mentions(request: str, node: Node) -> list[int]:
    """Positions in the request where this node is referred to (by name or text)."""
    low = request.lower()
    out = []
    for key in (node["props"].get("name"), node["props"].get("text")):
        if isinstance(key, str) and len(key) >= 3:
            k = key.lower()
            for cand in {k, k.removesuffix(" button"), k.removesuffix(" row"), k.removesuffix(" field")}:
                if len(cand) >= 3:
                    i = low.find(cand)
                    while i != -1:
                        out.append(i)
                        i = low.find(cand, i + 1)
    return out


def _lis_keep(seq: list[int]) -> set[int]:
    """Indices (into seq) of one longest increasing subsequence: the elements that did NOT move."""
    import bisect
    tails, tails_idx, prev = [], [], [-1] * len(seq)
    for i, v in enumerate(seq):
        j = bisect.bisect_left(tails, v)
        if j == len(tails):
            tails.append(v); tails_idx.append(i)
        else:
            tails[j] = v; tails_idx[j] = i
        prev[i] = tails_idx[j - 1] if j > 0 else -1
    keep, k = set(), tails_idx[-1] if tails_idx else -1
    while k != -1:
        keep.add(k); k = prev[k]
    return keep


def placement_violations(request: str, before: Node, after: Node) -> list[str]:
    """Inserted or moved nodes that ignore a spatial cue in the request.

    "Placed" nodes are new (top of an inserted subtree), re-parented, or reordered relative to their original
    siblings (outside the longest still-in-order run, so siblings merely shifted by an insert don't count).
    For each spatial cue ("below X", "above X", "underneath it", ...) the placed node must sit directly after
    (or before) X. Returns violation descriptions; empty when there are no cues or no placed nodes.
    """
    b_idx, a_idx = index_nodes(before), index_nodes(after)
    placed = []
    for nid, (node, parent) in a_idx.items():
        if parent is not None and nid not in b_idx and parent["id"] in b_idx:
            placed.append((node, parent))
        elif parent is not None and nid in b_idx and b_idx[nid][1] is not None and b_idx[nid][1]["id"] != parent["id"]:
            placed.append((node, parent))
    for pid, (pnode, _) in a_idx.items():
        if pid not in b_idx:
            continue
        common = [c["id"] for c in pnode["children"] if c["id"] in b_idx and b_idx[c["id"]][1] is not None
                  and b_idx[c["id"]][1]["id"] == pid]
        old_order = [c["id"] for c in b_idx[pid][0]["children"]]
        seq = [old_order.index(c) for c in common]
        keep = _lis_keep(seq)
        placed += [(a_idx[c][0], pnode) for i, c in enumerate(common) if i not in keep]
    if not placed:
        return []
    low = request.lower()
    # Cues only apply within their own clause ("Put A above B. Also, add C under D" = two clauses).
    bounds = [0] + [m.end() for m in re.finditer(r"[.;\n]|,? and then |,? then |\balso\b,?", low)] + [len(low)]
    clauses = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]
    mentions = sorted((pos, nid) for nid, (n, _) in b_idx.items() for pos in _mentions(request, n))
    out = []
    for node, parent in placed:
        sibs = [c["id"] for c in parent["children"]]
        pos = sibs.index(node["id"])
        is_new = node["id"] not in b_idx
        names = _mentions(request, node) if not is_new else []
        for c0, c1 in clauses:
            clause = low[c0:c1]
            # the clause must be about this node: it names it, or (for new nodes) it inserts something
            if not (any(c0 <= m < c1 for m in names) or
                    (is_new and re.search(r"\b(add|insert|duplicate|copy|create)\b", clause))):
                continue
            cues = [(c0 + m.start(), c0 + m.end(), "after") for m in re.finditer(_AFTER, clause)] + \
                   [(c0 + m.start(), c0 + m.end(), "before") for m in re.finditer(_BEFORE, clause)]
            for cstart, cend, rel in cues:
                if low[cend:cend + 6].strip().startswith(("it", "that", "them")):
                    refs = [nid for mpos, nid in mentions if c0 <= mpos < cstart and nid in sibs and nid != node["id"]][-1:]
                else:
                    refs = [nid for mpos, nid in mentions
                            if 0 <= mpos - cend <= 40 and mpos < c1 and nid in sibs and nid != node["id"]][:1]
                for ref_id in refs:
                    rpos = sibs.index(ref_id)
                    if not (pos == rpos + 1 if rel == "after" else pos == rpos - 1):
                        out.append(f"{node['id']} should be {rel} {ref_id}")
    return sorted(set(out))
