"""Deterministic grading of an edit against machine-checkable criteria (ideas borrowed from CADGenBench).

An item carries
  * ``checks``: must-hold conditions on the edited design ("keep-in"), e.g. node n41 has opacity 0.5, a new text
    node containing "required" sits in n21, n5 comes before n7. Subjective criteria stay as ``{"judge": text}``
    and are only graded by an LLM (eval), never in the RL reward.
  * ``allowed``: node ids the edit may touch ("keep-out" is everything else: those nodes must keep their props,
    children order and existence). New nodes are always allowed.

Score (CADGenBench editing composition, adapted):
  validity gate -> 0 if the patch doesn't apply or uses properties the language lacks;
  raw = 0.6 * keep_in + 0.3 * keep_out + 0.1 * no_new_lint;
  the no-op (unedited design) scores keep_in < 1 by construction but keep_out = 1, so raw is renormalized against
  it: score = max(0, (raw - raw_noop) / (1 - raw_noop)), mapping "do nothing" to 0 and a perfect edit to 1;
  pass = every keep-in check holds (an item is as good as its worst check) and keep_out == 1.
"""

from __future__ import annotations

import json
from typing import Any

from .leak import dsl_violations
from .rewards import lint
from .schema import Node, PatchError, apply_patch, index_nodes, parse, walk

_OPS = ("eq", "ne", "in", "lt", "gt", "le", "ge", "contains", "changed", "unchanged", "absent", "present")


def _norm(v: Any) -> Any:
    return v.upper() if isinstance(v, str) and v.startswith("#") else v


def _cmp(val: Any, spec: dict[str, Any], old: Any = None) -> bool:
    for op, want in spec.items():
        if op not in _OPS:
            continue
        v, w = _norm(val), _norm(want)
        try:
            ok = {"eq": lambda: v == w or (isinstance(v, (int, float)) and isinstance(w, (int, float))
                                           and abs(v - w) < 1e-6),
                  "ne": lambda: v != w,
                  "in": lambda: v in [_norm(x) for x in w],
                  "lt": lambda: v is not None and v < w, "gt": lambda: v is not None and v > w,
                  "le": lambda: v is not None and v <= w, "ge": lambda: v is not None and v >= w,
                  "contains": lambda: isinstance(v, str) and str(w).lower() in v.lower(),
                  "changed": lambda: (v != _norm(old)) == bool(w), "unchanged": lambda: (v == _norm(old)) == bool(w),
                  "absent": lambda: (v is None) == bool(w), "present": lambda: (v is not None) == bool(w)}[op]()
        except TypeError:
            ok = False
        if not ok:
            return False
    return True


def _node_matches(n: Node, spec: dict[str, Any]) -> bool:
    if spec.get("type") and n["type"] != spec["type"]:
        return False
    if spec.get("text_contains") and spec["text_contains"].lower() not in str(n["props"].get("text", "")).lower():
        return False
    if spec.get("name_contains") and spec["name_contains"].lower() not in str(n["props"].get("name", "")).lower():
        return False
    for k, cond in (spec.get("props") or {}).items():
        cond = cond if isinstance(cond, dict) else {"eq": cond}
        if not _cmp(n["props"].get(k), cond):
            return False
    return True


def eval_check(c: dict[str, Any], before: Node, after: Node) -> bool | None:
    """True/False for a deterministic check, None for a judge-only check or one this grader can't read.

    A malformed check (wrong value types) counts as failed, so compile_checks.py rejects it."""
    try:
        return _eval_check(c, before, after)
    except (TypeError, KeyError, AttributeError, ValueError, StopIteration):
        return False


def _eval_check(c: dict[str, Any], before: Node, after: Node) -> bool | None:
    b, a = index_nodes(before), index_nodes(after)
    if "judge" in c:
        return None
    if "deleted" in c:
        return c["deleted"] not in a
    if "exists" in c:
        return c["exists"] in a
    if "node" in c:
        nid = c["node"]
        if nid not in a:
            return False
        if "prop" in c:
            old = b[nid][0]["props"].get(c["prop"]) if nid in b else None
            return _cmp(a[nid][0]["props"].get(c["prop"]), {k: v for k, v in c.items() if k in _OPS}, old)
        if "parent" in c:
            p = a[nid][1]
            return p is not None and p["id"] == c["parent"]
        return None
    if "new_node" in c:
        spec = c["new_node"]
        for n, parent in walk(after):
            if n["id"] in b:
                continue
            if spec.get("parent") and not _in_subtree(after, spec["parent"], n["id"]):
                continue
            if _node_matches(n, spec):
                return True
        return False
    if "order" in c:
        ids = c["order"]
        if any(i not in a for i in ids):
            return False
        parent = a[ids[0]][1]
        if parent is None or any(a[i][1] is not parent for i in ids):
            return False
        pos = [next(k for k, ch in enumerate(parent["children"]) if ch["id"] == i) for i in ids]
        return pos == sorted(pos)
    if "same" in c:
        ids, props = c["same"], c.get("props", [])
        if any(i not in a for i in ids):
            return False
        return all(len({json.dumps(_norm(a[i][0]["props"].get(p))) for i in ids}) == 1 for p in props)
    if "count" in c:  # {"count": {"type": "text", "parent": "n4", ...}, "eq": 3}
        spec = c["count"]
        n = sum(1 for x, _ in walk(after) if _node_matches(x, spec)
                and (not spec.get("parent") or _in_subtree(after, spec["parent"], x["id"])))
        return _cmp(n, {k: v for k, v in c.items() if k in _OPS})
    return None


def _in_subtree(root: Node, anc_id: str, nid: str) -> bool:
    idx = index_nodes(root)
    if anc_id not in idx:
        return False
    return any(n["id"] == nid for n, _ in walk(idx[anc_id][0]))


def keep_out(before: Node, after: Node, allowed: list[str]) -> tuple[float, list[str]]:
    """1.0 when nothing outside ``allowed`` changed; each outside change costs a third."""
    b, a = index_nodes(before), index_nodes(after)
    allowed = set(allowed)
    bad = []
    for nid, (n, parent) in b.items():
        if nid in allowed:
            continue
        if nid not in a:
            bad.append(f"{nid} deleted")
            continue
        m, mp = a[nid]
        if n["props"] != m["props"]:
            bad.append(f"{nid} props changed")
        if (parent or {}).get("id") != (mp or {}).get("id"):
            bad.append(f"{nid} moved")
        kids_b = [c["id"] for c in n["children"]]
        kids_a = [c["id"] for c in m["children"] if c["id"] in b]
        if [k for k in kids_b if k in a] != kids_a:
            bad.append(f"{nid} children reordered")
    return max(0.0, 1 - len(bad) / 3), bad


def _raw(before: Node, after: Node, checks: list[dict], allowed: list[str]) -> tuple[float, float, float, list]:
    det = [(c, eval_check(c, before, after)) for c in checks]
    det = [(c, v) for c, v in det if v is not None]
    kin = sum(v for _, v in det) / len(det) if det else 1.0
    kout, bad = keep_out(before, after, allowed)
    nolint = 0.0 if set(lint(after)) - set(lint(before)) else 1.0
    return 0.6 * kin + 0.3 * kout + 0.1 * nolint, kin, kout, [c for c, v in det if not v] + bad


def grade_checks(before: Node | str, patch_text: str, checks: list[dict] | str, allowed: list[str] | str) -> dict:
    before = parse(before) if isinstance(before, str) else before
    checks = json.loads(checks) if isinstance(checks, str) else checks
    allowed = json.loads(allowed) if isinstance(allowed, str) else allowed
    try:
        after = apply_patch(before, patch_text)
    except (PatchError, ValueError, KeyError, IndexError) as e:
        return {"valid": False, "score": 0.0, "pass": False, "keep_in": 0.0, "keep_out": 0.0, "why": str(e)[:100]}
    if dsl_violations(before, after):
        return {"valid": False, "score": 0.0, "pass": False, "keep_in": 0.0, "keep_out": 0.0,
                "why": "not in the design language"}
    raw, kin, kout, failed = _raw(before, after, checks, allowed)
    noop, *_ = _raw(before, before, checks, allowed)
    score = max(0.0, (raw - noop) / (1 - noop)) if noop < 1 else 0.0
    return {"valid": True, "score": round(score, 4), "pass": kin == 1.0 and kout == 1.0, "keep_in": round(kin, 4),
            "keep_out": round(kout, 4), "why": "; ".join(map(str, failed))[:300]}
