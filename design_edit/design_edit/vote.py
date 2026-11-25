"""Self-consistency for design edits: sample several answers, keep the one whose result most samples agree on.

Two answers count as the same when they produce the same design up to the ids and layer names of newly inserted
nodes (arbitrary choices that don't change what a user sees). Invalid or no-op patches don't vote.
"""

from __future__ import annotations

import re
from collections import Counter

from .schema import PatchError, apply_patch, parse, serialize, strip_fences

_LINE = re.compile(r'^(\s*\w+)#(n\d+) "(?:[^"\\]|\\.)*"(.*)$')


def canonical(before_text: str, output: str) -> str | None:
    """The applied result with new node ids renumbered in order and new nodes' layer names dropped."""
    patch = strip_fences(output).strip()
    if not patch:
        return None
    try:
        before = parse(before_text)
        after_text = serialize(apply_patch(before, patch))
    except (PatchError, ValueError, KeyError, IndexError):
        return None
    if after_text == serialize(before):
        return None
    old_ids = set(re.findall(r"#(n\d+) ", before_text))
    new_ids: dict[str, str] = {}
    lines = []
    for line in after_text.splitlines():
        m = _LINE.match(line)
        if m and m.group(2) not in old_ids:
            nid = new_ids.setdefault(m.group(2), f"new{len(new_ids)}")
            line = f"{m.group(1)}#{nid}{m.group(3)}"
        lines.append(line)
    return "\n".join(lines)


def vote(before_text: str, outputs: list[str]) -> int:
    """Index of the chosen output: the first member of the largest group of equivalent valid answers (0 if none)."""
    keys = [canonical(before_text, o) for o in outputs]
    counts = Counter(k for k in keys if k is not None)
    if not counts:
        return 0
    best = max(counts.values())
    return next(i for i, k in enumerate(keys) if k is not None and counts[k] == best)
