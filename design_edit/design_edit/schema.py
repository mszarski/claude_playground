"""Design document model, text serialization, and the edit-patch language.

A design is a tree of nodes. Each node is a plain dict::

    {"id": "n3", "type": "frame", "props": {"name": "Card", "w": 360, ...}, "children": [...]}

Node types: ``frame`` (container, optionally auto-layout), ``text``, ``rect``, ``image``.

Serialization ("design DSL") is one node per line, indented two spaces per level::

    frame#n1 "Card" w=360 layout=col gap=12 pad=24 fill=#FFFFFF radius=12
      text#n2 "Title" text="Welcome back" size=24 weight=600 color=#111827

The model never rewrites the whole document. It emits a *patch*: one op per line.

    set <id> key=value [key=value ...]
    unset <id> key [key ...]
    delete <id>
    move <id> <parent-id> <index>
    insert <parent-id> <index> <node-line>

``insert`` takes a node line in the same DSL (without indentation). To insert a
subtree, insert the parent first and then its children under the new id.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Iterator

NODE_TYPES = ("frame", "text", "rect", "image")

# Canonical property order for serialization. Unknown keys are appended sorted.
PROP_ORDER = (
    "text", "layout", "w", "h", "x", "y", "gap", "pad", "align", "justify",
    "fill", "stroke", "radius", "opacity", "size", "weight", "color",
)

Node = dict[str, Any]


class PatchError(ValueError):
    """Raised when a patch cannot be parsed or applied."""


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------

def make_node(node_id: str, node_type: str, children: list[Node] | None = None, **props: Any) -> Node:
    if node_type not in NODE_TYPES:
        raise ValueError(f"unknown node type {node_type!r}")
    return {"id": node_id, "type": node_type, "props": dict(props), "children": list(children or [])}


def walk(node: Node, parent: Node | None = None) -> Iterator[tuple[Node, Node | None]]:
    """Pre-order traversal yielding ``(node, parent)``."""
    yield node, parent
    for child in node["children"]:
        yield from walk(child, node)


def index_nodes(root: Node) -> dict[str, tuple[Node, Node | None]]:
    out: dict[str, tuple[Node, Node | None]] = {}
    for node, parent in walk(root):
        if node["id"] in out:
            raise PatchError(f"duplicate id {node['id']}")
        out[node["id"]] = (node, parent)
    return out


def find(root: Node, node_id: str) -> Node | None:
    for node, _ in walk(root):
        if node["id"] == node_id:
            return node
    return None


def next_id(root: Node) -> str:
    nums = [int(n["id"][1:]) for n, _ in walk(root) if re.fullmatch(r"n\d+", n["id"])]
    return f"n{max(nums, default=0) + 1}"


# ---------------------------------------------------------------------------
# Value encoding
# ---------------------------------------------------------------------------

_BARE_RE = re.compile(r"^[A-Za-z0-9#._%-]+$")
_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


def format_value(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, (int, float)):
        return repr(value)
    s = str(value)
    if key in ("text", "name") or not _BARE_RE.match(s) or _NUM_RE.match(s) or s in ("true", "false"):
        return json.dumps(s, ensure_ascii=False)
    return s


def parse_value(raw: str) -> Any:
    if raw.startswith('"'):
        return json.loads(raw)
    if raw in ("true", "false"):
        return raw == "true"
    if _NUM_RE.match(raw):
        return float(raw) if "." in raw else int(raw)
    return raw


# token = quoted string | key=value (value possibly quoted) | bare word
_TOKEN_RE = re.compile(r'\s*(?:(\w+)=("(?:[^"\\]|\\.)*"|\S+)|("(?:[^"\\]|\\.)*")|(\S+))')


def tokenize(s: str) -> list[tuple[str, Any]]:
    """Split a line into tokens: ("kv", (k, v)), ("str", s) or ("word", w)."""
    out: list[tuple[str, Any]] = []
    pos = 0
    s = s.rstrip()
    while pos < len(s):
        m = _TOKEN_RE.match(s, pos)
        if not m or m.end() == pos:
            raise PatchError(f"cannot tokenize near {s[pos:]!r}")
        if m.group(1) is not None:
            try:
                out.append(("kv", (m.group(1), parse_value(m.group(2)))))
            except json.JSONDecodeError as e:
                raise PatchError(f"bad value {m.group(2)!r}") from e
        elif m.group(3) is not None:
            out.append(("str", json.loads(m.group(3))))
        else:
            out.append(("word", m.group(4)))
        pos = m.end()
    return out


# ---------------------------------------------------------------------------
# Document serialization
# ---------------------------------------------------------------------------

def _sorted_props(props: dict[str, Any]) -> list[tuple[str, Any]]:
    known = [(k, props[k]) for k in PROP_ORDER if k in props]
    extra = sorted((k, v) for k, v in props.items() if k not in PROP_ORDER and k != "name")
    return known + extra


def node_line(node: Node) -> str:
    parts = [f"{node['type']}#{node['id']}"]
    if "name" in node["props"]:
        parts.append(json.dumps(node["props"]["name"], ensure_ascii=False))
    parts += [f"{k}={format_value(k, v)}" for k, v in _sorted_props(node["props"])]
    return " ".join(parts)


def serialize(root: Node) -> str:
    lines: list[str] = []

    def rec(node: Node, depth: int) -> None:
        lines.append("  " * depth + node_line(node))
        for c in node["children"]:
            rec(c, depth + 1)

    rec(root, 0)
    return "\n".join(lines)


def parse_node_line(line: str) -> Node:
    toks = tokenize(line.strip())
    if not toks or toks[0][0] != "word" or "#" not in toks[0][1]:
        raise PatchError(f"expected type#id at start of node line: {line!r}")
    node_type, node_id = toks[0][1].split("#", 1)
    if node_type not in NODE_TYPES:
        raise PatchError(f"unknown node type {node_type!r}")
    if not node_id:
        raise PatchError("empty node id")
    props: dict[str, Any] = {}
    rest = toks[1:]
    if rest and rest[0][0] == "str":
        props["name"] = rest[0][1]
        rest = rest[1:]
    for kind, val in rest:
        if kind != "kv":
            raise PatchError(f"unexpected token {val!r} in node line")
        props[val[0]] = val[1]
    return {"id": node_id, "type": node_type, "props": props, "children": []}


def parse(text: str) -> Node:
    stack: list[tuple[int, Node]] = []
    root: Node | None = None
    for raw in text.splitlines():
        if not raw.strip():
            continue
        depth = (len(raw) - len(raw.lstrip(" "))) // 2
        node = parse_node_line(raw)
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if stack:
            stack[-1][1]["children"].append(node)
        elif root is None:
            root = node
        else:
            raise PatchError("multiple roots")
        stack.append((depth, node))
    if root is None:
        raise PatchError("empty document")
    return root


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------

def strip_fences(text: str) -> str:
    """Remove markdown code fences a model might wrap its patch in."""
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", text, re.S)
    return m.group(1) if m else text


def parse_patch(text: str) -> list[dict[str, Any]]:
    ops: list[dict[str, Any]] = []
    for lineno, raw in enumerate(strip_fences(text).splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        verb, _, rest = line.partition(" ")
        try:
            if verb == "set":
                toks = tokenize(rest)
                if len(toks) < 2 or toks[0][0] != "word" or any(k != "kv" for k, _ in toks[1:]):
                    raise PatchError("usage: set <id> key=value ...")
                ops.append({"op": "set", "id": toks[0][1], "props": dict(v for _, v in toks[1:])})
            elif verb == "unset":
                toks = tokenize(rest)
                if len(toks) < 2 or any(k != "word" for k, _ in toks):
                    raise PatchError("usage: unset <id> key ...")
                ops.append({"op": "unset", "id": toks[0][1], "keys": [v for _, v in toks[1:]]})
            elif verb == "delete":
                toks = tokenize(rest)
                if len(toks) != 1 or toks[0][0] != "word":
                    raise PatchError("usage: delete <id>")
                ops.append({"op": "delete", "id": toks[0][1]})
            elif verb == "move":
                parts = rest.split()
                if len(parts) != 3 or not parts[2].lstrip("-").isdigit():
                    raise PatchError("usage: move <id> <parent> <index>")
                ops.append({"op": "move", "id": parts[0], "parent": parts[1], "index": int(parts[2])})
            elif verb == "insert":
                parts = rest.split(None, 2)
                if len(parts) != 3 or not parts[1].lstrip("-").isdigit():
                    raise PatchError("usage: insert <parent> <index> <node-line>")
                ops.append({"op": "insert", "parent": parts[0], "index": int(parts[1]),
                            "node": parse_node_line(parts[2])})
            else:
                raise PatchError(f"unknown op {verb!r}")
        except PatchError as e:
            raise PatchError(f"line {lineno}: {e}") from None
    return ops


def format_op(op: dict[str, Any]) -> str:
    kind = op["op"]
    if kind == "set":
        kv = " ".join(f"{k}={format_value(k, v)}" for k, v in _sorted_props(op["props"]))
        if "name" in op["props"]:
            kv = f"name={format_value('name', op['props']['name'])} {kv}".strip()
        return f"set {op['id']} {kv}"
    if kind == "unset":
        return f"unset {op['id']} {' '.join(op['keys'])}"
    if kind == "delete":
        return f"delete {op['id']}"
    if kind == "move":
        return f"move {op['id']} {op['parent']} {op['index']}"
    if kind == "insert":
        return f"insert {op['parent']} {op['index']} {node_line(op['node'])}"
    raise ValueError(kind)


def format_patch(ops: list[dict[str, Any]]) -> str:
    return "\n".join(format_op(op) for op in ops)


def _is_ancestor(anc: Node, node: Node) -> bool:
    return any(n is node for n, _ in walk(anc))


def apply_ops(root: Node, ops: list[dict[str, Any]], *, inplace: bool = False) -> Node:
    doc = root if inplace else copy.deepcopy(root)
    for i, op in enumerate(ops, 1):
        idx = index_nodes(doc)
        kind = op["op"]

        def get(nid: str) -> tuple[Node, Node | None]:
            if nid not in idx:
                raise PatchError(f"op {i}: no node {nid!r}")
            return idx[nid]

        if kind == "set":
            node, _ = get(op["id"])
            node["props"].update(copy.deepcopy(op["props"]))
        elif kind == "unset":
            node, _ = get(op["id"])
            for k in op["keys"]:
                if k not in node["props"]:
                    raise PatchError(f"op {i}: {op['id']} has no prop {k!r}")
                del node["props"][k]
        elif kind == "delete":
            node, parent = get(op["id"])
            if parent is None:
                raise PatchError(f"op {i}: cannot delete root")
            parent["children"].remove(node)
        elif kind == "move":
            node, parent = get(op["id"])
            new_parent, _ = get(op["parent"])
            if parent is None:
                raise PatchError(f"op {i}: cannot move root")
            if _is_ancestor(node, new_parent):
                raise PatchError(f"op {i}: cannot move a node into itself")
            parent["children"].remove(node)
            if not 0 <= op["index"] <= len(new_parent["children"]):
                raise PatchError(f"op {i}: index {op['index']} out of range")
            new_parent["children"].insert(op["index"], node)
        elif kind == "insert":
            parent, _ = get(op["parent"])
            new = copy.deepcopy(op["node"])
            if new["id"] in idx:
                raise PatchError(f"op {i}: id {new['id']!r} already exists")
            if parent["type"] != "frame":
                raise PatchError(f"op {i}: can only insert into a frame")
            if not 0 <= op["index"] <= len(parent["children"]):
                raise PatchError(f"op {i}: index {op['index']} out of range")
            parent["children"].insert(op["index"], new)
        else:
            raise PatchError(f"op {i}: unknown op {kind!r}")
    return doc


def apply_patch(root: Node, patch_text: str) -> Node:
    return apply_ops(root, parse_patch(patch_text))


# ---------------------------------------------------------------------------
# Diff: the canonical patch turning ``a`` into ``b`` (used to build SFT targets)
# ---------------------------------------------------------------------------

def diff(a: Node, b: Node) -> list[dict[str, Any]]:
    if a["id"] != b["id"]:
        raise ValueError("roots must share an id")
    a_idx, b_idx = index_nodes(a), index_nodes(b)
    for nid, (bn, _) in b_idx.items():
        if nid in a_idx and a_idx[nid][0]["type"] != bn["type"]:
            raise ValueError(f"type change on {nid} is not representable")

    ops: list[dict[str, Any]] = []
    work = copy.deepcopy(a)

    # 1. Deletes: only top-most removed nodes (their subtrees go with them).
    for node, parent in walk(a):
        if node["id"] not in b_idx and parent is not None and parent["id"] in b_idx:
            ops.append({"op": "delete", "id": node["id"]})
    apply_ops(work, ops, inplace=True)

    # 2. Structure: visit b in pre-order and put each child where it belongs.
    struct_ops: list[dict[str, Any]] = []
    for bnode, _ in walk(b):
        for i, child in enumerate(bnode["children"]):
            w_idx = index_nodes(work)
            if child["id"] in w_idx:
                _, wparent = w_idx[child["id"]]
                if wparent is None or wparent["id"] != bnode["id"] or wparent["children"].index(w_idx[child["id"]][0]) != i:
                    op = {"op": "move", "id": child["id"], "parent": bnode["id"], "index": i}
                    apply_ops(work, [op], inplace=True)
                    struct_ops.append(op)
            else:
                shallow = {"id": child["id"], "type": child["type"],
                           "props": copy.deepcopy(child["props"]), "children": []}
                op = {"op": "insert", "parent": bnode["id"], "index": i, "node": shallow}
                apply_ops(work, [op], inplace=True)
                struct_ops.append(op)
    ops += struct_ops

    # 3. Props on nodes that already existed.
    w_idx = index_nodes(work)
    for bnode, _ in walk(b):
        if bnode["id"] not in a_idx:
            continue
        wprops, bprops = w_idx[bnode["id"]][0]["props"], bnode["props"]
        changed = {k: v for k, v in bprops.items() if wprops.get(k, _MISSING) != v}
        removed = [k for k in wprops if k not in bprops]
        if changed:
            ops.append({"op": "set", "id": bnode["id"], "props": changed})
        if removed:
            ops.append({"op": "unset", "id": bnode["id"], "keys": sorted(removed)})
    return ops


_MISSING = object()


def docs_equal(a: Node, b: Node) -> bool:
    return serialize(a) == serialize(b)
