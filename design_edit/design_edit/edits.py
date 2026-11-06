"""Synthetic edit operators: (design, rng) -> EditExample.

There are two families:

* **Forward edits** start from a clean design and apply a requested change:
  "make the Upgrade button green". before = clean, after = edited.
* **Reverse edits (repairs)** first break a clean design in a known way
  (low contrast, off-scale spacing, inconsistent buttons) and ask the model to
  fix it. before = corrupted, after = the original clean design, so the correct
  answer is known exactly.

The target patch is never written by hand. It is ``schema.diff(before, after)``,
so every training label applies cleanly.

Vague requests ("make the title bigger") carry ``relax`` entries so the reward
accepts any value that moves in the right direction, not only the one in the
reference answer.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from . import generator as g
from .rewards import contrast
from .schema import Node, find, next_id, walk

COLOR_WORDS = {**{k: v for k, v in g.BRANDS.items() if k != "black"}, "black": "#111827",
               "gray": "#6B7280", "yellow": "#CA8A04"}


@dataclass
class EditExample:
    before: Node
    after: Node
    instruction: str
    kind: str
    # Each entry: {"id": node id, "prop": key, "cmp": "gt" | "lt" | "any"}
    relax: list[dict[str, Any]] = field(default_factory=list)


EditFn = Callable[[Node, random.Random], "EditExample | None"]
FORWARD: dict[str, EditFn] = {}
REPAIR: dict[str, EditFn] = {}


def forward(fn: EditFn) -> EditFn:
    FORWARD[fn.__name__] = fn
    return fn


def repair(fn: EditFn) -> EditFn:
    REPAIR[fn.__name__] = fn
    return fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nodes(doc: Node, pred: Callable[[Node, Node | None], bool]) -> list[tuple[Node, Node | None]]:
    """Candidate targets matching ``pred`` that a request can name unambiguously.

    Nodes whose label is shared with another node (e.g. two identical "Get started" buttons)
    are excluded: "the Get started label" would have more than one right answer.
    """
    counts: dict[str, int] = {}
    for n, _ in walk(doc):
        counts[label(n)] = counts.get(label(n), 0) + 1
    return [(n, p) for n, p in walk(doc) if pred(n, p) and counts[label(n)] == 1]


def is_button(n: Node) -> bool:
    return n["type"] == "frame" and n["props"].get("name", "").endswith("button")


def button_label(n: Node) -> str:
    return n["children"][0]["props"]["text"] if n["children"] else n["props"].get("name", "")


def label(n: Node) -> str:
    """How an instruction refers to a node."""
    if is_button(n):
        return f"the “{button_label(n)}” button"
    return f"the {n['props'].get('name', n['type'])}"


def pick(rng: random.Random, items: list[Any]) -> Any:
    return rng.choice(items) if items else None


def phrase(rng: random.Random, *templates: str, **kw: Any) -> str:
    s = rng.choice(templates).format(**kw)
    s = s[0].upper() + s[1:]
    if rng.random() < 0.15:  # casual lowercase variant
        s = s[0].lower() + s[1:]
    return s


def step(scale: list[int], value: int, delta: int) -> int | None:
    above = [v for v in scale if v > value]
    below = [v for v in scale if v < value]
    if delta > 0:
        return above[min(delta, len(above)) - 1] if above else None
    return below[-min(-delta, len(below))] if below else None


def mutate(doc: Node) -> tuple[Node, Callable[[str], Node]]:
    after = copy.deepcopy(doc)
    return after, lambda nid: find(after, nid)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Forward edits
# ---------------------------------------------------------------------------

@forward
def recolor_button(doc: Node, rng: random.Random) -> EditExample | None:
    btn, _ = pick(rng, nodes(doc, lambda n, p: is_button(n))) or (None, None)
    if btn is None:
        return None
    word, hexv = rng.choice([kv for kv in COLOR_WORDS.items() if kv[1] != btn["props"].get("fill")])
    after, f = mutate(doc)
    nb = f(btn["id"])
    nb["props"]["fill"] = hexv
    relax: list[dict[str, Any]] = []
    lab = nb["children"][0] if nb["children"] else None
    if lab is not None and contrast(lab["props"].get("color", "#111827"), hexv) < 3.0:
        # The label must stay readable on the new fill; any readable color is fine.
        lab["props"]["color"] = "#FFFFFF" if contrast("#FFFFFF", hexv) >= contrast("#111827", hexv) else "#111827"
        relax.append({"id": lab["id"], "prop": "color", "cmp": "contrast", "bg": hexv, "min": 3.0})
    if rng.random() < 0.6:
        instr = phrase(rng, "make {b} {w}", "change {b} to {w}", "can {b} be {w} instead?", b=label(btn), w=word)
        relax.append({"id": btn["id"], "prop": "fill", "cmp": "hue", "target": hexv})
    else:
        instr = phrase(rng, "set the background of {b} to {h}", "{b} should use {h}", b=label(btn), h=hexv)
    return EditExample(doc, after, instr, "recolor_button", relax)


@forward
def change_text(doc: Node, rng: random.Random) -> EditExample | None:
    t, _ = pick(rng, nodes(doc, lambda n, p: n["type"] == "text" and n["props"].get("name") in
                           ("Title", "Heading", "Body", "Subheading", "Section title", "Logo"))) or (None, None)
    if t is None:
        return None
    pool = {"Title": g.TITLES, "Heading": g.TITLES, "Section title": ["Account", "Privacy", "Appearance"],
            "Logo": ["Fieldnote", "Kite", "Basil"]}.get(t["props"]["name"], g.BODIES)
    new = rng.choice([x for x in pool if x != t["props"]["text"]])
    after, f = mutate(doc)
    f(t["id"])["props"]["text"] = new
    instr = phrase(rng, "change “{o}” to “{n}”", "update {l} to say “{n}”",
                   "rewrite {l}: {n}", "the {ln} should read “{n}”",
                   o=t["props"]["text"], n=new, l=label(t), ln=t["props"]["name"].lower())
    return EditExample(doc, after, instr, "change_text")


@forward
def resize_text(doc: Node, rng: random.Random) -> EditExample | None:
    t, _ = pick(rng, nodes(doc, lambda n, p: n["type"] == "text")) or (None, None)
    if t is None:
        return None
    up = rng.random() < 0.6
    new = step(g.TYPE_SCALE, t["props"]["size"], 1 if up else -1)
    if new is None:
        return None
    after, f = mutate(doc)
    f(t["id"])["props"]["size"] = new
    if rng.random() < 0.5:
        instr = phrase(rng, "set {l} font size to {n}", "make {l} {n}px", l=label(t), n=new)
        relax = []
    else:
        instr = phrase(rng, "make {l} {d}", "{l} feels too {o}", "can you make {l} a bit {d}?",
                       l=label(t), d="bigger" if up else "smaller", o="small" if up else "big")
        relax = [{"id": t["id"], "prop": "size", "cmp": "gt" if up else "lt"}]
    return EditExample(doc, after, instr, "resize_text", relax)


@forward
def set_radius(doc: Node, rng: random.Random) -> EditExample | None:
    n, _ = pick(rng, nodes(doc, lambda n, p: n["type"] in ("frame", "image") and "radius" in n["props"])) or (None, None)
    if n is None:
        return None
    rounder = rng.random() < 0.6
    new = step(g.RADII, n["props"]["radius"], 2 if rounder else -2)
    if new is None:
        return None
    after, f = mutate(doc)
    f(n["id"])["props"]["radius"] = new
    if new == 0:
        instr, relax = phrase(rng, "make {l} square", "remove the rounded corners on {l}", l=label(n)), []
    elif rng.random() < 0.5:
        instr, relax = phrase(rng, "set the corner radius of {l} to {r}", "{l}: radius {r}px", l=label(n), r=new), []
    else:
        instr = phrase(rng, "make the corners of {l} {d}", "{l} looks too {o}",
                       l=label(n), d="rounder" if rounder else "sharper", o="boxy" if rounder else "rounded")
        relax = [{"id": n["id"], "prop": "radius", "cmp": "gt" if rounder else "lt"}]
    return EditExample(doc, after, instr, "set_radius", relax)


@forward
def set_spacing(doc: Node, rng: random.Random) -> EditExample | None:
    n, _ = pick(rng, nodes(doc, lambda n, p: n["type"] == "frame" and len(n["children"]) > 1
                           and "gap" in n["props"] and not is_button(n))) or (None, None)
    if n is None:
        return None
    prop = "pad" if "pad" in n["props"] and rng.random() < 0.4 else "gap"
    more = rng.random() < 0.6
    new = step(g.SPACING, n["props"][prop], 1 if more else -1)
    if new is None:
        return None
    after, f = mutate(doc)
    f(n["id"])["props"][prop] = new
    what = "padding" if prop == "pad" else "spacing between items"
    if rng.random() < 0.5:
        instr, relax = phrase(rng, "set the {w} in {l} to {v}", "{l} {w}: {v}px", w=what, l=label(n), v=new), []
    else:
        instr = phrase(rng, "{v} the {w} in {l}", "{l} feels {f}",
                       v="increase" if more else "reduce", w=what, l=label(n),
                       f=("cramped" if more else "too airy") if prop == "gap" else ("tight" if more else "too padded"))
        relax = [{"id": n["id"], "prop": prop, "cmp": "gt" if more else "lt"}]
    return EditExample(doc, after, instr, "set_spacing", relax)


@forward
def delete_node(doc: Node, rng: random.Random) -> EditExample | None:
    n, p = pick(rng, nodes(doc, lambda n, p: p is not None and p["id"] != doc["id"] and len(p["children"]) > 1
                           and not n["props"].get("name", "").endswith(" label"))) or (None, None)
    if n is None:
        return None
    after, f = mutate(doc)
    np_ = f(p["id"])
    np_["children"] = [c for c in np_["children"] if c["id"] != n["id"]]
    instr = phrase(rng, "remove {l}", "delete {l}", "get rid of {l}", "we don't need {l}", l=label(n))
    return EditExample(doc, after, instr, "delete_node")


@forward
def add_secondary_button(doc: Node, rng: random.Random) -> EditExample | None:
    btn, parent = pick(rng, nodes(doc, lambda n, p: is_button(n) and p is not None and p["props"].get("layout") == "row"
                                  and len(p["children"]) == 1)) or (None, None)
    if btn is None:
        return None
    lab = rng.choice(g.SECONDARY_LABELS)
    after, f = mutate(doc)
    nid = int(next_id(after)[1:])
    new_btn = {"id": f"n{nid}", "type": "frame", "children": [
        {"id": f"n{nid + 1}", "type": "text", "children": [],
         "props": dict(name=f"{lab} label", text=lab, size=14, weight=600, color="#111827")}],
        "props": dict(name=f"{lab} button", layout="row", pad=btn["props"].get("pad", 12), align="center",
                      justify="center", radius=btn["props"].get("radius", 8), fill="#FFFFFF", stroke="#D1D5DB")}
    before_it = rng.random() < 0.3
    f(parent["id"])["children"].insert(0 if before_it else 1, new_btn)
    instr = phrase(rng, "add a secondary “{n}” button {w} {l}", "put a “{n}” button {w} {l}",
                   "{l} needs a “{n}” option {w} it",
                   n=lab, w="before" if before_it else "next to", l=label(btn))
    return EditExample(doc, after, instr, "add_secondary_button")


@forward
def add_text(doc: Node, rng: random.Random) -> EditExample | None:
    t, parent = pick(rng, nodes(doc, lambda n, p: n["type"] == "text" and n["props"].get("name") in ("Title", "Heading"))) or (None, None)
    if t is None:
        return None
    body = rng.choice(g.BODIES)
    after, f = mutate(doc)
    np_ = f(parent["id"])
    idx = [c["id"] for c in np_["children"]].index(t["id"])
    np_["children"].insert(idx + 1, {"id": next_id(after), "type": "text", "children": [],
                                     "props": dict(name="Subtitle", text=body, size=14, weight=400, color="#6B7280")})
    instr = phrase(rng, "add a subtitle under {l} that says “{b}”",
                   "insert a muted line of text below {l}: “{b}”", l=label(t), b=body)
    return EditExample(doc, after, instr, "add_text")


@forward
def reorder(doc: Node, rng: random.Random) -> EditExample | None:
    p, _ = pick(rng, nodes(doc, lambda n, p: n["type"] == "frame" and len(n["children"]) >= 2 and not is_button(n)
                           and n["id"] != doc["id"])) or (None, None)
    if p is None:
        return None
    i, j = sorted(rng.sample(range(len(p["children"])), 2))
    a, b = p["children"][i], p["children"][j]
    after, f = mutate(doc)
    kids = f(p["id"])["children"]
    moved = kids.pop(j)
    kids.insert(i, moved)
    horizontal = p["props"].get("layout") == "row"
    instr = phrase(rng, "move {b} {pos} {a}", "put {b} {pos} {a}", "{b} should come {pos2} {a}",
                   b=label(b), a=label(a), pos="to the left of" if horizontal else "above",
                   pos2="before")
    return EditExample(doc, after, instr, "reorder")


@forward
def toggle_layout(doc: Node, rng: random.Random) -> EditExample | None:
    n, _ = pick(rng, nodes(doc, lambda n, p: n["type"] == "frame" and n["props"].get("layout") in ("row", "col")
                           and len(n["children"]) >= 2 and not is_button(n) and n["id"] != doc["id"])) or (None, None)
    if n is None:
        return None
    to = "col" if n["props"]["layout"] == "row" else "row"
    after, f = mutate(doc)
    f(n["id"])["props"]["layout"] = to
    instr = phrase(rng, "stack {l} vertically" if to == "col" else "lay out {l} horizontally",
                   "switch {l} to a {d} layout", l=label(n), d="vertical" if to == "col" else "horizontal")
    return EditExample(doc, after, instr, "toggle_layout")


@forward
def align_center(doc: Node, rng: random.Random) -> EditExample | None:
    n, _ = pick(rng, nodes(doc, lambda n, p: n["type"] == "frame" and n["props"].get("layout") == "col"
                           and n["props"].get("align") != "center" and n["id"] != doc["id"])) or (None, None)
    if n is None:
        return None
    after, f = mutate(doc)
    f(n["id"])["props"]["align"] = "center"
    instr = phrase(rng, "center the contents of {l}", "center-align everything in {l}", l=label(n))
    return EditExample(doc, after, instr, "align_center")


@forward
def set_opacity(doc: Node, rng: random.Random) -> EditExample | None:
    n, _ = pick(rng, nodes(doc, lambda n, p: p is not None and "opacity" not in n["props"])) or (None, None)
    if n is None:
        return None
    pct = rng.choice([25, 40, 50, 60, 75])
    after, f = mutate(doc)
    f(n["id"])["props"]["opacity"] = pct / 100
    instr = phrase(rng, "set {l} to {p}% opacity", "make {l} {q}% transparent", l=label(n), p=pct, q=100 - pct)
    return EditExample(doc, after, instr, "set_opacity")


# ---------------------------------------------------------------------------
# Repairs (reverse edits): corrupt a clean design, ask for the fix
# ---------------------------------------------------------------------------

@repair
def fix_contrast(doc: Node, rng: random.Random) -> EditExample | None:
    t, _ = pick(rng, nodes(doc, lambda n, p: n["type"] == "text" and p is not None
                           and p["props"].get("fill", "#FFFFFF") in ("#FFFFFF", "#F9FAFB"))) or (None, None)
    if t is None:
        return None
    broken, f = mutate(doc)
    f(t["id"])["props"]["color"] = rng.choice(["#E5E7EB", "#F3F4F6", "#D1D5DB", "#FAFAFA"])
    instr = phrase(rng, "{l} is hard to read, fix the contrast", "the {n} text is barely visible",
                   "fix the low-contrast text in {l}", l=label(t), n=t["props"]["name"])
    return EditExample(broken, doc, instr, "fix_contrast",
                       [{"id": t["id"], "prop": "color", "cmp": "contrast", "bg": "#FFFFFF", "min": 4.5}])


@repair
def fix_offscale_spacing(doc: Node, rng: random.Random) -> EditExample | None:
    n, _ = pick(rng, nodes(doc, lambda n, p: n["type"] == "frame" and n["props"].get("gap", 0) >= 8)) or (None, None)
    if n is None:
        return None
    broken, f = mutate(doc)
    bad = n["props"]["gap"] + rng.choice([-3, -1, 1, 3, 5])
    f(n["id"])["props"]["gap"] = bad
    instr = phrase(rng, "snap the spacing in {l} back to the spacing scale", "{l} has an off-grid gap, fix it",
                   "the spacing in {l} isn't on our 4px grid", l=label(n))
    ok = sorted({v for v in g.SPACING if abs(v - bad) <= 4} | {n["props"]["gap"]})
    return EditExample(broken, doc, instr, "fix_offscale_spacing",
                       [{"id": n["id"], "prop": "gap", "cmp": "allowed", "values": ok}])


@repair
def fix_offscale_type(doc: Node, rng: random.Random) -> EditExample | None:
    t, _ = pick(rng, nodes(doc, lambda n, p: n["type"] == "text")) or (None, None)
    if t is None:
        return None
    broken, f = mutate(doc)
    bad = t["props"]["size"] + rng.choice([-1, 1, 3])
    f(t["id"])["props"]["size"] = bad
    instr = phrase(rng, "{l} uses an off-scale font size, fix it", "snap {l} to the type scale", l=label(t))
    # Either neighbor on the scale is a valid fix.
    ok = sorted({max([v for v in g.TYPE_SCALE if v < bad], default=bad),
                 min([v for v in g.TYPE_SCALE if v > bad], default=bad), t["props"]["size"]} & set(g.TYPE_SCALE))
    return EditExample(broken, doc, instr, "fix_offscale_type",
                       [{"id": t["id"], "prop": "size", "cmp": "allowed", "values": ok}])


@repair
def fix_button_consistency(doc: Node, rng: random.Random) -> EditExample | None:
    btns = [n for n, _ in nodes(doc, lambda n, p: is_button(n))]
    if len(btns) < 2:
        return None
    b = rng.choice(btns)
    broken, f = mutate(doc)
    nb = f(b["id"])
    prop = rng.choice(["radius", "pad"])
    nb["props"][prop] = rng.choice([v for v in (0, 2, 4, 10, 14, 18, 20) if v != b["props"][prop]])
    instr = phrase(rng, "make the buttons consistent", "one of the buttons doesn't match the others, fix it",
                   "{l} looks different from the other buttons", l=label(b))
    return EditExample(broken, doc, instr, "fix_button_consistency")


@repair
def restore_missing_label(doc: Node, rng: random.Random) -> EditExample | None:
    btn, _ = pick(rng, nodes(doc, lambda n, p: is_button(n))) or (None, None)
    if btn is None:
        return None
    broken, f = mutate(doc)
    f(btn["id"])["children"] = []
    lab = button_label(btn)
    instr = phrase(rng, "the {n} is empty, give it a “{t}” label", "add the text “{t}” inside the {n}",
                   n=btn["props"]["name"], t=lab)
    return EditExample(broken, doc, instr, "restore_missing_label")


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def compose(doc: Node, rng: random.Random, n_edits: int = 2) -> EditExample | None:
    """Chain several forward edits into one multi-part request."""
    cur, parts, relax = doc, [], []
    orig = {n["id"]: n for n, _ in walk(doc)}
    names = list(FORWARD)
    rng.shuffle(names)
    for name in names:
        if len(parts) == n_edits:
            break
        ex = FORWARD[name](cur, rng)
        if ex is None:
            continue
        # Skip edits that clash with an earlier part (e.g. delete a node we just recolored).
        cur_ids = {n["id"] for n, _ in walk(ex.after)}
        if any(r["id"] not in cur_ids for r in relax):
            continue
        # Skip edits that reuse the id of a node deleted earlier in the chain.
        cur_before = {n["id"] for n, _ in walk(cur)}
        if any(nid in orig and nid not in cur_before for nid in cur_ids):
            continue
        parts.append(ex.instruction.rstrip(".?"))
        relax += ex.relax
        cur = ex.after
    if len(parts) < 2:
        return None
    joiner = rng.choice([". Also, ", ", and ", "; then ", ".\n"])
    text = parts[0] + "".join(joiner + p[0].lower() + p[1:] for p in parts[1:])
    return EditExample(doc, cur, text, "compose", relax)


ALL_EDITS: dict[str, EditFn] = {**FORWARD, **REPAIR, "compose": compose}
