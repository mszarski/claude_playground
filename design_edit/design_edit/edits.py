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


# --- Operation-coverage edits (added in v2). They teach unset, justify, weight and subtree
# --- inserts through requests that differ from the held-out NOVEL ones.

@forward
def remove_border(doc: Node, rng: random.Random) -> EditExample | None:
    n, _ = pick(rng, nodes(doc, lambda n, p: n["type"] == "frame" and "stroke" in n["props"]
                           and not is_button(n) and p is not None)) or (None, None)
    if n is None:
        return None
    after, f = mutate(doc)
    del f(n["id"])["props"]["stroke"]
    instr = phrase(rng, "remove the border from {l}", "get rid of the outline on {l}", "{l} doesn't need a stroke",
                   l=label(n))
    return EditExample(doc, after, instr, "remove_border")


@forward
def justify_items(doc: Node, rng: random.Random) -> EditExample | None:
    target = rng.choice(["end", "center", "start"])
    n, _ = pick(rng, nodes(doc, lambda n, p: n["type"] == "frame" and n["props"].get("layout") == "row"
                           and len(n["children"]) >= 2 and not is_button(n)
                           and n["props"].get("justify", "start") != target)) or (None, None)
    if n is None:
        return None
    after, f = mutate(doc)
    f(n["id"])["props"]["justify"] = target
    words = {"end": ("push the items in {l} to the right", "right-align the contents of {l}"),
             "center": ("center the items in {l} horizontally", "bunch the contents of {l} in the middle"),
             "start": ("move the items in {l} to the left", "left-align the contents of {l}")}[target]
    instr = phrase(rng, *words, "set justify to {t} on {l}", l=label(n), t=target)
    return EditExample(doc, after, instr, "justify_items")


@forward
def set_weight_value(doc: Node, rng: random.Random) -> EditExample | None:
    t, _ = pick(rng, nodes(doc, lambda n, p: n["type"] == "text" and n["props"].get("weight", 400) > 400)) or (None, None)
    if t is None:
        return None
    w = rng.choice([v for v in (400, 500) if v != t["props"]["weight"]])
    after, f = mutate(doc)
    f(t["id"])["props"]["weight"] = w
    if w == 400:
        instr = phrase(rng, "make {l} regular weight", "{l} is too heavy, use the normal weight",
                       "set the font weight of {l} to 400", l=label(t))
    else:
        instr = phrase(rng, "set the font weight of {l} to 500", "make {l} medium weight", l=label(t))
    return EditExample(doc, after, instr, "set_weight_value")


@forward
def lighten_text(doc: Node, rng: random.Random) -> EditExample | None:
    t, _ = pick(rng, nodes(doc, lambda n, p: n["type"] == "text" and n["props"].get("color") == "#111827"
                           and p is not None and p["props"].get("fill", "#FFFFFF") in ("#FFFFFF", "#F9FAFB"))) or (None, None)
    if t is None:
        return None
    after, f = mutate(doc)
    # v3: random lighter gray, so the model can't learn one fixed output color for this request.
    f(t["id"])["props"]["color"] = rng.choice(["#374151", "#4B5563", "#6B7280", "#9CA3AF"])
    instr = phrase(rng, "make {l} more muted", "{l} should use the secondary text color",
                   "tone down {l}, it's too dark", l=label(t))
    # Any lighter color that stays readable is fine.
    return EditExample(doc, after, instr, "lighten_text",
                       [{"id": t["id"], "prop": "color", "cmp": "lighter"}])


@forward
def add_field(doc: Node, rng: random.Random) -> EditExample | None:
    """Insert a 4-node form field (field > label, input > placeholder) after a named field."""
    fld, form_ = pick(rng, nodes(doc, lambda n, p: n["props"].get("name", "").endswith(" field")
                                 and len(n["children"]) == 2)) or (None, None)
    if fld is None:
        return None
    used = {n["props"].get("text") for n, _ in walk(doc)}
    name = pick(rng, [x for x in g.FIELDS + ["Phone", "Job title", "Team size"] if x not in used])
    if name is None:
        return None
    after, f = mutate(doc)
    base = int(next_id(after)[1:])
    lab, inp = fld["children"]
    ph = inp["children"][0] if inp["children"] else None
    new_inp_kids = [] if ph is None else [{"id": f"n{base + 3}", "type": "text", "children": [],
                                           "props": {**ph["props"], "name": f"{name} placeholder",
                                                     "text": f"Enter {name.lower()}"}}]
    new = {"id": f"n{base}", "type": "frame", "props": {**fld["props"], "name": f"{name} field"}, "children": [
        {"id": f"n{base + 1}", "type": "text", "children": [], "props": {**lab["props"], "name": f"{name} label", "text": name}},
        {"id": f"n{base + 2}", "type": "frame", "children": new_inp_kids, "props": {**inp["props"], "name": f"{name} input"}}]}
    kids = f(form_["id"])["children"]
    kids.insert([c["id"] for c in kids].index(fld["id"]) + 1, new)
    instr = phrase(rng, "add a \u201c{n}\u201d field after the {o} field", "insert a {n} input below {o}, same style",
                   n=name, o=fld["props"]["name"].removesuffix(" field"))
    return EditExample(doc, after, instr, "add_field")


# --- v3 diversity edits: generic and explicit, so the model learns operations rather than one
# --- request -> one patch mappings. justify=between and "primary style" stay held out (NOVEL).

PROP_WORDS = {"size": "font size", "weight": "font weight", "color": "text color", "fill": "background",
              "stroke": "border color", "w": "width", "h": "height", "radius": "corner radius",
              "gap": "gap", "pad": "padding", "opacity": "opacity", "align": "cross-axis alignment",
              "justify": "main-axis alignment"}
PALETTE = ["#111827", "#374151", "#6B7280", "#FFFFFF", "#F3F4F6", "#E5E7EB", "#D1D5DB",
           *g.BRANDS.values(), "#CA8A04", "#0891B2", "#65A30D"]


def _prop_choices(n: Node, p: Node | None) -> dict[str, list]:
    t = n["type"]
    if t == "text":
        return {"size": g.TYPE_SCALE, "weight": [300, 400, 500, 600, 700, 800], "color": PALETTE}
    out: dict[str, list] = {"radius": g.RADII, "fill": PALETTE, "opacity": [0.2, 0.4, 0.5, 0.7, 0.9]}
    if t in ("rect", "image"):
        out |= {"w": [24, 36, 48, 64, 120, 200], "h": [16, 20, 24, 48, 120, 160]}
    if t == "frame" and p is not None:
        out |= {"stroke": PALETTE, "w": [240, 280, 320, 360, 400, 480]}
        if n["props"].get("layout") in ("row", "col"):
            out |= {"gap": g.SPACING, "pad": g.SPACING, "align": ["start", "center", "end"]}
        if n["props"].get("layout") == "row":
            out["justify"] = ["start", "center", "end"]
    return out


@forward
def set_prop(doc: Node, rng: random.Random) -> EditExample | None:
    n, p = pick(rng, nodes(doc, lambda n, p: p is not None)) or (None, None)
    if n is None:
        return None
    choices = _prop_choices(n, p)
    prop = rng.choice(sorted(choices))
    vals = [v for v in choices[prop] if v != n["props"].get(prop)]
    v = rng.choice(vals)
    after, f = mutate(doc)
    f(n["id"])["props"][prop] = v
    shown = f"{int(v * 100)}%" if prop == "opacity" else (f"{v}px" if prop in ("size", "w", "h", "radius", "gap", "pad") and rng.random() < 0.5 else v)
    instr = phrase(rng, "set the {w} of {l} to {v}", "change {l}'s {w} to {v}", "{l}: {w} {v}",
                   w=PROP_WORDS[prop], l=label(n), v=shown)
    return EditExample(doc, after, instr, "set_prop")


@forward
def unset_prop(doc: Node, rng: random.Random) -> EditExample | None:
    optional = ("stroke", "opacity", "radius", "fill")
    n, _ = pick(rng, nodes(doc, lambda n, p: p is not None and n["type"] != "text"
                           and any(k in n["props"] for k in optional))) or (None, None)
    if n is None:
        return None
    prop = rng.choice([k for k in optional if k in n["props"]])
    after, f = mutate(doc)
    del f(n["id"])["props"][prop]
    instr = phrase(rng, "remove the {w} from {l}", "clear {l}'s {w}", "{l} shouldn't have a {w}",
                   w={"stroke": "border", "fill": "background fill"}.get(prop, PROP_WORDS[prop]), l=label(n))
    return EditExample(doc, after, instr, "unset_prop")


@forward
def restyle_button(doc: Node, rng: random.Random) -> EditExample | None:
    """Explicit multi-part restyle: fill + label color + border, every part stated in the request."""
    btn, _ = pick(rng, nodes(doc, lambda n, p: is_button(n) and n["children"])) or (None, None)
    if btn is None:
        return None
    after, f = mutate(doc)
    nb = f(btn["id"])
    lab = nb["children"][0]
    if rng.random() < 0.6:
        word, hexv = rng.choice([kv for kv in COLOR_WORDS.items() if kv[0] not in ("gray", "yellow")
                                 and kv[1] != nb["props"].get("fill")])
        nb["props"]["fill"] = hexv
        lab["props"]["color"] = "#FFFFFF"
        had_border = nb["props"].pop("stroke", None) is not None
        instr = phrase(rng, "make {l} {w} with white text{b}", "restyle {l}: {w} background, white label{b}",
                       l=label(btn), w=word, b=" and no border" if had_border else "")
    else:
        if nb["props"].get("stroke") and nb["props"].get("fill") == "#FFFFFF":
            return None  # already an outline button: the request would be a no-op
        nb["props"]["fill"] = "#FFFFFF"
        nb["props"]["stroke"] = "#D1D5DB"
        lab["props"]["color"] = "#111827"
        instr = phrase(rng, "make {l} white with a gray border and dark text", "restyle {l} as an outline button: "
                       "white fill, #D1D5DB border, #111827 label", l=label(btn))
    return EditExample(doc, after, instr, "restyle_button")


@forward
def duplicate_node(doc: Node, rng: random.Random) -> EditExample | None:
    """Copy a node (with its subtree) right after itself: exercises subtree inserts and index arithmetic."""
    n, p = pick(rng, nodes(doc, lambda n, p: p is not None and p["id"] != doc["id"]
                           and p["props"].get("layout") in ("row", "col")
                           and sum(1 for _ in walk(n)) <= 8)) or (None, None)
    if n is None:
        return None
    after, f = mutate(doc)
    nxt = int(next_id(after)[1:])

    def clone(node: Node) -> Node:
        nonlocal nxt
        c = {"id": f"n{nxt}", "type": node["type"], "props": dict(node["props"]), "children": []}
        nxt += 1
        c["children"] = [clone(k) for k in node["children"]]
        return c

    kids = f(p["id"])["children"]
    kids.insert([k["id"] for k in kids].index(n["id"]) + 1, clone(n))
    instr = phrase(rng, "duplicate {l}", "make a copy of {l} right after it", "copy {l} and place the copy after it",
                   l=label(n))
    return EditExample(doc, after, instr, "duplicate_node")


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


@repair
def reset_opacity(doc: Node, rng: random.Random) -> EditExample | None:
    """Teaches ``unset``: a node was made semi-transparent by accident."""
    n, _ = pick(rng, nodes(doc, lambda n, p: p is not None and "opacity" not in n["props"])) or (None, None)
    if n is None:
        return None
    broken, f = mutate(doc)
    f(n["id"])["props"]["opacity"] = rng.choice([0.3, 0.5, 0.6, 0.8])
    instr = phrase(rng, "{l} looks washed out, make it fully opaque again", "remove the transparency on {l}",
                   "reset the opacity of {l}", l=label(n))
    return EditExample(broken, doc, instr, "reset_opacity")


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


# ---------------------------------------------------------------------------
# Novel edits: EVALUATION ONLY. Never used to build training data, so they
# measure whether the model learned the patch language or just the templates.
# ---------------------------------------------------------------------------

NOVEL: dict[str, EditFn] = {}


def novel(fn: EditFn) -> EditFn:
    NOVEL[fn.__name__] = fn
    return fn


@novel
def set_weight(doc: Node, rng: random.Random) -> EditExample | None:
    t, _ = pick(rng, nodes(doc, lambda n, p: n["type"] == "text" and n["props"].get("weight", 400) < 700
                           and not n["props"].get("name", "").endswith(" label"))) or (None, None)
    if t is None:
        return None
    after, f = mutate(doc)
    f(t["id"])["props"]["weight"] = 700
    instr = phrase(rng, "make {l} bold", "bold {l}", "{l} should be bold", l=label(t))
    return EditExample(doc, after, instr, "set_weight")


@novel
def add_border(doc: Node, rng: random.Random) -> EditExample | None:
    n, _ = pick(rng, nodes(doc, lambda n, p: n["type"] == "frame" and "stroke" not in n["props"]
                           and p is not None and not is_button(n))) or (None, None)
    if n is None:
        return None
    after, f = mutate(doc)
    f(n["id"])["props"]["stroke"] = "#E5E7EB"
    instr = phrase(rng, "add a subtle border around {l}", "give {l} a light outline", "put a thin border on {l}",
                   l=label(n))
    return EditExample(doc, after, instr, "add_border", [{"id": n["id"], "prop": "stroke", "cmp": "any"}])


@novel
def promote_button(doc: Node, rng: random.Random) -> EditExample | None:
    """Secondary -> primary style. Needs ``unset`` (drop the stroke), which is rare in training."""
    prim = [n for n, _ in walk(doc) if is_button(n) and "stroke" not in n["props"]]
    sec, _ = pick(rng, nodes(doc, lambda n, p: is_button(n) and "stroke" in n["props"] and n["children"])) or (None, None)
    if sec is None or not prim:
        return None
    brand = prim[0]["props"]["fill"]
    after, f = mutate(doc)
    nb = f(sec["id"])
    nb["props"]["fill"] = brand
    del nb["props"]["stroke"]
    nb["children"][0]["props"]["color"] = "#FFFFFF"
    instr = phrase(rng, "make {l} a primary button", "{l} should be styled like the main call to action, "
                   "same color as the other primary buttons", "promote {l} to primary style", l=label(sec))
    return EditExample(doc, after, instr, "promote_button",
                       [{"id": nb["children"][0]["id"], "prop": "color", "cmp": "contrast", "bg": brand, "min": 3.0}])


@novel
def set_width(doc: Node, rng: random.Random) -> EditExample | None:
    n, _ = pick(rng, nodes(doc, lambda n, p: n["type"] == "frame" and "w" in n["props"] and p is not None)) or (None, None)
    if n is None:
        return None
    w = rng.choice([v for v in (320, 400, 440, 480, 560, 640) if v != n["props"]["w"]])
    after, f = mutate(doc)
    f(n["id"])["props"]["w"] = w
    instr = phrase(rng, "make {l} {w}px wide", "set the width of {l} to {w}", "resize {l} to {w} wide", l=label(n), w=w)
    return EditExample(doc, after, instr, "set_width")


@novel
def darken_text(doc: Node, rng: random.Random) -> EditExample | None:
    t, _ = pick(rng, nodes(doc, lambda n, p: n["type"] == "text" and n["props"].get("color") == "#6B7280")) or (None, None)
    if t is None:
        return None
    after, f = mutate(doc)
    f(t["id"])["props"]["color"] = "#374151"
    instr = phrase(rng, "make {l} darker", "{l} is too faint, darken it", "bump up the contrast on {l}", l=label(t))
    return EditExample(doc, after, instr, "darken_text", [{"id": t["id"], "prop": "color", "cmp": "darker"}])


@novel
def spread_out(doc: Node, rng: random.Random) -> EditExample | None:
    n, _ = pick(rng, nodes(doc, lambda n, p: n["type"] == "frame" and n["props"].get("layout") == "row"
                           and len(n["children"]) >= 2 and n["props"].get("justify") != "between"
                           and not is_button(n))) or (None, None)
    if n is None:
        return None
    after, f = mutate(doc)
    f(n["id"])["props"]["justify"] = "between"
    instr = phrase(rng, "spread the items in {l} out to the edges", "push the contents of {l} apart (space between)",
                   l=label(n))
    return EditExample(doc, after, instr, "spread_out")


@novel
def duplicate_setting(doc: Node, rng: random.Random) -> EditExample | None:
    """Copy a settings row with a new label: a multi-node insert of a whole subtree."""
    row, parent = pick(rng, nodes(doc, lambda n, p: n["props"].get("name", "").endswith(" row")
                                  and len(n["children"]) == 2)) or (None, None)
    if row is None:
        return None
    used = {n["props"].get("text") for n, _ in walk(doc)}
    new = pick(rng, [s for s in g.SETTINGS if s not in used])
    if new is None:
        return None
    after, f = mutate(doc)
    base = int(next_id(after)[1:])
    lab, tog = row["children"]
    copy_row = {"id": f"n{base}", "type": "frame", "props": {**row["props"], "name": f"{new} row"}, "children": [
        {"id": f"n{base + 1}", "type": "text", "children": [], "props": {**lab["props"], "name": f"{new} label", "text": new}},
        {"id": f"n{base + 2}", "type": "rect", "children": [],
         "props": {**tog["props"], "name": f"{new} toggle", "fill": "#D1D5DB"}}]}
    kids = f(parent["id"])["children"]
    kids.insert([c["id"] for c in kids].index(row["id"]) + 1, copy_row)
    old = lab["props"]["text"]
    instr = phrase(rng, "add a \u201c{n}\u201d setting below \u201c{o}\u201d, styled the same, toggled off",
                   "duplicate the \u201c{o}\u201d row underneath it, rename it \u201c{n}\u201d and switch it off",
                   n=new, o=old)
    return EditExample(doc, after, instr, "duplicate_setting")


ALL_EDITS: dict[str, EditFn] = {**FORWARD, **REPAIR, "compose": compose}  # training kinds only
