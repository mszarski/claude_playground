"""Detect training rows whose *effect* resembles a held-out edit type (edits.NOVEL / edits.GOAL).

The request-text filter in distill_goals.HELD_OUT misses rows where the request says something else ("make the
Pro card feel more premium") but the teacher's patch does exactly a held-out edit (adds a border, bolds text).
Training on those would quietly teach the held-out conventions, so teacher-written rows are also checked here on
the before/after documents.
"""

from __future__ import annotations

from .edits import is_button
from .rewards import _luminance
from .schema import Node, walk

GREYS = {"#E5E7EB", "#D1D5DB", "#9CA3AF", "#F3F4F6", "#E5E5E5", "#CCCCCC", "#D4D4D4"}


def held_out_signature(before: Node, after: Node) -> str | None:
    """Name of the held-out edit type this change looks like, or None."""
    old = {n["id"]: n for n, _ in walk(before)}
    for n, _ in walk(after):
        o = old.get(n["id"])
        p = n["props"]
        if o is None:
            if n["type"] == "frame" and p.get("name", "").endswith(" row") and n["children"]:
                return "duplicate_setting"
            continue
        q = o["props"]
        if p.get("justify") == "between" and q.get("justify") != "between":
            return "spread_out"
        if "w" in p and p.get("w") != q.get("w"):
            return "set_width"
        if p.get("stroke") and not q.get("stroke"):
            return "add_border"
        if q.get("name", "").endswith(" plan") and p.get("stroke") != q.get("stroke"):
            return "highlight_plan"
        if n["type"] == "text" and p.get("weight", 400) > q.get("weight", 400) and p.get("weight", 400) >= 600:
            return "set_weight"
        if (n["type"] == "text" and p.get("color") != q.get("color") and isinstance(p.get("color"), str)
                and isinstance(q.get("color"), str) and _luminance(p["color"]) < _luminance(q["color"])
                and _luminance(q["color"]) < 0.5):
            return "darken_text"  # an already dark-grey text made darker (fixing white-on-white is still allowed)
        if q.get("name", "").endswith(" toggle") and p.get("fill") != q.get("fill"):
            return "toggle_on"
        if is_button(o):
            if q.get("stroke") and not p.get("stroke") and p.get("fill") != q.get("fill"):
                return "promote_button"
            if p.get("opacity", 1) < q.get("opacity", 1) or (p.get("fill") in GREYS and q.get("fill") not in GREYS):
                return "disable_button"
    return None


KNOWN_PROPS = {"name", "text", "layout", "gap", "pad", "align", "justify", "size", "weight", "color", "fill", "stroke",
               "radius", "opacity", "w", "h", "x", "y"}
ENUMS = {"layout": {"row", "col"}, "align": {"start", "end", "center", "stretch"},
         "justify": {"start", "end", "center", "between"}}
_HEX = __import__("re").compile(r"^#[0-9A-Fa-f]{6}$")


def dsl_violations(before: Node, after: Node) -> list[str]:
    """Properties the patch introduced that the design language (and renderer) doesn't support."""
    old = {n["id"]: n["props"] for n, _ in walk(before)}
    out = []
    for n, _ in walk(after):
        q = old.get(n["id"], {})
        for k, v in n["props"].items():
            if q.get(k) == v:
                continue
            if k not in KNOWN_PROPS:
                out.append(f"{k} is not a property of this design language")
            elif k in ("color", "fill", "stroke") and not (isinstance(v, str) and _HEX.match(v)):
                out.append(f"{k}={v} is not a 6-digit hex color")
            elif k in ENUMS and v not in ENUMS[k]:
                out.append(f"{k}={v} is not one of {'|'.join(sorted(ENUMS[k]))}")
    return out
