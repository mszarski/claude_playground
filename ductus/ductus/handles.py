"""Primitive handles (SPEC 7): draggable controls for primitive *parameters*
(serif length, join height, notch depth, taper position...), not points.

Primitives call :func:`handle` while a glyph is being built. A handle only
exists if the value driving it is a plain parameter (or a parameter times a
constant), so dragging can be turned into a parameter change. Dragging
writes a per-glyph parameter override, ``<glyph>.@<param>: {dv: ...}``,
which changes the parameter for that glyph only.
"""
from __future__ import annotations

import contextvars
import math
from contextlib import contextmanager
from typing import Any

from .params import BinOp, Param, val

_sink: contextvars.ContextVar[list | None] = contextvars.ContextVar("ductus_handles", default=None)


def param_of(x: Any) -> tuple[Param, float] | None:
    """``x`` as (param, factor) if it is ``param`` or ``param * c`` /
    ``c * param`` / ``param / c`` with a constant ``c``; else None."""
    if isinstance(x, Param):
        return x, 1.0
    if isinstance(x, BinOp) and x.sym in ("*", "/"):
        a, b = x.a, x.b
        if isinstance(a, Param) and not hasattr(b, "_eval"):
            return a, (float(b) if x.sym == "*" else 1.0 / float(b))
        if x.sym == "*" and isinstance(b, Param) and not hasattr(a, "_eval"):
            return b, float(a)
    return None


@contextmanager
def collecting():
    token = _sink.set([])
    try:
        yield _sink.get()
    finally:
        _sink.reset(token)


def handle(expr, anchor, axis, rate: float = 1.0, label: str | None = None):
    """Register a handle for ``expr`` at ``anchor`` that moves along ``axis``
    (a direction; normalised here) by ``rate`` font units per unit of
    ``expr``. No-op outside a glyph build or if ``expr`` isn't a parameter."""
    sink = _sink.get()
    if sink is None:
        return
    pf = param_of(expr)
    if pf is None:
        return
    p, factor = pf
    ax, ay = float(axis[0]), float(axis[1])
    n = math.hypot(ax, ay)
    per_unit = factor * rate
    if n < 1e-9 or abs(per_unit) < 1e-9 or p.name is None:
        return
    sink.append({"param": p.name, "x": float(anchor[0]), "y": float(anchor[1]),
                 "ax": ax / n, "ay": ay / n, "per_unit": per_unit, "value": val(p),
                 "range": list(p.range) if p.range else None, "label": label or p.name})
