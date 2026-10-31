"""Terminal transforms applied to stroke ends tagged as terminals.

A terminal style is a callable ``style(end: StrokeEnd) -> list[Contour]``
returning extra contours to union with the glyph. Strokes always end with a
flat cap; ``serifs.none`` adds nothing.
"""
from __future__ import annotations

import numpy as np

from .geometry import Contour
from .params import val


def _none(end):
    return []


def _slab(length=None, thickness=None, where=None):
    """Rectangular slab centred on the end, flush with the flat cap,
    ``thickness`` deep into the stroke, extending ``length`` past each side.
    ``where`` optionally restricts it to ends whose terminal tag is in it."""
    def style(end):
        if where is not None and end.kind not in where:
            return []
        from .parts import current_font
        pen = current_font().pen
        L = val(length) if length is not None else val(pen.width) * 0.45
        T = val(thickness) if thickness is not None else pen.hairline * 0.75
        d = end.outward
        n = np.array([-d[1], d[0]])
        half = end.width / 2 + L
        p = end.pos
        c = Contour()
        for corner, nm in ((p - n * half, "a"), (p + n * half, "b"),
                           (p + n * half - d * T, "c"), (p - n * half - d * T, "d")):
            c.add(corner, "on", f"{end.name}.serif.{nm}")
        return [c]
    style.__name__ = "slab"
    return style


class serifs:
    none = staticmethod(_none)
    slab = staticmethod(_slab)
