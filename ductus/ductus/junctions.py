"""Junctions (SPEC 5.3): how a curved part leaves (or enters) a stem.

One :class:`Join` describes the meeting of a stroke or bowl with a stem's
edge, and is used by arches (n h m r u) and by stem bowls / lobes (a d q,
b p). It fixes, in stem-relative terms:

* ``at``     -- height of the crotch: where the part's *inner* edge meets the
               stem edge;
* ``thin``   -- thickness of the part where it meets the stem (hairline
               thinning), measured across the part;
* ``angle``  -- direction the part leaves the stem in (degrees from the
               horizontal, positive = upwards, towards the part's side);
* ``notch``  -- optional depth of a notch cut into the stem at the crotch,
               ``notch_w`` its height along the stem.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .handles import handle
from .params import val


@dataclass
class Join:
    at: Any
    thin: Any
    angle: Any = 30
    notch: Any = 0
    notch_w: Any = 20

    def resolved(self) -> tuple[float, float, float, float, float]:
        return val(self.at), val(self.thin), val(self.angle), val(self.notch), val(self.notch_w)


def join(at, thin, angle=30, notch=0, notch_w=20) -> Join:
    return Join(at, thin, angle, notch, notch_w)


def contact(J: Join, edge_x: float, side: int, rising: bool = True):
    """Geometry of a part meeting the stem edge at ``edge_x``.

    ``side`` = +1 if the part lies to the right of the edge, -1 to the left.
    ``rising`` = the part's other edge lies *above* the crotch (arch tops,
    the top of a lobe); False for joins whose other edge is below (bottom
    of a lobe, the cup of u).

    Returns (crotch, outer, start, direction_deg, thin): the crotch point on
    the edge, the part's outer contact on the edge, a stroke start point on
    the part's centre line slightly inside the stem, the direction of travel
    leaving the stem, and the thickness there.
    """
    at, thin, ang, _, _ = J.resolved()
    a = math.radians(ang)
    sgn = 1 if rising else -1
    # leaving direction: away from the stem on ``side``, tilted by ``angle``
    dx, dy = side * math.cos(a), math.sin(a) * sgn
    # vertical span along the stem edge of a band of thickness ``thin``
    span = thin / max(math.cos(a), 0.2)
    crotch = (edge_x, at)
    outer = (edge_x, at + sgn * span)
    mid = (edge_x, at + sgn * span / 2)
    inset = max(thin, 8)
    start = (mid[0] - dx * inset, mid[1] - dy * inset)
    direction = math.degrees(math.atan2(dy, dx))
    return crotch, outer, start, direction, thin


def notch_cut(builder, J: Join, edge_x: float, side: int, name: str = "notch"):
    """Cut a small triangular notch into the stem at the crotch (if any)."""
    at, _, _, depth, w = J.resolved()
    handle(J.notch, (edge_x - side * max(depth, 0.0), at), (-side, 0), label="notch depth")
    cb = builder.cut(name)
    d = max(depth, 0.0)
    # with depth 0 the triangle is degenerate (zero area) but keeps the structure
    cb.move((edge_x + side * 0.5, at + w / 2), "a").line((edge_x - side * d, at), "tip") \
        .line((edge_x + side * 0.5, at - w / 2), "b").close()
    return cb


def join_handles(J: Join, edge_x: float, side: int, rising: bool = True):
    """Handles for a join's crotch height and entry thickness."""
    crotch, outer, _, _, _ = contact(J, edge_x, side, rising)
    a = math.radians(val(J.angle))
    sgn = 1 if rising else -1
    handle(J.at, crotch, (0, 1), label="join height")
    handle(J.thin, outer, (0, sgn), rate=1 / max(math.cos(a), 0.2), label="join thickness")
