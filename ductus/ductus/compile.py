"""The compiler pipeline: ``compile_font(font, params, overrides) -> CompiledFont``.

A pure function of its inputs. The result is plain data (``to_json()``)
plus a few internal handles used by lints.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pathops
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.recordingPen import RecordingPen

from .geometry import Contour, bez, flatten, point_in_polygon, svg_path
from .overrides import Override
from .params import evaluating, val
from .parts import DuctusError, Font, GlyphBuilder, GlyphDef, realize
from .pen import expand_stroke
from .skeleton import Stroke


@dataclass
class CompiledGlyph:
    name: str
    unicode: int | None
    advance: int
    contours: list[Contour]                # final: merged, oriented, rounded, spaced
    bounds: tuple | None
    sides: tuple[str, str]
    skeleton: list[dict]                   # strokes for display (spaced coords)
    points: dict[str, list[float]]         # named skeleton points (spaced coords)
    outline_points: dict[str, list[float]] # named expanded points, pre-merge (spaced)
    raw_signature: str                     # pre-merge structure (compatible by construction)
    signature: str                         # post-merge structure
    shift: float = 0.0
    strokes: list[Stroke] = field(default_factory=list, repr=False)

    def to_json(self) -> dict:
        return {
            "name": self.name, "unicode": self.unicode, "advance": self.advance,
            "bounds": self.bounds, "sides": self.sides,
            "svg": svg_path(self.contours),
            "contours": [c.to_json() for c in self.contours],
            "skeleton": self.skeleton, "points": self.points,
            "outline_points": self.outline_points,
        }


@dataclass
class CompiledFont:
    family: str
    style: str
    upm: int
    metrics: dict[str, float]
    params: dict[str, float]
    param_specs: dict[str, dict]
    glyphs: dict[str, CompiledGlyph]
    warnings: list[str]
    overrides_used: list[str]
    compile_ms: float = 0.0

    def to_json(self) -> dict:
        return {
            "family": self.family, "style": self.style, "upm": self.upm,
            "metrics": self.metrics, "params": self.params, "param_specs": self.param_specs,
            "glyphs": {k: g.to_json() for k, g in self.glyphs.items()},
            "warnings": self.warnings, "compile_ms": round(self.compile_ms, 1),
        }


def resolve_params(font: Font, params: dict[str, float] | None) -> dict[str, float]:
    values = {n: p.default for n, p in font.params.items()}
    for k, v in (params or {}).items():
        if k not in values:
            raise DuctusError(f"unknown parameter {k!r} (have {sorted(values)})")
        values[k] = float(v)
    return values


def compile_font(font: Font, params: dict[str, float] | None = None,
                 overrides: dict[str, Override] | None = None,
                 terminals=None, glyphs: list[str] | None = None) -> CompiledFont:
    t0 = time.perf_counter()
    values = resolve_params(font, params)
    overrides = overrides or {}
    used: set[str] = set()
    warnings: list[str] = []
    out: dict[str, CompiledGlyph] = {}
    with evaluating(values):
        metrics = {k: val(getattr(font, k)) for k in ("ascender", "descender", "xheight", "capheight")}
        for gname, gdef in font.glyphs.items():
            if glyphs is not None and gname not in glyphs:
                continue
            try:
                out[gname] = _compile_glyph(font, gdef, overrides, used, terminals)
            except DuctusError:
                raise
            except Exception as e:  # make the failing glyph obvious
                raise DuctusError(f"glyph {gname!r}: {type(e).__name__}: {e}") from e
    if glyphs is None:
        for name in overrides:
            if name not in used:
                warnings.append(f"override {name!r} does not match any point; ignored")
    specs = {n: {"default": p.default, "range": p.range} for n, p in font.params.items()}
    return CompiledFont(font.family, font.style, font.upm, metrics, values, specs, out,
                        warnings, sorted(used), (time.perf_counter() - t0) * 1000)


def _move(c: Contour, i: int, dx: float, dy: float):
    """Move point ``i`` and the off-curve handles attached to it."""
    n = len(c.pts)
    for j in (i, (i - 1) % n, (i + 1) % n):
        if j == i or c.types[j] == "off":
            c.pts[j][0] += dx
            c.pts[j][1] += dy


def _orient(group: list[Contour]) -> list[Contour]:
    """Even nesting depth -> counter-clockwise, odd -> clockwise."""
    polys = [flatten(c, 8) for c in group]
    res = []
    for i, c in enumerate(group):
        depth = sum(point_in_polygon(c.pts[0], polys[j])
                    for j in range(len(group)) if j != i)
        ccw = c.area() > 0
        res.append(c if ccw == (depth % 2 == 0) else c.reversed())
    return res


def _compile_glyph(font: Font, gdef: GlyphDef, overrides, used: set, terminals) -> CompiledGlyph:
    gb = GlyphBuilder(gdef, font)
    gdef.fn(gb)
    r = realize(gb)

    # 3. skeleton overrides
    nodes = {n.fullname: n for s in r.strokes for n in s.nodes}
    raw_points = {nm: (cb.contour, i) for cb in r.contours for i, nm in enumerate(cb.contour.names) if nm}
    for name, o in overrides.items():
        if o.stage != "skeleton" or not name.startswith(gdef.name + "."):
            continue
        if name in nodes:
            nodes[name].pos = nodes[name].pos + np.array([o.dx, o.dy])
            used.add(name)
        elif name in raw_points:
            _move(*raw_points[name], o.dx, o.dy)
            used.add(name)

    # 4. expand strokes
    groups: list[list[Contour]] = []
    ends = []
    for s in r.strokes:
        e = expand_stroke(s, font.pen)
        groups.append(e.contours)
        ends += e.ends
    for cb in r.contours:
        groups.append([cb.contour.copy()])

    # 5. terminals
    style = terminals or gb.terminal_style or font.terminals
    for end in ends:
        if end.kind:
            for c in style(end):
                groups.append([c])

    # 6. outline overrides
    for name, o in overrides.items():
        if o.stage != "outline" or not name.startswith(gdef.name + "."):
            continue
        hit = False
        for grp in groups:
            for c in grp:
                for i, nm in enumerate(c.names):
                    if nm and (nm == name or nm in (name + ".l", name + ".r")):
                        _move(c, i, o.dx, o.dy)
                        hit = True
        if hit:
            used.add(name)

    groups = [_orient(g) for g in groups]
    flat = [c for g in groups for c in g]
    raw_signature = "|".join(c.signature() for c in flat)

    # 7. remove overlaps, fix direction (outer CCW = PostScript convention), round
    final = merge(flat)

    # 8. spacing
    bounds = _bounds(final)
    sides = gdef.sides or (gb.inherited.sides if gb.inherited else None) or _detect_sides(r.strokes)
    adv_spec = gdef.advance if gdef.advance is not None else (gb.inherited.advance if gb.inherited else None)
    if bounds is None:
        shift = 0.0
        advance = round(val(adv_spec)) if adv_spec is not None else 0
    else:
        lsb = val(font.sidebearings[sides[0]])
        rsb = val(font.sidebearings[sides[1]])
        shift = float(round(lsb - bounds[0]))
        advance = round(val(adv_spec)) if adv_spec is not None else round(bounds[2] - bounds[0] + lsb + rsb)
        for c in final:
            for p in c.pts:
                p[0] += shift
        bounds = (bounds[0] + shift, bounds[1], bounds[2] + shift, bounds[3])

    points = {n.fullname: [round(float(n.pos[0]) + shift, 2), round(float(n.pos[1]), 2)]
              for s in r.strokes for n in s.nodes}
    outline_points = {}
    for c in flat:
        for p, nm in zip(c.pts, c.names):
            if nm:
                outline_points[nm] = [round(p[0] + shift, 2), round(p[1], 2)]
    for nm, (c, i) in raw_points.items():
        points[nm] = [round(c.pts[i][0] + shift, 2), round(c.pts[i][1], 2)]

    return CompiledGlyph(
        name=gdef.name, unicode=gdef.unicode, advance=int(advance), contours=final,
        bounds=bounds, sides=tuple(sides),
        skeleton=[s.to_json(shift) for s in r.strokes], points=points,
        outline_points=outline_points, raw_signature=raw_signature,
        signature="|".join(c.signature() for c in final), shift=shift, strokes=r.strokes,
    )


def merge(contours: list[Contour], round_to_grid: bool = True) -> list[Contour]:
    """Union with skia-pathops; returns outer-CCW contours rounded to integers."""
    if not contours:
        return []
    path = pathops.Path()
    pen = path.getPen()
    for c in contours:
        c.draw(pen)
    path.simplify(fix_winding=True, keep_starting_points=True, clockwise=False)
    rp = RecordingPen()
    path.draw(rp)
    out = Contour.from_ops(rp.value)
    if round_to_grid:
        for c in out:
            for p in c.pts:
                p[0] = float(round(p[0]))
                p[1] = float(round(p[1]))
    return out


def _bounds(contours: list[Contour]):
    if not contours:
        return None
    bp = BoundsPen(None)
    for c in contours:
        c.draw(bp)
    return bp.bounds


def _detect_sides(strokes: list[Stroke]) -> tuple[str, str]:
    """A side is 'straight' if a vertical line segment of the skeleton sits
    at that side's extreme, otherwise 'round'."""
    if not strokes:
        return ("straight", "straight")
    xs = []
    verticals = []
    for s in strokes:
        for P, kind in s.cubics():
            xs.append(bez(P, np.linspace(0, 1, 9))[:, 0])
            if kind == "line" and abs(P[3][0] - P[0][0]) < 0.5 and abs(P[3][1] - P[0][1]) > 1:
                verticals.append(P[0][0])
    xs = np.concatenate(xs)
    lo, hi = xs.min(), xs.max()
    left = "straight" if any(abs(v - lo) < 1 for v in verticals) else "round"
    right = "straight" if any(abs(v - hi) < 1 for v in verticals) else "round"
    return (left, right)
