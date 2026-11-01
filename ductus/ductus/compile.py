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
    lsb: float = 0.0
    rsb: float = 0.0
    primitives: list[str] = field(default_factory=list)
    stages: dict = field(default_factory=dict)            # per-stage debug geometry (svg paths)
    handles: list = field(default_factory=list)           # primitive handles (spaced coords)
    strokes: list[Stroke] = field(default_factory=list, repr=False)

    def to_json(self) -> dict:
        return {
            "name": self.name, "unicode": self.unicode, "advance": self.advance,
            "bounds": self.bounds, "sides": self.sides,
            "shift": self.shift, "lsb": self.lsb, "rsb": self.rsb, "primitives": self.primitives,
            "stages": {k: svg_path(translated_contours(v, self.shift)) for k, v in self.stages.items() if v},
            "handles": [{**h, "x": round(h["x"] + self.shift, 2), "y": round(h["y"], 2)} for h in self.handles],
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
            # per-glyph parameter overrides: ``<glyph>.@<param>: {dv: ...}``
            local = {}
            for oname, o in overrides.items():
                if oname.startswith(gname + ".@"):
                    pname = oname[len(gname) + 2:]
                    if pname in values:
                        local[pname] = values[pname] + o.dv
                        used.add(oname)
            try:
                if local:
                    with evaluating({**values, **local}):
                        out[gname] = _compile_glyph(font, gdef, overrides, used, terminals)
                else:
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
        if getattr(c, "built_cw", False):        # open stroke: known winding
            res.append(c.reversed() if depth % 2 == 0 else c)
            continue
        a = c.area()
        if abs(a) < 1.0:            # degenerate (e.g. a collapsed wedge): keep as built,
            res.append(c)           # flipping it would change the point order
            continue
        ccw = a > 0
        res.append(c if ccw == (depth % 2 == 0) else c.reversed())
    return res


def _compile_glyph(font: Font, gdef: GlyphDef, overrides, used: set, terminals) -> CompiledGlyph:
    from .handles import collecting
    with collecting() as handles:
        return _compile_glyph_inner(font, gdef, overrides, used, terminals, handles)


def _compile_glyph_inner(font, gdef, overrides, used, terminals, handles) -> CompiledGlyph:
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
            nd = nodes[name]
            nd.pos = nd.pos + np.array([o.dx, o.dy])
            nd.dw += o.dw
            nd.da += o.da
            nd.dpa += o.dp
            nd.tension += o.dt
            used.add(name)
        elif name in raw_points:
            _move(*raw_points[name], o.dx, o.dy)
            used.add(name)

    # 4. expand strokes
    groups: list[list[Contour]] = []
    ends = []
    stage = {"strokes": [], "contours": [], "terminals": [], "cuts": []}
    for s in r.strokes:
        e = expand_stroke(s, font.pen)
        if not s.closed:
            # an open stroke's outline is built clockwise (left side forward,
            # right side back): orient it by construction, not by its signed
            # area, which flips when an extreme setting twists the outline
            for c in e.contours:
                c.built_cw = True
        groups.append(e.contours)
        stage["strokes"] += e.contours
        ends += e.ends
    by_group: dict[int, list[Contour]] = {}
    cuts: list[Contour] = []
    for cb in r.contours:
        if cb.negative:
            cuts.append(cb.contour.copy())
            continue
        stage["contours"].append(cb.contour)
        if cb.group is None:
            groups.append([cb.contour.copy()])
        else:
            key = id(cb.group)
            if key not in by_group:
                by_group[key] = []
                groups.append(by_group[key])
            by_group[key].append(cb.contour.copy())

    # 5. terminals
    style = terminals or gb.terminal_style or font.terminals
    prims = set()
    from .terminals import TerminalSet, as_terminal
    for end in ends:
        if end.kind:
            st = style.style_for(end.kind) if isinstance(style, TerminalSet) else style
            term = as_terminal(st(end))
            if term.add or term.cut:
                prims.add(f"terminal:{getattr(st, '__name__', 'custom')}")
            for c in term.add:
                groups.append([c])
            stage["terminals"] += term.add
            cuts += term.cut
    prims |= {f"part:{n}" for n in r.parts}
    prims |= {f"contour:{cb.kind}" for cb in r.contours}
    if r.strokes:
        prims.add("stroke")

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
    # cuts are not reoriented: pathops normalises them on their own, and
    # flipping them would make their point order depend on values
    raw_signature = "|".join(c.signature() for c in flat) + "".join("|-" + c.signature() for c in cuts)

    stage["cuts"] = list(cuts)
    # 7. remove overlaps, subtract cuts, fix direction (outer CCW), round
    final = merge(flat, cuts=cuts)

    # 8. spacing
    bounds = _bounds(final)
    sides = gdef.sides or (gb.inherited.sides if gb.inherited else None) or \
        _detect_sides(r.strokes, stage["contours"])   # terminals (serifs) don't decide the side
    adv_spec = gdef.advance if gdef.advance is not None else (gb.inherited.advance if gb.inherited else None)
    if bounds is None:
        shift = 0.0
        advance = round(val(adv_spec)) if adv_spec is not None else 0
    else:
        lsb = val(font.sidebearings[sides[0]])
        rsb = val(font.sidebearings[sides[1]])
        if gdef.sb is not None:
            lsb += val(gdef.sb[0]); rsb += val(gdef.sb[1])
        for key in ("lsb", "rsb"):
            o = overrides.get(f"{gdef.name}.{key}")
            if o is not None:
                used.add(o.name)
                if key == "lsb":
                    lsb += o.dx
                else:
                    rsb += o.dx
        shift = float(round(lsb - bounds[0]))
        advance = round(val(adv_spec)) if adv_spec is not None else round(bounds[2] - bounds[0] + lsb + rsb)
        for c in final:
            for p in c.pts:
                p[0] += shift
        bounds = (bounds[0] + shift, bounds[1], bounds[2] + shift, bounds[3])
    if bounds is None:
        lsb = rsb = 0.0

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
        skeleton=[s.to_json(shift, font.pen) for s in r.strokes], points=points,
        outline_points=outline_points, raw_signature=raw_signature,
        signature="|".join(c.signature() for c in final), shift=shift,
        lsb=round(lsb, 2), rsb=round(rsb, 2), primitives=sorted(prims),
        stages=stage, handles=list(handles), strokes=r.strokes,
    )


def _path(contours: list[Contour]):
    path = pathops.Path()
    pen = path.getPen()
    for c in contours:
        c.draw(pen)
    return path


def merge(contours: list[Contour], round_to_grid: bool = True, cuts: list[Contour] | None = None) -> list[Contour]:
    """Union with skia-pathops (minus ``cuts``); returns outer-CCW contours,
    rounded to integers."""
    if not contours:
        return []
    path = _path(contours)
    path.simplify(fix_winding=True, keep_starting_points=True, clockwise=False)
    if cuts:
        cp = _path(cuts)
        cp.simplify(fix_winding=True, keep_starting_points=True, clockwise=False)
        path = pathops.op(path, cp, pathops.PathOp.DIFFERENCE, fix_winding=True,
                          keep_starting_points=True, clockwise=False)
    rp = RecordingPen()
    path.draw(rp)
    out = Contour.from_ops(rp.value)
    if round_to_grid:
        for c in out:
            for p in c.pts:
                p[0] = float(round(p[0]))
                p[1] = float(round(p[1]))
        # slivers that rounding collapsed onto a point or a line
        out = [c for c in out if abs(c.area()) >= 4.0]      # no real contour is 2x2 u
        out = [_drop_spikes(c) for c in out]
    return out


def _drop_spikes(c: Contour, max_len: float = 12.0, min_turn_deg: float = 170.0) -> Contour:
    """Remove needle spikes boolean ops leave where two edges meet almost
    tangentially: an on-curve point where the outline turns back on itself
    (> ``min_turn_deg``) next to a short segment (< ``max_len``). Real tips (serifs, beaks, vertices) turn far less."""
    import math as _m
    pts, types, names = [list(p) for p in c.pts], list(c.types), list(c.names)
    changed = True
    while changed and len(pts) > 4:
        changed = False
        n = len(pts)
        for i in range(n):
            if types[i] != "on" or types[(i + 1) % n] != "on":
                continue
            a, b, d = pts[i - 1], pts[i], pts[(i + 1) % n]
            vin = (b[0] - a[0], b[1] - a[1]); vout = (d[0] - b[0], d[1] - b[1])
            lin, lout = _m.hypot(*vin), _m.hypot(*vout)
            if lin < 1e-9 or lout < 1e-9 or min(lin, lout) > max_len:
                continue
            cosang = (vin[0] * vout[0] + vin[1] * vout[1]) / (lin * lout)
            if cosang < _m.cos(_m.radians(min_turn_deg)):
                del pts[i], types[i], names[i]
                changed = True
                break
    out = Contour()
    for p, t, nm in zip(pts, types, names):
        out.add(p, t, nm)
    return out


def translated_contours(contours: list[Contour], dx: float) -> list[Contour]:
    out = []
    for c in contours:
        cc = c.copy()
        for p in cc.pts:
            p[0] += dx
        out.append(cc)
    return out


def _bounds(contours: list[Contour]):
    if not contours:
        return None
    bp = BoundsPen(None)
    for c in contours:
        c.draw(bp)
    return bp.bounds


def _detect_sides(strokes: list[Stroke], contours: list[Contour] = ()) -> tuple[str, str]:
    """A side is 'straight' if a vertical line -- a skeleton line segment or
    a straight contour edge -- sits at that side's extreme, else 'round'."""
    xs = []
    verticals = []
    for s in strokes:
        for P, kind in s.cubics():
            xs.append(bez(P, np.linspace(0, 1, 9))[:, 0])
            if kind == "line" and abs(P[3][0] - P[0][0]) < 0.5 and abs(P[3][1] - P[0][1]) > 1:
                verticals.append(P[0][0])
    for c in contours:
        poly = flatten(c, 8)
        xs.append(poly[:, 0])
        n = len(c.pts)
        for i in range(n):
            j = (i + 1) % n
            if c.types[i] == "on" and c.types[j] == "on":
                (x0, y0), (x1, y1) = c.pts[i], c.pts[j]
                if abs(x1 - x0) < 0.5 and abs(y1 - y0) > 20:
                    verticals.append(x0)
    if not xs:
        return ("straight", "straight")
    xs = np.concatenate(xs)
    lo, hi = xs.min(), xs.max()
    left = "straight" if any(abs(v - lo) < 8 for v in verticals) else "round"
    right = "straight" if any(abs(v - hi) < 8 for v in verticals) else "round"
    return (left, right)
