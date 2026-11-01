"""The DSL: ``Font``, ``@part``, ``@glyph`` and the builders they run with.

Naming rules (principle 2):

* A glyph is a scope named after the glyph: ``n``.
* Each part instance is a scope inside it, named by ``name=`` or, by
  default, by the part function's name without trailing underscores:
  ``n.arch``. Nested part instances nest: ``x.outer.inner``.
* A stroke or raw contour drawn inside a part is named ``scope.stroke``.
  If a part draws exactly one item (and calls no sub-parts), the item takes
  the part instance's name directly -- so ``arch(name="arch2")`` gives
  ``m.arch2``, not ``m.arch2.arch``.
* Points are ``<stroke>.<node>``: ``n.arch.shoulder``. Expanded outline
  points add ``.l`` / ``.r`` (left/right of the direction of travel).
"""
from __future__ import annotations

import copy
import functools
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .geometry import Contour
from .params import pt, val
from .skeleton import Node, Stroke


class DuctusError(Exception):
    pass


_current_font: "Font | None" = None


def current_font() -> "Font":
    if _current_font is None:
        raise DuctusError("no Font() has been created yet")
    return _current_font


# --------------------------------------------------------------------------

@dataclass
class GlyphDef:
    name: str
    fn: Callable
    unicode: int | None
    advance: Any = None
    sides: tuple[str, str] | None = None   # ("straight"|"round", ...) or None = detect
    sb: tuple | None = None                # additive (left, right) sidebearing adjustments


class Font:
    def __init__(self, family: str, upm: int = 1000, style: str = "Regular",
                 ascender=750, descender=-250, xheight=500, capheight=700):
        global _current_font
        self.family = family
        self.style = style
        self.upm = upm
        self.ascender = ascender
        self.descender = descender
        self.xheight = xheight
        self.capheight = capheight
        self.pen = None
        from .terminals import serifs
        self.terminals = serifs.none
        # sidebearings by side type; numbers, Exprs or zero-arg callables
        self.sidebearings: dict[str, Any] = {
            "straight": lambda: val(self.pen.width) * 0.5 + 30,
            "round": lambda: (val(self.pen.width) * 0.5 + 30) * 0.6,
        }
        # per-glyph spacing: {glyph: (left, right)} added to that glyph's
        # sidebearings (numbers or Exprs); kerning: {(left, right): value}
        self.spacing: dict[str, tuple] = {}
        self.kerning: dict[tuple[str, str], Any] = {}
        self.glyphs: dict[str, GlyphDef] = {}
        self.params: dict = {}          # filled in by the loader
        self.source_path = None
        _current_font = self

    def kern(self, left: str, right: str, value):
        """Kern the pair ``left`` ``right`` by ``value`` font units (negative = closer)."""
        self.kerning[(left, right)] = value

    def glyph(self, name: str, unicode: int | None = None, advance=None, sides=None, sb=None):
        def deco(fn):
            u = unicode if unicode is not None else _unicode_for(name)
            self.glyphs[name] = GlyphDef(name, fn, u, advance, sides, sb)
            return fn
        return deco


def _unicode_for(name: str) -> int | None:
    if len(name) == 1:
        return ord(name)
    from fontTools.agl import toUnicode
    s = toUnicode(name)
    return ord(s) if len(s) == 1 else None


def glyph(name: str, unicode: int | None = None, advance=None, sides=None):
    """Register a glyph on the most recently created Font."""
    return current_font().glyph(name, unicode, advance, sides)


# --------------------------------------------------------------------------
# parts

class PartCall:
    """A deferred part invocation. Its arguments stay editable until the
    glyph is realized, so ``g["stem"].top = ascender`` works after
    ``g.inherit("n")``."""

    def __init__(self, fn: Callable, kwargs: dict, name: str):
        object.__setattr__(self, "_fn", fn)
        object.__setattr__(self, "_kwargs", kwargs)
        object.__setattr__(self, "name", name)

    def __getattr__(self, k):
        try:
            return self._kwargs[k]
        except KeyError:
            raise AttributeError(k) from None

    def __setattr__(self, k, v):
        if k == "name":
            object.__setattr__(self, k, v)
        else:
            self._kwargs[k] = v

    def __repr__(self):
        return f"<part {self._fn.__name__} as {self.name!r}>"


def part(fn: Callable) -> Callable[..., PartCall]:
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())[1:]  # drop the builder arg
    default_name = fn.__name__.strip("_") or fn.__name__

    @functools.wraps(fn)
    def make(*args, name: str | None = None, **kwargs) -> PartCall:
        bound = inspect.Signature(params).bind(*args, **kwargs)
        return PartCall(fn, dict(bound.arguments), name or default_name)

    make.__ductus_part__ = True
    return make


# --------------------------------------------------------------------------
# builders

class PathBuilder:
    def __init__(self, stroke: Stroke):
        self.stroke = stroke

    def _node(self, p, name, dir, tension=1.0, terminal=True, width=0, pen_angle=0):
        i = len(self.stroke.nodes)
        n = Node(name or f"p{i}", np.array(pt(p), float),
                 None if dir is None else val(dir), val(tension), terminal, dw=val(width), dpa=val(pen_angle))
        self.stroke.nodes.append(n)
        return n

    # ``dir``: direction of travel in degrees (None = inferred)
    # ``tension``: >1 flatter / tighter corners, <1 rounder (both sides of the node)
    # ``width``: added to the pen width at this node (interpolated along segments)
    # ``pen_angle``: added to the pen's angle at this node (degrees, interpolated
    #   along segments): the nib turns as the stroke goes

    def start(self, p, name: str = "start", dir=None, terminal=True, tension=1.0, width=0, pen_angle=0):
        if self.stroke.nodes:
            raise DuctusError("start() called twice")
        self._node(p, name, dir, tension, terminal, width, pen_angle)
        return self

    def curve_to(self, p, name: str | None = None, dir=None, tension=1.0, terminal=True, width=0, pen_angle=0):
        self._need_start()
        self.stroke.kinds.append("curve")
        self._node(p, name, dir, tension, terminal, width, pen_angle)
        return self

    def line_to(self, p, name: str | None = None, dir=None, terminal=True, tension=1.0, width=0, pen_angle=0):
        self._need_start()
        self.stroke.kinds.append("line")
        self._node(p, name, dir, tension, terminal, width, pen_angle)
        return self

    def close(self, kind: str = "curve"):
        self._need_start()
        self.stroke.kinds.append(kind)
        self.stroke.closed = True
        return self

    def _need_start(self):
        if not self.stroke.nodes:
            raise DuctusError(f"path {self.stroke.name!r}: call start() first")


class ContourBuilder:
    """Escape hatch: a raw outline contour with named points."""

    def __init__(self, name: str):
        self.name = name
        self.contour = Contour()
        self.fullname = ""
        self.group = None       # contours sharing a group are oriented together (outer + counter)
        self.kind = "contour"
        self.negative = False   # True: subtracted from the glyph (apertures, cuts)

    def move(self, p, name=None):
        self.contour.add(pt(p), "on", name); return self

    def line(self, p, name=None):
        self.contour.add(pt(p), "on", name); return self

    def curve(self, c1, c2, p, name=None):
        self.contour.add(pt(c1), "off"); self.contour.add(pt(c2), "off")
        self.contour.add(pt(p), "on", name); return self

    def curve_close(self, c1, c2):
        """Final curve back to the first point."""
        self.contour.add(pt(c1), "off"); self.contour.add(pt(c2), "off")
        return self

    def close(self):
        """End the contour (it closes back to its first point). Points are
        never dropped here: structure must not depend on coordinates."""
        return self


class Builder:
    def __init__(self, scope: str, font: Font):
        self.scope = scope
        self.font = font
        self.calls: list[PartCall] = []
        self._strokes: list[Stroke] = []
        self._contours: list[ContourBuilder] = []

    @property
    def pen(self):
        return self.font.pen

    def __iadd__(self, other):
        if isinstance(other, PartCall):
            self.calls.append(other)
        elif isinstance(other, (list, tuple)):
            for o in other:
                self += o
        elif isinstance(other, (Stroke, PathBuilder, ContourBuilder)):
            pass  # already registered when created
        else:
            raise DuctusError(f"cannot add {other!r} to a glyph")
        return self

    def __getitem__(self, name: str) -> PartCall:
        for c in self.calls:
            if c.name == name:
                return c
        raise KeyError(f"{self.scope}: no part instance named {name!r} "
                       f"(have {[c.name for c in self.calls]})")

    def line(self, a, b, name: str = "line", names=("start", "end"), terminal=True, pen=None,
             profile=None, bias=None) -> Stroke:
        s = Stroke(name, [], ["line"], pen=pen, profile=profile, bias=bias)
        ta, tb = terminal if isinstance(terminal, tuple) else (terminal, terminal)
        s.nodes = [Node(names[0], np.array(pt(a), float), terminal=ta),
                   Node(names[1], np.array(pt(b), float), terminal=tb)]
        self._strokes.append(s)
        return s

    def path(self, name: str = "path", pen=None, inside=None, profile=None, bias=None,
             smooth=False) -> PathBuilder:
        """``inside`` (closed paths): share of the pen's contrast placed on
        the inner side (0.5 = symmetric; default: the pen's ``inside``).
        ``profile``: width along the stroke (:mod:`ductus.profiles`).
        ``bias`` (open paths): share of the modulation on the left of travel.
        ``smooth`` (open paths): nodes without a ``dir`` get the direction that
        makes the curvature continuous through them (Hobby / METAFONT),
        instead of the chord between their neighbours."""
        s = Stroke(name, [], [], pen=pen, inside=inside, profile=profile, bias=bias, smooth=bool(smooth))
        self._strokes.append(s)
        return PathBuilder(s)

    def contour(self, name: str = "contour") -> ContourBuilder:
        cb = ContourBuilder(name)
        self._contours.append(cb)
        return cb

    def cut(self, name: str = "cut") -> ContourBuilder:
        """A contour subtracted from the glyph after overlap removal
        (apertures, notches)."""
        cb = self.contour(name)
        cb.negative = True
        cb.kind = "cut"
        return cb

    def lobe(self, edge_x, side, far, bottom, top_join, bottom_join, thick, thin_bottom, **kw):
        """Bowl closed by a stem, meeting it at two joins; see :func:`ductus.contours.lobe`."""
        from .contours import lobe
        return lobe(self, edge_x, side, far, bottom, top_join, bottom_join, thick, thin_bottom, **kw)

    def bowl(self, left, right, bottom, top, thick, thin, **kw):
        """Contour-part bowl (outer + counter); see :mod:`ductus.contours`."""
        from .contours import bowl
        return bowl(self, left, right, bottom, top, thick, thin, **kw)


class GlyphBuilder(Builder):
    def __init__(self, gdef: GlyphDef, font: Font):
        super().__init__(gdef.name, font)
        self.gdef = gdef
        self.terminal_style = None
        self.inherited: GlyphDef | None = None

    def inherit(self, name: str):
        """Run another glyph's definition into this one. Names are re-scoped
        under this glyph (``h.arch.shoulder``); overrides are not inherited."""
        src = self.font.glyphs.get(name)
        if src is None:
            raise DuctusError(f"{self.scope}: cannot inherit unknown glyph {name!r}")
        self.inherited = src
        src.fn(self)
        return self

    def terminals(self, style):
        self.terminal_style = style
        return self


@dataclass
class Realized:
    strokes: list[Stroke] = field(default_factory=list)
    contours: list[ContourBuilder] = field(default_factory=list)
    parts: list[str] = field(default_factory=list)     # part functions used (for reports)


def realize(b: Builder) -> Realized:
    """Run all part calls (recursively) and assign full names."""
    out = Realized()
    for s in b._strokes:
        s.fullname = f"{b.scope}.{s.name}"
        out.strokes.append(s)
    for cb in b._contours:
        cb.fullname = f"{b.scope}.{cb.name}"
        out.contours.append(cb)
    for call in b.calls:
        child = Builder(f"{b.scope}.{call.name}", b.font)
        call._fn(child, **copy.copy(call._kwargs))
        sub = realize(child)
        out.parts.append(call._fn.__name__)
        out.parts += sub.parts
        # a part drawing exactly one item takes the part instance's name; cuts
        # (notches, apertures) are decorations and don't count
        drawn = child._strokes + [c for c in child._contours if not c.negative]
        if not child.calls and len(drawn) == 1:
            drawn[0].fullname = child.scope
        out.strokes += sub.strokes
        out.contours += sub.contours
    if isinstance(b, GlyphBuilder):
        _assign_point_names(b.scope, out)
    return out


def _assign_point_names(glyph: str, r: Realized):
    seen: dict[str, str] = {}

    def claim(name, what):
        if name in seen:
            raise DuctusError(f"duplicate name {name!r} ({what} and {seen[name]}); "
                              f"pass name=... to one of the parts")
        seen[name] = what

    for s in r.strokes:
        claim(s.fullname, "stroke")
        for n in s.nodes:
            n.fullname = f"{s.fullname}.{n.name}"
            claim(n.fullname, "point")
    for cb in r.contours:
        claim(cb.fullname, "contour")
        for i, nm in enumerate(cb.contour.names):
            if nm:
                cb.contour.names[i] = f"{cb.fullname}.{nm}"
                claim(cb.contour.names[i], "point")
