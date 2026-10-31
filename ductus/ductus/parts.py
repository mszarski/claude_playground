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
        self.glyphs: dict[str, GlyphDef] = {}
        self.params: dict = {}          # filled in by the loader
        self.source_path = None
        _current_font = self

    def glyph(self, name: str, unicode: int | None = None, advance=None, sides=None):
        def deco(fn):
            u = unicode if unicode is not None else _unicode_for(name)
            self.glyphs[name] = GlyphDef(name, fn, u, advance, sides)
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

    def _node(self, p, name, dir, tension=1.0, terminal=True):
        i = len(self.stroke.nodes)
        n = Node(name or f"p{i}", np.array(pt(p), float),
                 None if dir is None else val(dir), val(tension), terminal)
        self.stroke.nodes.append(n)
        return n

    def start(self, p, name: str = "start", dir=None, terminal=True):
        if self.stroke.nodes:
            raise DuctusError("start() called twice")
        self._node(p, name, dir, terminal=terminal)
        return self

    def curve_to(self, p, name: str | None = None, dir=None, tension=1.0, terminal=True):
        self._need_start()
        self.stroke.kinds.append("curve")
        self._node(p, name, dir, tension, terminal)
        return self

    def line_to(self, p, name: str | None = None, dir=None, terminal=True):
        self._need_start()
        self.stroke.kinds.append("line")
        self._node(p, name, dir, terminal=terminal)
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

    def move(self, p, name=None):
        self.contour.add(pt(p), "on", name); return self

    def line(self, p, name=None):
        self.contour.add(pt(p), "on", name); return self

    def curve(self, c1, c2, p, name=None):
        self.contour.add(pt(c1), "off"); self.contour.add(pt(c2), "off")
        self.contour.add(pt(p), "on", name); return self

    def close(self):
        c = self.contour
        if len(c.pts) > 1 and c.types[-1] == "on" and c.pts[-1] == c.pts[0]:
            c.pts.pop(); c.types.pop(); c.names.pop()
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

    def line(self, a, b, name: str = "line", names=("start", "end"), terminal=True, pen=None) -> Stroke:
        s = Stroke(name, [], ["line"], pen=pen)
        ta, tb = terminal if isinstance(terminal, tuple) else (terminal, terminal)
        s.nodes = [Node(names[0], np.array(pt(a), float), terminal=ta),
                   Node(names[1], np.array(pt(b), float), terminal=tb)]
        self._strokes.append(s)
        return s

    def path(self, name: str = "path", pen=None) -> PathBuilder:
        s = Stroke(name, [], [], pen=pen)
        self._strokes.append(s)
        return PathBuilder(s)

    def contour(self, name: str = "contour") -> ContourBuilder:
        cb = ContourBuilder(name)
        self._contours.append(cb)
        return cb


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
        if not child.calls and len(child._strokes) + len(child._contours) == 1:
            for item in child._strokes + child._contours:
                item.fullname = child.scope
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
