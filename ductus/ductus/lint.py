"""Lints run by ``ductus test`` (spec section 7)."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .compile import CompiledFont, CompiledGlyph, compile_font, merge
from .geometry import flatten, point_in_polygon
from .overrides import Override
from .params import evaluating, val
from .parts import Font

STEM_TOLERANCE = 1.5  # font units (outlines are rounded to integers)


@dataclass
class Finding:
    check: str
    level: str        # "error" | "warning" | "info"
    message: str
    glyph: str | None = None

    def __str__(self):
        where = f"[{self.glyph}] " if self.glyph else ""
        return f"{self.level.upper():7} {self.check}: {where}{self.message}"


# -- per-glyph checks -------------------------------------------------------

def check_self_intersections(cg: CompiledGlyph) -> list[Finding]:
    """After overlap removal, merging again must be a no-op."""
    if not cg.contours:
        return []
    again = merge(cg.contours, round_to_grid=False)
    a0 = sum(c.area() for c in cg.contours)
    a1 = sum(c.area() for c in again)
    if len(again) != len(cg.contours) or abs(a0 - a1) > max(1.0, 1e-3 * abs(a0)):
        return [Finding("overlaps", "error",
                        f"outline still overlaps itself (contours {len(cg.contours)}->{len(again)}, "
                        f"area {a0:.0f}->{a1:.0f})", cg.name)]
    return []


def check_direction(cg: CompiledGlyph) -> list[Finding]:
    """PostScript/CFF convention: outer contours counter-clockwise, holes clockwise."""
    out = []
    polys = [flatten(c, 8) for c in cg.contours]
    for i, c in enumerate(cg.contours):
        depth = sum(point_in_polygon(c.pts[0], polys[j]) for j in range(len(polys)) if j != i)
        want_ccw = depth % 2 == 0
        if (c.area() > 0) != want_ccw:
            out.append(Finding("direction", "error",
                               f"contour {i} should be {'counter-' if want_ccw else ''}clockwise", cg.name))
    return out


def _span_at(contours, y: float, x: float) -> float | None:
    xs = []
    for c in contours:
        poly = flatten(c, 24)
        a, b = poly, np.roll(poly, -1, axis=0)
        cross = (a[:, 1] > y) != (b[:, 1] > y)
        if cross.any():
            t = (y - a[cross, 1]) / (b[cross, 1] - a[cross, 1])
            xs.extend(a[cross, 0] + t * (b[cross, 0] - a[cross, 0]))
    xs = sorted(xs)
    for l, r in zip(xs[0::2], xs[1::2]):
        if l - 0.5 <= x <= r + 0.5:
            return r - l
    return None


def check_stems(cg: CompiledGlyph, expected: float) -> list[Finding]:
    """Every vertical straight stroke must measure ``expected`` horizontally.
    Measured at several heights; the narrowest reading is used so that
    junctions (crossbars, arches) don't count."""
    out = []
    for s in cg.strokes:
        for P, kind in s.cubics():
            if kind != "line" or abs(P[3][0] - P[0][0]) > 0.01 or abs(P[3][1] - P[0][1]) < 50:
                continue
            x = P[0][0] + cg.shift
            spans = [_span_at(cg.contours, P[0][1] + f * (P[3][1] - P[0][1]), x)
                     for f in (0.15, 0.3, 0.7, 0.85)]
            spans = [w for w in spans if w is not None]
            if not spans:
                out.append(Finding("stems", "error", f"{s.fullname}: stem not found in outline", cg.name))
                continue
            w = min(spans)
            if abs(w - expected) > STEM_TOLERANCE:
                out.append(Finding("stems", "error",
                                   f"{s.fullname}: measured {w:.1f}, expected {expected:.1f}", cg.name))
    return out


# -- font-level checks -------------------------------------------------------

def check_overrides(cf: CompiledFont) -> list[Finding]:
    return [Finding("overrides", "error", w) for w in cf.warnings if w.startswith("override")]


def default_sweep(font: Font) -> list[dict[str, float]]:
    """Each ranged parameter at both ends of its range, plus all-min/all-max."""
    ranged = {n: p.range for n, p in font.params.items() if p.range}
    settings = [{n: r[i]} for n, r in ranged.items() for i in (0, 1)]
    settings += [{n: r[0] for n, r in ranged.items()}, {n: r[1] for n, r in ranged.items()}]
    return settings


def parse_sweep(specs: list[str]) -> list[dict[str, float]]:
    """``["stem=40,160", "contrast=0,0.9"]`` -> cartesian product."""
    axes = []
    for s in specs:
        name, _, values = s.partition("=")
        axes.append([(name.strip(), float(v)) for v in values.split(",") if v.strip()])
    return [dict(combo) for combo in itertools.product(*axes)]


def check_compatibility(font: Font, overrides, settings: list[dict[str, float]],
                        base: CompiledFont | None = None) -> list[Finding]:
    """Point compatibility across a sweep.

    * pre-merge outlines (one contour set per stroke/terminal) must match
      exactly -- these are what the variable font build will interpolate;
    * post-merge outlines are reported (info) when they differ, since
      overlap removal may legitimately change topology (spec section 6).
    """
    base = base or compile_font(font, overrides=overrides)
    out = []
    merged_diff: dict[str, list[str]] = {}
    for st in settings:
        cf = compile_font(font, st, overrides)
        label = ", ".join(f"{k}={v:g}" for k, v in st.items())
        for name, g in cf.glyphs.items():
            b = base.glyphs[name]
            if g.raw_signature != b.raw_signature:
                out.append(Finding("compatibility", "error",
                                   f"stroke outlines differ in structure at {label}", name))
            if g.signature != b.signature:
                merged_diff.setdefault(name, []).append(label)
    for name, labels in merged_diff.items():
        out.append(Finding("compatibility", "info",
                           f"merged outline topology changes at {len(labels)} setting(s) "
                           f"(e.g. {labels[0]}); variable builds keep strokes un-merged", name))
    return out


def run_lints(font: Font, overrides: dict[str, Override] | None = None,
              sweep: list[dict[str, float]] | None = None) -> tuple[list[Finding], CompiledFont | None]:
    findings: list[Finding] = []
    try:
        cf = compile_font(font, overrides=overrides)
    except Exception as e:
        return [Finding("compile", "error", f"{type(e).__name__}: {e}")], None
    with evaluating(cf.params):
        expected_stem = val(font.pen.width)
    for cg in cf.glyphs.values():
        findings += check_self_intersections(cg)
        findings += check_direction(cg)
        findings += check_stems(cg, expected_stem)
    findings += check_overrides(cf)
    for w in cf.warnings:
        if not w.startswith("override"):
            findings.append(Finding("compile", "warning", w))
    findings += check_compatibility(font, overrides, sweep if sweep is not None else default_sweep(font), cf)
    return findings, cf
