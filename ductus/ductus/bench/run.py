"""``ductus bench`` and ``ductus fit``."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..compile import CompiledFont, compile_font
from ..overrides import Override
from .metrics import GlyphScore, score_glyph
from .reference import Reference


def override_stats(glyph: str, overrides: dict[str, Override]) -> tuple[int, float]:
    mine = [o for n, o in overrides.items() if n.startswith(glyph + ".")]
    return len(mine), float(sum(abs(o.dx) + abs(o.dy) + abs(o.dw) for o in mine))


def score_font(cf: CompiledFont, ref: Reference, glyphs: list[str]) -> dict[str, GlyphScore]:
    out = {}
    for g in glyphs:
        cg = cf.glyphs[g]
        rname = ref.name_for(cg.unicode, g)
        if rname is None:
            raise KeyError(f"reference has no glyph for {g!r}")
        rc, radv = ref.glyph(rname)
        out[g] = score_glyph(g, cg.contours, rc, cg.advance, radv, cf.upm)
    return out


@dataclass
class FitResult:
    glyph: str
    params: dict[str, float]
    before: GlyphScore
    after: GlyphScore
    evaluations: int


def _shape_key(g):
    """Outline points relative to the bbox's left edge (spacing-invariant)."""
    x0 = g.bounds[0] if g.bounds else 0.0
    return [(round(p[0] - x0, 3), round(p[1], 3)) for c in g.contours for p in c.pts]


def glyph_params(font, glyph: str, overrides, exclude=()) -> tuple[list[str], list[str]]:
    """Ranged parameters that change ``glyph``, split by perturbation into
    (shape params, spacing-only params)."""
    base = compile_font(font, overrides=overrides, glyphs=[glyph]).glyphs[glyph]
    shape, spacing = [], []
    for name, p in font.params.items():
        if not p.range or name in exclude:
            continue
        lo, hi = p.range
        probe = lo + 0.73 * (hi - lo) if abs(p.default - (lo + 0.73 * (hi - lo))) > 1e-9 else lo
        g = compile_font(font, {name: probe}, overrides, glyphs=[glyph]).glyphs[glyph]
        if _shape_key(g) != _shape_key(base):
            shape.append(name)
        elif g.advance != base.advance or g.shift != base.shift:
            spacing.append(name)
    return shape, spacing


def fit_glyph(font, ref: Reference, glyph: str, overrides=None, params: list[str] | None = None,
              maxiter: int = 400, base_params: dict[str, float] | None = None,
              samples: int = 256, restarts: int = 3, exclude=()) -> FitResult:
    """Tune ranged parameters (not overrides) to minimise shape error."""
    from scipy.optimize import minimize
    overrides = overrides or {}
    base_params = dict(base_params or {})
    if params:
        names, spacing = list(params), []
    else:
        names, spacing = glyph_params(font, glyph, overrides, exclude)
    if not names:
        raise ValueError(f"no ranged parameters affect {glyph!r}")
    ranges = np.array([font.params[n].range for n in names], float)
    x0 = np.array([(base_params.get(n, font.params[n].default) - lo) / (hi - lo) for n, (lo, hi) in zip(names, ranges)])
    evals = [0]

    def run(x):
        x = np.clip(x, 0, 1)
        vals = {**base_params, **{n: lo + xi * (hi - lo) for n, xi, (lo, hi) in zip(names, x, ranges)}}
        cf = compile_font(font, vals, overrides, glyphs=[glyph])
        return vals, score_font(cf, ref, [glyph])[glyph]

    def objective(x):
        evals[0] += 1
        _, s = run(x)
        # mean outline distance is smooth in the parameters; raster IoU is
        # piecewise and stalls the simplex. Advance error weighs in too.
        return (s.mean_dist + 0.05 * s.hausdorff) / 10 + 10 * float(np.sum(np.clip(x - 1, 0, None) + np.clip(-x, 0, None)))

    _, before = run(x0)
    # global: quasi-random sample of the parameter box, then local Nelder-Mead
    # from the best few (a single local start gets stuck, e.g. trading
    # hairline thickness against counter shape)
    from scipy.stats import qmc
    d = len(names)
    starts = [x0] + list(qmc.Sobol(d, seed=1).random(samples))
    scored = sorted(((objective(x), i) for i, x in enumerate(starts)))
    best_x, best_f = None, float("inf")
    for _, i in scored[:restarts]:
        xs = starts[i]
        simplex = np.vstack([xs] + [np.clip(xs + 0.12 * np.eye(d)[k] * (1 if xs[k] < 0.5 else -1), 0, 1)
                                    for k in range(d)])
        res = minimize(objective, xs, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-6,
                                "initial_simplex": simplex, "adaptive": True})
        if res.fun < best_f:
            best_x, best_f = res.x, res.fun
    res = type("R", (), {"x": best_x})
    vals, after = run(res.x)
    # spacing: shape scores ignore position, so fit sidebearing params
    # afterwards, one at a time, against the advance difference
    from scipy.optimize import minimize_scalar
    for name in spacing:
        lo, hi = font.params[name].range

        def adv_err(v, name=name):
            evals[0] += 1
            cf = compile_font(font, {**vals, name: v}, overrides, glyphs=[glyph])
            return abs(score_font(cf, ref, [glyph])[glyph].advance_delta)

        vals[name] = float(minimize_scalar(adv_err, bounds=(lo, hi), method="bounded",
                                           options={"xatol": 0.25}).x)
    cf = compile_font(font, vals, overrides, glyphs=[glyph])
    after = score_font(cf, ref, [glyph])[glyph]
    return FitResult(glyph, {n: round(vals[n], 4) for n in names + spacing}, before, after, evals[0])
