"""``ductus bench`` and ``ductus fit``."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..compile import CompiledFont, compile_font
from ..overrides import Override
from ..compile import translated_contours
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
        terms = translated_contours(cg.stages.get("terminals", []), cg.shift) if cg.stages else []
        out[g] = score_glyph(g, cg.contours, rc, cg.advance, radv, cf.upm, terminals=terms)
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


def expand_params(font, patterns) -> list[str]:
    import fnmatch
    out = []
    for pat in patterns:
        hits = [n for n, p in font.params.items() if p.range and fnmatch.fnmatchcase(n, pat)]
        if not hits:
            raise ValueError(f"no ranged parameter matches {pat!r}")
        out += [h for h in hits if h not in out]
    return out


def fit_glyph(font, ref: Reference, glyph, overrides=None, params: list[str] | None = None,
              maxiter: int = 400, base_params: dict[str, float] | None = None,
              samples: int = 256, restarts: int = 3, exclude=(), method: str = "nelder-mead",
              objective_kind: str = "distance") -> FitResult:
    """Tune ranged parameters (not overrides) to minimise shape error."""
    from scipy.optimize import minimize
    overrides = overrides or {}
    base_params = dict(base_params or {})
    glyphs = [glyph] if isinstance(glyph, str) else list(glyph)
    glyph = glyphs[0]
    if params:
        names = expand_params(font, params)
        names = [n for n in names if n not in exclude]
        # split the requested params into shape vs spacing-only (by perturbation)
        shape_all, spacing_all = set(), set()
        for gname in glyphs:
            sh, sp = glyph_params(font, gname, overrides, exclude)
            shape_all |= set(sh); spacing_all |= set(sp)
        spacing = [n for n in names if n in spacing_all and n not in shape_all]
        names = [n for n in names if n not in spacing]
    else:
        names, spacing = glyph_params(font, glyph, overrides, exclude)
    if not names and not spacing:
        raise ValueError(f"no ranged parameters affect {glyph!r}")
    ranges = np.array([font.params[n].range for n in names], float)
    x0 = np.array([(base_params.get(n, font.params[n].default) - lo) / (hi - lo) for n, (lo, hi) in zip(names, ranges)])
    evals = [0]

    def run_all(x):
        x = np.clip(x, 0, 1)
        vals = {**base_params, **{n: lo + xi * (hi - lo) for n, xi, (lo, hi) in zip(names, x, ranges)}}
        try:
            cf = compile_font(font, vals, overrides, glyphs=glyphs)
            return vals, score_font(cf, ref, glyphs)
        except Exception:
            return vals, None

    def run(x):
        vals, sc = run_all(x)
        return vals, (sc[glyph] if sc else None)

    def objective(x):
        evals[0] += 1
        _, sc = run_all(x)
        if sc is None:
            return 1e3
        # mean outline distance is smooth in the parameters; raster IoU is
        # piecewise and stalls the simplex.
        pen = 10 * float(np.sum(np.clip(x - 1, 0, None) + np.clip(-x, 0, None)))
        if objective_kind == "target":
            # joint fits: IoU blend, plus a hinge penalty for any glyph that
            # falls below the benchmark target -- the average may not buy a
            # better mean by sacrificing one glyph
            base = sum(5 * (1 - s.iou) + (s.mean_dist + 0.15 * s.hausdorff) / 10 for s in sc.values()) / len(sc)
            hinge = sum(50 * max(0.0, 0.93 - s.iou) + 0.5 * max(0.0, s.hausdorff - 20) for s in sc.values())
            return base + hinge + pen
        if objective_kind == "ends":
            # stroke-end detail: outline distance near the terminals, with
            # IoU and overall distance as ballast and the target hinge
            def e(s):
                return s.ends_mean if s.ends_mean == s.ends_mean else 0.0
            base = sum(e(s) / 4 + 5 * (1 - s.iou) + s.mean_dist / 10 for s in sc.values()) / len(sc)
            hinge = sum(50 * max(0.0, 0.93 - s.iou) + 0.5 * max(0.0, s.hausdorff - 20) for s in sc.values())
            return base + hinge + pen
        if objective_kind == "iou":
            # polishing towards the benchmark target: IoU plus a distance term
            return sum(5 * (1 - s.iou) + (s.mean_dist + 0.15 * s.hausdorff) / 10
                       for s in sc.values()) / len(sc) + pen
        return sum((s.mean_dist + 0.15 * s.hausdorff) / 10 for s in sc.values()) / len(sc) + pen * float(np.sum(np.clip(x - 1, 0, None) + np.clip(-x, 0, None)))

    _, before = run(x0)
    if not names:                                   # spacing-only fit: skip the shape search
        best_x = x0
        res = type("R", (), {"x": x0})
        vals, after = run(x0)
        return _fit_spacing(font, ref, glyphs, glyph, overrides, vals, names, spacing, before, evals)
    # global: quasi-random sample of the parameter box, then local Nelder-Mead
    # from the best few (a single local start gets stuck, e.g. trading
    # hairline thickness against counter shape)
    from scipy.stats import qmc
    d = len(names)
    starts = [x0] + list(qmc.Sobol(d, seed=1).random_base2(max(1, int(np.ceil(np.log2(max(samples, 2)))))))
    scored = sorted(((objective(x), i) for i, x in enumerate(starts)))
    best_x, best_f = None, float("inf")
    for _, i in scored[:restarts]:
        xs = starts[i]
        simplex = np.vstack([xs] + [np.clip(xs + 0.12 * np.eye(d)[k] * (1 if xs[k] < 0.5 else -1), 0, 1)
                                    for k in range(d)])
        if method == "powell":
            # coordinate-wise line searches: better than a simplex at polishing
            # many weakly coupled parameters from a good start
            res = minimize(objective, xs, method="Powell", bounds=[(0, 1)] * d,
                           options={"maxiter": max(2, maxiter // (10 * d)), "xtol": 1e-4, "ftol": 1e-6})
        else:
            res = minimize(objective, xs, method="Nelder-Mead",
                           options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-6,
                                    "initial_simplex": simplex, "adaptive": True})
        if res.fun < best_f:
            best_x, best_f = res.x, res.fun
    res = type("R", (), {"x": best_x})
    vals, after = run(res.x)
    return _fit_spacing(font, ref, glyphs, glyph, overrides, vals, names, spacing, before, evals)

def _fit_spacing(font, ref, glyphs, glyph, overrides, vals, names, spacing, before, evals):
    """Shape scores ignore position, so sidebearing params are fitted
    afterwards, one at a time, against the advance difference."""
    from scipy.optimize import minimize_scalar
    for name in spacing:
        lo, hi = font.params[name].range

        def adv_err(v, name=name):
            evals[0] += 1
            cf = compile_font(font, {**vals, name: v}, overrides, glyphs=glyphs)
            return sum(abs(s.advance_delta) for s in score_font(cf, ref, glyphs).values())

        vals[name] = float(minimize_scalar(adv_err, bounds=(lo, hi), method="bounded",
                                           options={"xatol": 0.25}).x)
    cf = compile_font(font, vals, overrides, glyphs=glyphs)
    scores = score_font(cf, ref, glyphs)
    r = FitResult(glyph, {n: round(vals[n], 4) for n in names + spacing}, before, scores[glyph], evals[0])
    r.scores = scores
    return r
