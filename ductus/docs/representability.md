# Representability benchmark

How closely Ductus can represent a high-contrast display serif (reference:
Geist Serif, used for measurement only -- see SPEC section 8), and how much
hand work it took. One row per glyph per iteration; overlays live in
`docs/iterations/NN/`.

* **IoU** -- raster intersection-over-union at 400 px/em (1.0 = identical).
* **Hausdorff / mean** -- outline distance in font units (worst / average).
* **Δadv** -- advance width difference (spacing is scored separately from shape).
* **Overrides / Σ|Δ|** -- count and total magnitude of hand adjustments.

## Findings so far

**`o` (iterations 01–03): IoU 0.766 → 0.986, Hausdorff 17 → 1.8 units, zero overrides.**

* 01 — generic genre guess. Too narrow, counter too large.
* 02 — `ductus fit` (single-start Nelder–Mead on raster IoU) reached 0.951 but
  settled on a 20-unit hairline where the reference reads ~33: a local
  optimum, and IoU is piecewise-flat so the simplex stalls.
* Language gap found: one handle ratio per quadrant can't shape the counter —
  the reference's handles at the top extrema and at the side extrema have
  very different relative lengths. **Extended the bowl primitive** with
  `(h, v)` fullness (horizontal vs vertical handles, i.e. a superellipse with
  independent exponents). The fit uses it heavily (0.70 vs 0.42 outer,
  0.43 vs 0.82 counter).
* **Fitter fixed** rather than hand-tuning: Sobol sampling + Nelder–Mead from
  the best 3 starts, on the *smooth* mean outline distance instead of raster
  IoU, with shape fitted first and spacing-only params (detected by
  perturbation) fitted afterwards against the advance.
* 03 — residual error is ≤ 1.8 units everywhere (under a pixel at 400 px/em);
  the fitted walls are physically sensible (stem 100.5, hairline 33.5).

**Proposed targets:** to be fixed after `n` (SPEC asks for them after the
first two glyphs). Working assumption from `o`: IoU ≥ 0.97 and Hausdorff
≤ 8 units with ≤ 2 overrides per glyph — `o` shows params alone can reach
~0.986, so a lower bar would hide language gaps.

**Primitive pressure log** (recurring override kinds → candidate primitives):
none yet — no overrides were needed for `o`.

| iter | date | glyph | IoU | Hausdorff | mean | Δadv | overrides | Σ\|Δ\| | primitives | notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 01 | 2026-10-31 | `o` | 0.7658 | 17.0 | 8.4 | -54 | 0 | 0 | bowl | first guess: generic genre params, no fitting |
| 02 | 2026-10-31 | `o` | 0.9514 | 10.0 | 1.9 | -20 | 0 | 0 | bowl | params from ductus fit (826 evals); no overrides |
| 03 | 2026-10-31 | `o` | 0.9863 | 1.8 | 0.7 | +0 | 0 | 0 | bowl | bowl primitive with (h,v) fullness; global+local fitter, shape/spacing fitted separately; no overrides |
