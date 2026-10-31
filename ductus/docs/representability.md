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

### All benchmark glyphs (iterations 04–05) — 0 overrides throughout

Combined overlay: `docs/iterations/05/_all.png`. Every glyph is built from
the serif kit (`ductus/kits/serif.py`) with parameters from `ductus fit`;
no hand overrides anywhere.

| glyph | IoU | what got it there | remaining gap → next primitive |
|---|---:|---|---|
| o | 0.984 | bowl with (h, v) fullness | — |
| H | 0.970 | joint n+H serif fit; **serif `slope`** (wedge-like taper to a sharp tip) | crossbar is a rectangle; reference bar is slightly thicker |
| n | 0.931 | arch with taper **profile**, **bias**, own pen; **wedge** head with cut | head wedge top edge: reference rises less steeply; arch crotch slightly lower |
| e | 0.922 | bowl − **aperture cut**; bar contour | terminal's upper edge: a cut can't make the reference's thin up-sweep → an **aperture with a curved terminal stroke** |
| v | 0.904 | monoline diagonals, **flat cut** at the vertex, **asymmetric sloped top serifs** (was 0.855 symmetric) | vertex: reference is a narrower notch below the baseline |
| s | 0.871 | spine stroke with **swell profile**, **beak** terminals | proportions: the fit trades the top arc against the spine — a two-stroke s (arc + spine) may represent it better |
| g | 0.848 | upper bowl, hairline link, **lower loop as a contour bowl** (was 0.733 as a pen stroke) | link bulges left; ear angle |
| a | 0.785 | **rotated, asymmetric bowl** (was 0.685 with an upright one) | bowl→stem junction and hood thickness: needs the SPEC's **`join()` primitive** (bowl meeting a stem with a hairline) |

Language gaps found this round, each closed by a primitive rather than
overrides: bowl rotation + asymmetric walls (a), sloped serif tops (n, H, v),
asymmetric per-side serifs on diagonals (v), contour loops (g), curved
aperture edges (e), terminals that cut, per-glyph sidebearings (v's negative
sidebearings). Bugs found by the benchmark: side detection counted serif
tips as the glyph's sides; overlays were drawn upside-down; a fit run while
the kit was being edited produced a bogus sidebearing (fits should run
against a pinned source).

**Coupling.** Shared parameters are fitted once and then shared: fitting `n`
moved `stem` 86 → 83, which cost `o` 0.03 IoU until `round_gain` (the
"rounds heavier than straights" optical rule) was refitted. The final pass
should be one joint fit of all shared params over all glyphs.

**Spacing.** Two shared sidebearings (straight/round) plus per-glyph
adjustments leave Δadv errors of −27…+89 (g is worst). Spacing-from-shape is
milestone M5; until then the benchmark scores shape and spacing separately.

### Proposed targets (after o and n, SPEC M2)

* **IoU ≥ 0.93 and Hausdorff ≤ 20 units, with ≤ 2 overrides per glyph.**
  `o` shows parameters alone reach ~0.98 on simple shapes; `n` (0.931,
  20.4 u) is the first glyph with junctions and terminals and sets the bar
  for complex ones. Met by o, H, n (borderline); not yet by e, v, s, g, a.

| iter | date | glyph | IoU | Hausdorff | mean | Δadv | overrides | Σ\|Δ\| | primitives | notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 01 | 2026-10-31 | `o` | 0.7658 | 17.0 | 8.4 | -54 | 0 | 0 | bowl | first guess: generic genre params, no fitting |
| 02 | 2026-10-31 | `o` | 0.9514 | 10.0 | 1.9 | -20 | 0 | 0 | bowl | params from ductus fit (826 evals); no overrides |
| 03 | 2026-10-31 | `o` | 0.9863 | 1.8 | 0.7 | +0 | 0 | 0 | bowl | bowl primitive with (h,v) fullness; global+local fitter, shape/spacing fitted separately; no overrides |
| 04 | 2026-10-31 | `o` | 0.9551 | 4.9 | 1.6 | -17 | 0 | 0 | bowl | serif kit + M2 primitives; params by ductus fit (per glyph, joint n+H serifs, joint spacing); 0 overrides |
| 04 | 2026-10-31 | `n` | 0.9311 | 20.4 | 2.5 | +34 | 0 | 0 | bracketed, wedge | serif kit + M2 primitives; params by ductus fit (per glyph, joint n+H serifs, joint spacing); 0 overrides |
| 04 | 2026-10-31 | `H` | 0.9702 | 5.0 | 1.1 | -6 | 0 | 0 | contour, bracketed | serif kit + M2 primitives; params by ductus fit (per glyph, joint n+H serifs, joint spacing); 0 overrides |
| 04 | 2026-10-31 | `e` | 0.9268 | 18.0 | 2.6 | -27 | 0 | 0 | bowl, contour, cut | serif kit + M2 primitives; params by ductus fit (per glyph, joint n+H serifs, joint spacing); 0 overrides |
| 04 | 2026-10-31 | `a` | 0.7850 | 40.1 | 6.7 | +3 | 0 | 0 | bowl, beak, wedge | serif kit + M2 primitives; params by ductus fit (per glyph, joint n+H serifs, joint spacing); 0 overrides |
| 04 | 2026-10-31 | `s` | 0.8707 | 13.3 | 3.9 | -4 | 0 | 0 | beak | serif kit + M2 primitives; params by ductus fit (per glyph, joint n+H serifs, joint spacing); 0 overrides |
| 04 | 2026-10-31 | `g` | 0.8478 | 36.7 | 4.9 | +89 | 0 | 0 | bowl, flat | serif kit + M2 primitives; params by ductus fit (per glyph, joint n+H serifs, joint spacing); 0 overrides |
| 04 | 2026-10-31 | `v` | 0.9039 | 27.8 | 2.9 | +0 | 0 | 0 | bracketed, flat | serif kit + M2 primitives; params by ductus fit (per glyph, joint n+H serifs, joint spacing); 0 overrides |
| 05 | 2026-10-31 | `o` | 0.9838 | 1.6 | 0.8 | -19 | 0 | 0 | bowl | 05: round_gain refit after the n fit moved the shared stem; 0 overrides |
| 05 | 2026-10-31 | `n` | 0.9311 | 20.4 | 2.5 | +34 | 0 | 0 | bracketed, wedge | 05: round_gain refit after the n fit moved the shared stem; 0 overrides |
| 05 | 2026-10-31 | `H` | 0.9702 | 5.0 | 1.1 | -6 | 0 | 0 | contour, bracketed | 05: round_gain refit after the n fit moved the shared stem; 0 overrides |
| 05 | 2026-10-31 | `e` | 0.9222 | 18.0 | 2.8 | -27 | 0 | 0 | bowl, contour, cut | 05: round_gain refit after the n fit moved the shared stem; 0 overrides |
| 05 | 2026-10-31 | `a` | 0.7850 | 40.1 | 6.7 | +3 | 0 | 0 | bowl, beak, wedge | 05: round_gain refit after the n fit moved the shared stem; 0 overrides |
| 05 | 2026-10-31 | `s` | 0.8707 | 13.3 | 3.9 | -4 | 0 | 0 | beak | 05: round_gain refit after the n fit moved the shared stem; 0 overrides |
| 05 | 2026-10-31 | `g` | 0.8478 | 36.7 | 4.9 | +89 | 0 | 0 | bowl, flat | 05: round_gain refit after the n fit moved the shared stem; 0 overrides |
| 05 | 2026-10-31 | `v` | 0.9039 | 27.8 | 2.9 | +0 | 0 | 0 | bracketed, flat | 05: round_gain refit after the n fit moved the shared stem; 0 overrides |
