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

### Iteration 07 — joint fit of the shared parameters

Shared parameters were found by perturbation: `stem`, `hairline`,
`overshoot`, `round_gain` and the serif curve (`serif_ca`, `serif_cb`,
`serif_cup`) shape 2–6 benchmark glyphs each; `sb_straight` / `sb_round`
space 5–6. They were fitted **jointly over all eight glyphs** with a
`target` objective (IoU + distance, plus a hinge penalty for any glyph
falling below IoU 0.93 or above 20 u Hausdorff — the mean may not be bought
with one glyph), then spacing jointly, then each glyph's own parameters
re-polished against the new shared values.

| glyph | 06 | 07 |
|---|---:|---:|
| o | 0.983 | 0.985 |
| n | 0.941 | **0.950** |
| H | 0.970 | **0.989** |
| e | 0.937 | 0.935 |
| a | 0.938 | 0.937 |
| s | 0.936 | **0.940** |
| g | 0.931 | 0.931 |
| v | 0.958 | 0.960 |

Mean IoU 0.951 → 0.953, min 0.931. Lesson: the per-glyph re-polish made 4
of 8 glyphs *worse* (its objective trades a little IoU for distance), and a
blind merge would have pushed g below target. Polish results are now only
accepted if they improve the glyph and keep it on target
(`ductus fit --only-improve`).

### Iteration 06 — every benchmark glyph above target, 0 overrides

Combined overlay: `docs/iterations/06/_all.png`.

| glyph | IoU 05 → 06 | Hausdorff 06 | what closed the gap |
|---|---:|---:|---|
| o | 0.984 → 0.983 | 1.6 | — (bowl, (h, v) fullness) |
| H | 0.970 → 0.970 | 5.0 | — (sloped bracketed serifs) |
| v | 0.904 → **0.958** | 15.2 | measuring showed both arms are constant width and the vertex is the thick arm **clipped by the thin arm's outer edge** → a vertex-clip cut (tapers were the wrong idea) |
| n | 0.931 → **0.941** | 18.4 | arch rebuilt on the **`join` primitive** (crotch height, hairline thickness, angle, notch); head refit fixed a stem top 20 u too high |
| a | 0.785 → **0.938** | 7.8 | bowl rebuilt as a **lobe** hanging on the stem at two **joins**; the a's own hood pen angle; a bug fix (terminal cuts were 3 widths wide and sliced the bowl) |
| e | 0.922 → **0.937** | 6.6 | aperture edge leaves the terminal tip at its own angle and depth (was a straight line + sag) |
| s | 0.871 → **0.936** | 5.9 | **per-node width parameters** (pen × profile can't weight each curve of the s independently); bottom beak points *up* (wrong guess before); s gets its own beak params (it was sharing a's — fitting a silently changed s) |
| g | 0.848 → **0.931** | 18.7 | lower loop's (h, v) fullness; a **flat bar** across the loop top with a trim cut — the reference loop is tilted *and* flat-topped, which one contour bowl can't do |

**Primitives added this round:** `join` (junctions.py) used by arches, the
u cup, r's ear and the new **`lobe`** contour part; per-node width params;
vertex clip; flat bar + trim. **Fitter:** Powell polishing with an IoU-blended
objective (`--method powell --objective iou`), staged fits (sub-part params
first), and measured starting points for the benchmark when the search
stalled in a wrong basin (v, g — measurement only, it's the benchmark).

**Bugs found by the benchmark:** oversized terminal cut boxes clipping
neighbouring parts; the lobe counter crossing its outer contour with a thin
bottom wall; naming collapse broken by cut contours (`n.arch.arch.*`);
shared beak params coupling a and s.

**Still open:** spacing — two shared sidebearings leave Δadv up to +67 (g)
and +34 (n); that is M5 (spacing from shape). A final *joint* fit of all
shared params over all glyphs is still to do.

### Proposed targets (after o and n, SPEC M2)

* **IoU ≥ 0.93 and Hausdorff ≤ 20 units, with ≤ 2 overrides per glyph.**
  `o` shows parameters alone reach ~0.98 on simple shapes; `n` (0.931,
  20.4 u) is the first glyph with junctions and terminals and sets the bar
  for complex ones. **Met by all eight benchmark glyphs as of iteration 06, with 0 overrides.**

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
| 06 | 2026-11-01 | `o` | 0.9834 | 1.6 | 0.8 | -19 | 0 | 0 | bowl | 06: join primitive (arches + lobe), bounded terminal cuts, v vertex clip, s node widths, g flat bar; Powell/IoU polish; 0 overrides |
| 06 | 2026-11-01 | `n` | 0.9407 | 18.4 | 2.2 | +34 | 0 | 0 | cut, bracketed, wedge | 06: join primitive (arches + lobe), bounded terminal cuts, v vertex clip, s node widths, g flat bar; Powell/IoU polish; 0 overrides |
| 06 | 2026-11-01 | `H` | 0.9702 | 5.0 | 1.1 | -6 | 0 | 0 | contour, bracketed | 06: join primitive (arches + lobe), bounded terminal cuts, v vertex clip, s node widths, g flat bar; Powell/IoU polish; 0 overrides |
| 06 | 2026-11-01 | `e` | 0.9365 | 6.6 | 2.2 | -27 | 0 | 0 | bowl, contour, cut | 06: join primitive (arches + lobe), bounded terminal cuts, v vertex clip, s node widths, g flat bar; Powell/IoU polish; 0 overrides |
| 06 | 2026-11-01 | `a` | 0.9377 | 7.8 | 2.1 | +2 | 0 | 0 | lobe, beak, wedge | 06: join primitive (arches + lobe), bounded terminal cuts, v vertex clip, s node widths, g flat bar; Powell/IoU polish; 0 overrides |
| 06 | 2026-11-01 | `s` | 0.9363 | 5.9 | 2.0 | -5 | 0 | 0 | beak | 06: join primitive (arches + lobe), bounded terminal cuts, v vertex clip, s node widths, g flat bar; Powell/IoU polish; 0 overrides |
| 06 | 2026-11-01 | `g` | 0.9306 | 18.7 | 2.2 | +67 | 0 | 0 | bowl, contour, cut | 06: join primitive (arches + lobe), bounded terminal cuts, v vertex clip, s node widths, g flat bar; Powell/IoU polish; 0 overrides |
| 06 | 2026-11-01 | `v` | 0.9576 | 15.2 | 1.4 | -2 | 0 | 0 | cut, bracketed, flat | 06: join primitive (arches + lobe), bounded terminal cuts, v vertex clip, s node widths, g flat bar; Powell/IoU polish; 0 overrides |
| 07 | 2026-11-01 | `o` | 0.9849 | 1.6 | 0.8 | -14 | 0 | 0 | bowl | 07: joint fit of shared params (stem, hairline, overshoot, round_gain, serif curves) over all 8 glyphs with a no-glyph-below-target objective; per-glyph re-polish accepted only where it improved; 0 overrides |
| 07 | 2026-11-01 | `n` | 0.9501 | 17.5 | 1.9 | +29 | 0 | 0 | cut, bracketed, wedge | 07: joint fit of shared params (stem, hairline, overshoot, round_gain, serif curves) over all 8 glyphs with a no-glyph-below-target objective; per-glyph re-polish accepted only where it improved; 0 overrides |
| 07 | 2026-11-01 | `H` | 0.9893 | 2.8 | 0.7 | -13 | 0 | 0 | contour, bracketed | 07: joint fit of shared params (stem, hairline, overshoot, round_gain, serif curves) over all 8 glyphs with a no-glyph-below-target objective; per-glyph re-polish accepted only where it improved; 0 overrides |
| 07 | 2026-11-01 | `e` | 0.9354 | 6.7 | 2.4 | -25 | 0 | 0 | bowl, contour, cut | 07: joint fit of shared params (stem, hairline, overshoot, round_gain, serif curves) over all 8 glyphs with a no-glyph-below-target objective; per-glyph re-polish accepted only where it improved; 0 overrides |
| 07 | 2026-11-01 | `a` | 0.9370 | 7.8 | 2.1 | +2 | 0 | 0 | lobe, beak, wedge | 07: joint fit of shared params (stem, hairline, overshoot, round_gain, serif curves) over all 8 glyphs with a no-glyph-below-target objective; per-glyph re-polish accepted only where it improved; 0 overrides |
| 07 | 2026-11-01 | `s` | 0.9395 | 6.8 | 1.8 | +2 | 0 | 0 | beak | 07: joint fit of shared params (stem, hairline, overshoot, round_gain, serif curves) over all 8 glyphs with a no-glyph-below-target objective; per-glyph re-polish accepted only where it improved; 0 overrides |
| 07 | 2026-11-01 | `g` | 0.9312 | 18.7 | 2.2 | +67 | 0 | 0 | bowl, contour, cut | 07: joint fit of shared params (stem, hairline, overshoot, round_gain, serif curves) over all 8 glyphs with a no-glyph-below-target objective; per-glyph re-polish accepted only where it improved; 0 overrides |
| 07 | 2026-11-01 | `v` | 0.9597 | 15.0 | 1.3 | +0 | 0 | 0 | cut, bracketed, flat | 07: joint fit of shared params (stem, hairline, overshoot, round_gain, serif curves) over all 8 glyphs with a no-glyph-below-target objective; per-glyph re-polish accepted only where it improved; 0 overrides |
