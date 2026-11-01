# Representability benchmark

How closely Ductus can represent a high-contrast display serif (reference:
Geist Serif, used for measurement only -- see SPEC section 8), and how much
hand work it took. One row per glyph per iteration; overlays live in
`docs/iterations/NN/`.

* **IoU** -- intersection-over-union of the outlines (1.0 = identical); exact and aligned at the best offset from iteration 09, a 400 px/em raster with bbox-centred alignment before.
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

### Iteration 14 — the pen angle can turn along a stroke

New in the language: a skeleton node can turn the nib —
`curve_to(..., pen_angle=12)`, or a `dp` override on any node (the editor
shows each node's pen angle; keys `9` / `0` turn it by 1°, ⇧ 5°). The turn is
interpolated along the adjacent segments like a node's width delta, and the
direction model normalises per sample, so a vertical stroke stays exactly
`width` whatever the nib does. The nib is held within ±80° (the
normalisation divides by cos of the angle; at the all-max sweep corner the
t's pen plus a 35° turn passed 90° and the width blew up). Zero turn is
bit-identical to before: no existing glyph changed.

Geist kit: per-node turns on the s (`s_t1_pa` … `s_t2_pa`), the arch
(`n_join_pa`, `n_shoulder_pa`, `n_leg_pa`), the a's hood (`a_term_pa`,
`a_top_pa`) and the t's tail (`t_bottom_pa`, `t_end_pa`), all 0 by default.

Result: **s 0.955 → 0.960**, a 0.961 → 0.962; the arch and the t's tail
did not use it (the fits left those turns at 0). The fitted turns on the s
are small (≤ 3.3°): Geist's weight distribution is mostly not a turning
broad nib — its width profile and per-node widths already carry most of
it. Mean IoU 0.976 → 0.976 (0.9761 → 0.9764); 0 overrides.

### Iteration 13 — the rest: mean IoU 0.970 → 0.976, every glyph ≥ 0.955

| glyph | 12 | 13 | | glyph | 12 | 13 |
|---|---:|---:|---|---|---:|---:|
| o | 0.984 | 0.984 | | h | 0.966 | **0.974** |
| n | 0.973 | **0.982** | | m | 0.962 | **0.969** |
| H | 0.995 | 0.995 | | b | 0.959 | **0.975** |
| e | 0.951 | **0.969** | | u | 0.977 | 0.981 |
| a | 0.959 | 0.961 | | r | 0.976 | 0.980 |
| s | 0.954 | 0.955 | | f | 0.973 | 0.978 |
| g | 0.962 | 0.966 | | t | 0.959 | **0.979** |
| v | 0.973 | 0.976 | | i | 0.992 | 0.992 |

0 overrides; every Hausdorff ≤ 16.4 u. What it took:

* **Pinned ranges.** `t_bar_r`, `a_bx_in` and `g_link_tp` sat exactly on
  their range bounds: a fitted value stretches its range to include
  itself but never past, so no fit could move them further (the t's bar
  was 8 u short and could not grow). Widened.
* **e** shared its wall thickness with the o: it now has its own per side
  (`e_thick_dl`, `e_thick_dr`; the left wall was too thick its whole
  height), and the bar stops short of the curved outer edge (`e_bar_rin`).
* **Coupled moves.** The b's bowl wall and its reach only help together
  (+9 u each: 0.959 → 0.966; either alone is worse), and so on across most
  glyphs. The greedy search now also tries random 2–3-parameter moves, and
  a multi-glyph version fitted the arch shared by n, h and m (mean 0.967
  → 0.975), after which r, f, u and t were re-polished on their own params.
* **A metric guard.** The search found an o with "IoU 1.007": skia can
  return a wrong intersection *consistently* (both the intersection and
  difference ops), which the earlier cross-check could not see. The
  intersection may not exceed either area and must agree with a coarse
  raster to 1%; otherwise the 0.5 u raster is used. Current scores were
  unaffected; the merge also drops contours under 4 u² (a 1 u sliver at
  the refitted v's vertex).
* Tried and not used: the a's own stem weight and bottom (`a_stem_w`,
  `a_stem_bot`); tilted top/bottom tangents for the s (`s_top_dir`,
  `s_bot_dir`: horizontal is optimal). The s (0.955) and a (0.961) look
  close to what this model can do; the s's remaining error is a weight
  distribution along the stroke that a single pen angle can't follow.

### Iteration 12 — g and s: g 0.946 → 0.962, s 0.947 → 0.954

Both were the lowest glyphs after iteration 11. Mean IoU 0.968 → 0.970,
min 0.946 → 0.951 (e); 0 overrides. A difference map (Geist-only ink vs
ours-only ink) was what found both problems; the outline overlays alone
had been misread.

* **g: a structural notch.** In Geist the link does not run into the lower
  loop on the left: a wedge-shaped notch opens to the left between the
  link's bottom and the loop's thin top-left arm, closing ~7 u wide under
  the crossbar; the two meet only through the crossbar. Ours filled the
  wedge solid. No fit could find it — a global fit of all 30-odd link,
  loop and bar params gained 0.0002, and a lower, thinner loop top or more
  rotation (the loop is tilted, not rotated) made things far worse — so
  the notch is now modelled directly: a wedge cut with its tip at
  (`g_notch_x`, `g_notch_y`), `g_notch_len` long, edges `g_notch_up` /
  `g_notch_dn` degrees off the horizontal (length 0 = no notch, the old
  g). With it in place a constrained greedy search over all g params went
  0.948 → 0.962.
* **s: weight distribution along the stroke.** The error was spread
  thinly; the s needed more freedom, not a fix: its left/right extremes can
  lean (`s_left_dir`, `s_right_dir`; Geist's right extreme leans ~2°),
  every node has its own tension (`s_left_t`, `s_mid_t`, `s_right_t`,
  `s_bot_dt`), and each end its own weight (`s_t1_dw`, `s_t2_dw`: the
  bottom end's lower edge sat 10 u high, and the cut's fill poked out past
  it as a notch). All default to the old shape, which the display serif's
  S shares.

Lowest now: e 0.951, a 0.959, t 0.959, b 0.959.

### Iteration 11 — stroke ends: outline distance near the ends 1.48 → 1.07 u

The question was how to get the finer detail at the ends of strokes. About
a quarter of all error was within 30 u of a terminal — 89% of the v's, 71%
of the i's, 66% of the r's — and IoU barely rewards it: a 4 u tip or a
small fillet is almost no area. Now:

* **Measured.** `ductus bench` reports, per glyph, the mean outline
  distance at outline points within 30 u of a terminal shape (`ends`), and
  a per-style summary line. Over the 13 glyphs with terminals:
  **1.48 → 1.07 u** (−28%); bracketed 0.98 → 0.72, wedge 1.73 → 1.09,
  cuts 1.37 → 0.93; beaks no longer occur (below).
* **Fitted on.** `ductus fit --objective ends`: that distance, with IoU and
  the whole-outline mean as ballast and the target hinge. Used for a joint
  fit of every terminal parameter over all 16 glyphs, then of the head
  wedge over the 7 headed glyphs.
* **Primitive fixes found on the way:**
  * *Brackets on diagonals.* A bracketed serif dropped its bracket
    vertically from the stroke's corner; on a slanted stroke that left a
    notch under the serif (v's top serifs; also the display serif's v and
    A). The bracket now starts on the stroke's own edge. v 0.966 → 0.973.
  * *Head inset at joins.* Geist's stem is set in on the right between an
    arch or ear join and the head (~4 u on n and m, ~8 u on r; none on i,
    u, or the h's long ascender): `head_inset`, `head_inset_y`,
    `r_head_dinset`. r 0.957 → 0.976, n 0.968 → 0.973.
  * *t tail* is cut at an angle, not vertically: `t_tail_cut`
    (Hausdorff 16.5 → 9.7).
  * *Spurs* get the head wedge's fine-shape controls (`spur_tip`, `_flat`,
    `_bracket`, `_corner`, `_ca`, `_cb`, `_sweep`).
  * *No beaks where Geist has none.* The s's beaks had been fitted down to
    nothing, leaving a 2 u flap outside the stroke; the a's hood beak (107 u
    deep) stood in for the hood's curved outer edge and left a notch where
    that edge curved away. Both are plain cuts in the Geist kit now (the
    display serif keeps its beaks/balls). The a gives up 0.0008 IoU for
    having no artifact. (Running the beak's back edge along the stroke
    direction instead was tried and is wrong for curved strokes: the a's
    Hausdorff went to 42.)
  * Rounded corners on cuts were *not* added: Geist's cut corners on r, f
    and t are sharp.
* Robustness: when skia can't produce a self-consistent intersection
  even after nudging, the IoU falls back to a 0.5 u raster for that one
  evaluation instead of failing the fit.

Mean IoU 0.9645 → 0.9682 (min 0.946, g), mean Hausdorff 10.0 → 9.9 u
(max 16.3, v). Still 0 overrides.

### Iteration 10 — g and t off the distance limit

Both sat at Hausdorff 20 u in iteration 09; now t **0.958 / 16.5 u**
(from 0.946 / 19.9) and g **0.946 / 15.4 u** (from 0.939 / 20.0). Mean IoU
0.963 → 0.964; no other glyph changed. Still 0 overrides.

* **t** — Geist's t head is a flat top whose left flank is one concave curve
  down into the crossbar's tip; ours rose in a straight slope to a point
  17 u too tall. The wedge's sloped edge can now sag into a curve
  (`sweep`, a cubic that is straight at 0; clamped to 40% of the rise so
  it can't fold past the tip), and the t takes the existing `flat`:
  `t_wedge_flat`, `t_wedge_sweep`. Where the new flank meets the tapered
  bar almost tangentially, the boolean merge left an 8 u needle: merge now
  drops on-curve points where the outline doubles back (> 170°) next to a
  segment under 12 u.
* **g** — two spots were at the limit, and fixing one exposed the other.
  (1) The notch between the link and the lower loop: Geist's loop rises to
  only ~19 u under the link, ours ~30 u, filling the notch; a thinner loop
  top (`g_loop_top_t` −4) took Hausdorff to 17.8 at an IoU cost, which a
  greedy search that also rewards lower Hausdorff recovered. (2) The upper
  bowl's shoulder under the ear sits lower in Geist: new per-quadrant
  fullness for the bowl's top-right (`g_tr_dh`, `g_tr_dv`), plus
  `g_full_dv`, `g_in_full_dv`, `g_rot`. Also made fittable but not
  ultimately used: the link's root (`g_link_root_dy/dir`) and the ear's
  root (`g_ear_root_dx`); `g_loop_rot`'s range was widened (the fit sat on
  its old 15° bound).

### Iteration 09 — exact scoring, head and foot shape, joint fit: mean IoU 0.949 → 0.963, 0 overrides

| glyph | 08 | 09 | | glyph | 08 | 09 |
|---|---:|---:|---|---|---:|---:|
| o | 0.985 | 0.984 | | h | 0.934 | **0.966** |
| n | 0.950 | **0.968** | | m | 0.933 | **0.959** |
| H | 0.989 | 0.995 | | b | 0.943 | **0.959** |
| e | 0.935 | 0.951 | | u | 0.956 | 0.972 |
| a | 0.937 | **0.956** | | r | 0.935 | **0.959** |
| s | 0.940 | 0.941 | | f | 0.953 | 0.965 |
| g | 0.931 | 0.939 | | t | 0.943 | 0.946 |
| v | 0.960 | 0.966 | | i | 0.956 | **0.989** |

Mean IoU 0.949 → 0.963 (min 0.931 → 0.939), mean Hausdorff 12.7 → 10.5 u,
overrides 3 → **0**. Overlays: `docs/iterations/09/_all.png`.

What it took, in order:

1. **Exact scoring.** IoU is now the area of the outlines' intersection
   over their union (skia-pathops), not a 400 px/em raster (2.5 u per
   pixel); the reference is shifted to the offset that maximises it,
   instead of matching bbox centres. Outlines are on the integer grid, so
   the bbox centre moved in half-unit steps, worth ~0.005 IoU on a glyph
   with tall stems: the score was step-shaped, the fitters stalled on the
   steps, and r's iteration-08 override had only been exploiting one (r
   scores the same without it). The two scorers agree to ±0.0005 where the
   old one was stable, so earlier rows stay comparable — except `e`, whose
   bbox-centred alignment was 2.5 u off (0.935 → 0.950 from alignment
   alone).
   *Caveat found on the way:* skia's boolean ops can silently return a wrong
   outline when edges nearly coincide (one inflated m by 0.015). Every
   intersection area is checked against area(A) − area(A − B) and
   recomputed with a 0.001 u nudge if they disagree.
2. **Feet**: Geist's bracket is a straight diagonal (~31°) with a small
   fillet at the stem. The existing `serif_slope` already expressed that;
   the fitter had never reached it from a curved start. Set by probe,
   before the head fit: n 0.950 → 0.961, m 0.932 → 0.943, i 0.963 → 0.973.
3. **Head wedge** (n h m u r i b): Geist's head has a short flat top, a
   ~4 u vertical tip and an underside that runs straight from the tip and
   then fillets into the stem; ours was a sharp-cornered wedge with one deep
   curve, 13 u too tall. New wedge controls `tip`, `flat`, `bracket`
   (blend from the hollow curve to a straight-plus-fillet underside),
   `corner`, `ca`, `cb` — all degenerate at 0, so the point structure is
   fixed. Fitted jointly over the 7 glyphs (after the feet): n 0.961 → 0.969,
   i 0.973 → 0.989, u 0.956 → 0.968, m 0.943 → 0.948.
4. **Joint fit of the shared parameters over all 16 glyphs** (target
   objective, no glyph allowed below target): H 0.987 → 0.995, m 0.948 →
   0.958, r 0.945 → 0.954.
   h's two overrides became unnecessary: one no longer helped, the other is
   now a kit param `h_dwidth` (like `m_width`, `u_width`).
5. **Per-glyph polish** (kept only if the glyph improves), then a greedy
   coordinate search constrained to Hausdorff ≤ 20 for the four lowest:
   h 0.939 → 0.966, a 0.938 → 0.956, b 0.941 → 0.959.

Arch controls (`n_shoulder_dw`, `n_leg_t`) were added for the joint fit but
did not move by themselves; the remaining error on a, s, g, e is spread
thinly (mean distance ~2 u) with no single feature to fix. The lowest
glyphs, g (Hausdorff 20.0) and t (19.9), sit at the distance limit: their
worst points are where the g's link meets the lower loop and the root of
its ear, and the top of the t's wedge.

### Iteration 08 — the rest of `hamburgefonstiv`: 16 glyphs at target, 3 overrides

Added `h m b u r f t i`. All 16 benchmark glyphs meet the target (IoU ≥ 0.93,
Hausdorff ≤ 20 u, ≤ 2 overrides); 3 overrides in total (`h` 2, `r` 1).
Overlays: `docs/iterations/08/_all.png`.

| glyph | IoU | Hausdorff | overrides | what it took |
|---|---:|---:|---:|---|
| h | 0.934 | 14.0 | 2 | arch joins on the h's (slightly heavier) stem edge; `h.@n_width −6`, `h.@serif_len −10` |
| m | 0.933 | 17.5 | 0 | two arches, own width |
| b | 0.943 | 16.0 | 0 | lobe with a crown node; the stem is chiselled off, not footed |
| u | 0.956 | 11.0 | 0 | own cup (the n's arch turned round is not a u); heads longer than the n's (`u_head_dlen`) |
| r | 0.935 | 15.3 | 1 | ear cut almost vertically (no beak); lopsided foot; `r.@foot_wide_dl −6` |
| f | 0.953 | 15.2 | 0 | hook end/weight (`f_top_dw` thins it across the top), tapered bar, lopsided foot |
| t | 0.943 | 20.0 | 0 | own pen (thick crescent tail), tapered bar |
| i | 0.956 | 17.9 | 0 | — |

New genre-kit controls (all default to the old shapes, so the display serif
is unchanged): `u_head_dlen/dtop`, `r_cut_angle`, `foot_wide` terminal with
`foot_wide_dl/dr/ca/cb` (feet longer on the right, straighter brackets),
tapered crossbars (`bar_taper`, `bar_tip_y`, `bar_tip_dx`), `f_top_dw`,
`f_end_dir/w`, `f_bar_t/dy`, `t_angle/contrast`, `t_bar_t/dy`,
`t_turn_w/end_w`, `b_foot_angle`, `b_crown_x/t`.

Engine fixes found by these glyphs:

* **Terminal cuts bounded by the cap's own corners.** A flat terminal's cut
  box reached half a stroke width past the stroke on each side and sliced a
  slot into the b's bowl (Hausdorff 25.9 → 21.0 before refitting). Its depth
  is likewise only as deep as a tilted cap protrudes; the old depth deleted
  the i's dot (in the display serif too).
* **Merge drops slivers that rounding collapses** to a point or line.
* **Stem lint takes the most common reading,** not the narrowest: the u's
  left stem is short enough that the head wedge fell inside the sampled span.
* **Fit plateaus.** Two coupled controls can be invisible one at a time: the
  bar's tip height does nothing while the taper is 0, so the fitter left the
  bar rectangular (setting both by a probe gave f +0.02). Likewise the IoU is
  measured with bounding-box centres aligned, so lengthening one end of a
  glyph shifts the whole comparison; the r and t needed *both* ends moved
  together (r: foot −6 with ear −7; t: bar and tail ~+18 each), which neither
  Powell nor Nelder–Mead found. The last t step was a greedy coordinate
  search constrained to Hausdorff ≤ 20.
* The r's override is on a knife edge (dv −5 → 0.930, −6 → 0.935): integer
  rounding of the outline moves the bbox centre by half a unit. Worth an
  alignment that doesn't depend on extremes (e.g. area centroid) in a later
  iteration.

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
  for complex ones. **Met by all eight benchmark glyphs as of iteration 06, with 0 overrides; by all 16 (`hamburgefonstiv` + H) as of iteration 08, with 3; with 0 as of iteration 09.**

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
| 08 | 2026-11-01 | `o` | 0.9849 | 1.6 | 0.8 | -14 | 0 | 0 | bowl | 08: rest of hamburgefonstiv (h m b u r f t i); new kit controls (u heads, r flat-cut ear, lopsided feet, tapered f/t bars, f hook top weight, t pen, b chisel foot + lobe crown); terminal cuts bounded by the cap's corners; 3 overrides (h 2, r 1) |
| 08 | 2026-11-01 | `n` | 0.9501 | 17.5 | 1.8 | +29 | 0 | 0 | cut, bracketed, wedge | 08: rest of hamburgefonstiv (h m b u r f t i); new kit controls (u heads, r flat-cut ear, lopsided feet, tapered f/t bars, f hook top weight, t pen, b chisel foot + lobe crown); terminal cuts bounded by the cap's corners; 3 overrides (h 2, r 1) |
| 08 | 2026-11-01 | `H` | 0.9893 | 2.8 | 0.6 | -13 | 0 | 0 | contour, bracketed | 08: rest of hamburgefonstiv (h m b u r f t i); new kit controls (u heads, r flat-cut ear, lopsided feet, tapered f/t bars, f hook top weight, t pen, b chisel foot + lobe crown); terminal cuts bounded by the cap's corners; 3 overrides (h 2, r 1) |
| 08 | 2026-11-01 | `e` | 0.9354 | 6.7 | 2.4 | -25 | 0 | 0 | bowl, contour, cut | 08: rest of hamburgefonstiv (h m b u r f t i); new kit controls (u heads, r flat-cut ear, lopsided feet, tapered f/t bars, f hook top weight, t pen, b chisel foot + lobe crown); terminal cuts bounded by the cap's corners; 3 overrides (h 2, r 1) |
| 08 | 2026-11-01 | `a` | 0.9370 | 7.8 | 2.1 | +2 | 0 | 0 | lobe, beak, wedge | 08: rest of hamburgefonstiv (h m b u r f t i); new kit controls (u heads, r flat-cut ear, lopsided feet, tapered f/t bars, f hook top weight, t pen, b chisel foot + lobe crown); terminal cuts bounded by the cap's corners; 3 overrides (h 2, r 1) |
| 08 | 2026-11-01 | `s` | 0.9395 | 6.8 | 1.8 | +2 | 0 | 0 | beak | 08: rest of hamburgefonstiv (h m b u r f t i); new kit controls (u heads, r flat-cut ear, lopsided feet, tapered f/t bars, f hook top weight, t pen, b chisel foot + lobe crown); terminal cuts bounded by the cap's corners; 3 overrides (h 2, r 1) |
| 08 | 2026-11-01 | `g` | 0.9312 | 18.7 | 2.2 | +67 | 0 | 0 | bowl, contour, cut | 08: rest of hamburgefonstiv (h m b u r f t i); new kit controls (u heads, r flat-cut ear, lopsided feet, tapered f/t bars, f hook top weight, t pen, b chisel foot + lobe crown); terminal cuts bounded by the cap's corners; 3 overrides (h 2, r 1) |
| 08 | 2026-11-01 | `v` | 0.9597 | 15.0 | 1.3 | +0 | 0 | 0 | cut, bracketed, flat | 08: rest of hamburgefonstiv (h m b u r f t i); new kit controls (u heads, r flat-cut ear, lopsided feet, tapered f/t bars, f hook top weight, t pen, b chisel foot + lobe crown); terminal cuts bounded by the cap's corners; 3 overrides (h 2, r 1) |
| 08 | 2026-11-01 | `h` | 0.9337 | 14.0 | 2.4 | +8 | 2 | 0 | cut, bracketed, wedge | 08: rest of hamburgefonstiv (h m b u r f t i); new kit controls (u heads, r flat-cut ear, lopsided feet, tapered f/t bars, f hook top weight, t pen, b chisel foot + lobe crown); terminal cuts bounded by the cap's corners; 3 overrides (h 2, r 1) |
| 08 | 2026-11-01 | `m` | 0.9326 | 17.5 | 2.5 | +29 | 0 | 0 | cut, bracketed, wedge | 08: rest of hamburgefonstiv (h m b u r f t i); new kit controls (u heads, r flat-cut ear, lopsided feet, tapered f/t bars, f hook top weight, t pen, b chisel foot + lobe crown); terminal cuts bounded by the cap's corners; 3 overrides (h 2, r 1) |
| 08 | 2026-11-01 | `u` | 0.9561 | 11.0 | 1.9 | +2 | 0 | 0 | cut, wedge | 08: rest of hamburgefonstiv (h m b u r f t i); new kit controls (u heads, r flat-cut ear, lopsided feet, tapered f/t bars, f hook top weight, t pen, b chisel foot + lobe crown); terminal cuts bounded by the cap's corners; 3 overrides (h 2, r 1) |
| 08 | 2026-11-01 | `b` | 0.9425 | 16.0 | 2.1 | -22 | 0 | 0 | lobe, flat, wedge | 08: rest of hamburgefonstiv (h m b u r f t i); new kit controls (u heads, r flat-cut ear, lopsided feet, tapered f/t bars, f hook top weight, t pen, b chisel foot + lobe crown); terminal cuts bounded by the cap's corners; 3 overrides (h 2, r 1) |
| 08 | 2026-11-01 | `r` | 0.9346 | 15.3 | 2.2 | +43 | 1 | 0 | bracketed, flat, wedge | 08: rest of hamburgefonstiv (h m b u r f t i); new kit controls (u heads, r flat-cut ear, lopsided feet, tapered f/t bars, f hook top weight, t pen, b chisel foot + lobe crown); terminal cuts bounded by the cap's corners; 3 overrides (h 2, r 1) |
| 08 | 2026-11-01 | `f` | 0.9534 | 15.2 | 1.6 | +71 | 0 | 0 | contour, bracketed, flat | 08: rest of hamburgefonstiv (h m b u r f t i); new kit controls (u heads, r flat-cut ear, lopsided feet, tapered f/t bars, f hook top weight, t pen, b chisel foot + lobe crown); terminal cuts bounded by the cap's corners; 3 overrides (h 2, r 1) |
| 08 | 2026-11-01 | `t` | 0.9426 | 20.0 | 2.1 | +69 | 0 | 0 | contour, flat, wedge | 08: rest of hamburgefonstiv (h m b u r f t i); new kit controls (u heads, r flat-cut ear, lopsided feet, tapered f/t bars, f hook top weight, t pen, b chisel foot + lobe crown); terminal cuts bounded by the cap's corners; 3 overrides (h 2, r 1) |
| 08 | 2026-11-01 | `i` | 0.9564 | 17.9 | 1.6 | +75 | 0 | 0 | contour, bracketed, wedge | 08: rest of hamburgefonstiv (h m b u r f t i); new kit controls (u heads, r flat-cut ear, lopsided feet, tapered f/t bars, f hook top weight, t pen, b chisel foot + lobe crown); terminal cuts bounded by the cap's corners; 3 overrides (h 2, r 1) |
| 09 | 2026-11-01 | `o` | 0.9840 | 1.8 | 0.8 | -14 | 0 | 0 | bowl | 09: exact IoU at the best offset; head wedge (flat tip/top, bracketed underside) and straight-bracket feet fitted; joint fit of shared params over 16 glyphs; per-glyph polish + constrained greedy; 0 overrides |
| 09 | 2026-11-01 | `n` | 0.9675 | 11.7 | 1.3 | +1 | 0 | 0 | cut, bracketed, wedge | 09: exact IoU at the best offset; head wedge (flat tip/top, bracketed underside) and straight-bracket feet fitted; joint fit of shared params over 16 glyphs; per-glyph polish + constrained greedy; 0 overrides |
| 09 | 2026-11-01 | `H` | 0.9948 | 2.0 | 0.4 | -10 | 0 | 0 | contour, bracketed | 09: exact IoU at the best offset; head wedge (flat tip/top, bracketed underside) and straight-bracket feet fitted; joint fit of shared params over 16 glyphs; per-glyph polish + constrained greedy; 0 overrides |
| 09 | 2026-11-01 | `e` | 0.9506 | 7.5 | 1.8 | -25 | 0 | 0 | bowl, contour, cut | 09: exact IoU at the best offset; head wedge (flat tip/top, bracketed underside) and straight-bracket feet fitted; joint fit of shared params over 16 glyphs; per-glyph polish + constrained greedy; 0 overrides |
| 09 | 2026-11-01 | `a` | 0.9562 | 13.3 | 1.6 | +1 | 0 | 0 | lobe, beak, wedge | 09: exact IoU at the best offset; head wedge (flat tip/top, bracketed underside) and straight-bracket feet fitted; joint fit of shared params over 16 glyphs; per-glyph polish + constrained greedy; 0 overrides |
| 09 | 2026-11-01 | `s` | 0.9414 | 6.0 | 1.8 | +1 | 0 | 0 | beak | 09: exact IoU at the best offset; head wedge (flat tip/top, bracketed underside) and straight-bracket feet fitted; joint fit of shared params over 16 glyphs; per-glyph polish + constrained greedy; 0 overrides |
| 09 | 2026-11-01 | `g` | 0.9386 | 20.0 | 2.1 | +66 | 0 | 0 | bowl, contour, cut | 09: exact IoU at the best offset; head wedge (flat tip/top, bracketed underside) and straight-bracket feet fitted; joint fit of shared params over 16 glyphs; per-glyph polish + constrained greedy; 0 overrides |
| 09 | 2026-11-01 | `v` | 0.9660 | 16.9 | 1.2 | +0 | 0 | 0 | cut, bracketed, flat | 09: exact IoU at the best offset; head wedge (flat tip/top, bracketed underside) and straight-bracket feet fitted; joint fit of shared params over 16 glyphs; per-glyph polish + constrained greedy; 0 overrides |
| 09 | 2026-11-01 | `h` | 0.9655 | 10.8 | 1.4 | -2 | 0 | 0 | cut, bracketed, wedge | 09: exact IoU at the best offset; head wedge (flat tip/top, bracketed underside) and straight-bracket feet fitted; joint fit of shared params over 16 glyphs; per-glyph polish + constrained greedy; 0 overrides |
| 09 | 2026-11-01 | `m` | 0.9589 | 11.7 | 1.6 | +0 | 0 | 0 | cut, bracketed, wedge | 09: exact IoU at the best offset; head wedge (flat tip/top, bracketed underside) and straight-bracket feet fitted; joint fit of shared params over 16 glyphs; per-glyph polish + constrained greedy; 0 overrides |
| 09 | 2026-11-01 | `u` | 0.9715 | 7.2 | 1.2 | +0 | 0 | 0 | cut, wedge | 09: exact IoU at the best offset; head wedge (flat tip/top, bracketed underside) and straight-bracket feet fitted; joint fit of shared params over 16 glyphs; per-glyph polish + constrained greedy; 0 overrides |
| 09 | 2026-11-01 | `b` | 0.9594 | 12.0 | 1.8 | -16 | 0 | 0 | lobe, flat, wedge | 09: exact IoU at the best offset; head wedge (flat tip/top, bracketed underside) and straight-bracket feet fitted; joint fit of shared params over 16 glyphs; per-glyph polish + constrained greedy; 0 overrides |
| 09 | 2026-11-01 | `r` | 0.9587 | 8.9 | 1.4 | +30 | 0 | 0 | bracketed, flat, wedge | 09: exact IoU at the best offset; head wedge (flat tip/top, bracketed underside) and straight-bracket feet fitted; joint fit of shared params over 16 glyphs; per-glyph polish + constrained greedy; 0 overrides |
| 09 | 2026-11-01 | `f` | 0.9649 | 12.3 | 1.2 | +64 | 0 | 0 | contour, bracketed, flat | 09: exact IoU at the best offset; head wedge (flat tip/top, bracketed underside) and straight-bracket feet fitted; joint fit of shared params over 16 glyphs; per-glyph polish + constrained greedy; 0 overrides |
| 09 | 2026-11-01 | `t` | 0.9458 | 19.9 | 2.0 | +67 | 0 | 0 | contour, flat, wedge | 09: exact IoU at the best offset; head wedge (flat tip/top, bracketed underside) and straight-bracket feet fitted; joint fit of shared params over 16 glyphs; per-glyph polish + constrained greedy; 0 overrides |
| 09 | 2026-11-01 | `i` | 0.9888 | 6.1 | 0.5 | +47 | 0 | 0 | contour, bracketed, wedge | 09: exact IoU at the best offset; head wedge (flat tip/top, bracketed underside) and straight-bracket feet fitted; joint fit of shared params over 16 glyphs; per-glyph polish + constrained greedy; 0 overrides |
| 10 | 2026-11-01 | `o` | 0.9840 | 1.8 | 0.8 | -14 | 0 | 0 | bowl | 10: t head with flat top + concave flank (wedge sweep); g lower loop/link notch and upper-bowl shoulder; merge drops needle spikes; 0 overrides |
| 10 | 2026-11-01 | `n` | 0.9675 | 11.7 | 1.3 | +1 | 0 | 0 | cut, bracketed, wedge | 10: t head with flat top + concave flank (wedge sweep); g lower loop/link notch and upper-bowl shoulder; merge drops needle spikes; 0 overrides |
| 10 | 2026-11-01 | `H` | 0.9948 | 2.0 | 0.4 | -10 | 0 | 0 | contour, bracketed | 10: t head with flat top + concave flank (wedge sweep); g lower loop/link notch and upper-bowl shoulder; merge drops needle spikes; 0 overrides |
| 10 | 2026-11-01 | `e` | 0.9506 | 7.5 | 1.8 | -25 | 0 | 0 | bowl, contour, cut | 10: t head with flat top + concave flank (wedge sweep); g lower loop/link notch and upper-bowl shoulder; merge drops needle spikes; 0 overrides |
| 10 | 2026-11-01 | `a` | 0.9562 | 13.3 | 1.6 | +1 | 0 | 0 | lobe, beak, wedge | 10: t head with flat top + concave flank (wedge sweep); g lower loop/link notch and upper-bowl shoulder; merge drops needle spikes; 0 overrides |
| 10 | 2026-11-01 | `s` | 0.9414 | 6.0 | 1.8 | +1 | 0 | 0 | beak | 10: t head with flat top + concave flank (wedge sweep); g lower loop/link notch and upper-bowl shoulder; merge drops needle spikes; 0 overrides |
| 10 | 2026-11-01 | `g` | 0.9456 | 15.4 | 1.8 | +66 | 0 | 0 | bowl, contour, cut | 10: t head with flat top + concave flank (wedge sweep); g lower loop/link notch and upper-bowl shoulder; merge drops needle spikes; 0 overrides |
| 10 | 2026-11-01 | `v` | 0.9660 | 16.9 | 1.2 | +0 | 0 | 0 | cut, bracketed, flat | 10: t head with flat top + concave flank (wedge sweep); g lower loop/link notch and upper-bowl shoulder; merge drops needle spikes; 0 overrides |
| 10 | 2026-11-01 | `h` | 0.9655 | 10.8 | 1.4 | -2 | 0 | 0 | cut, bracketed, wedge | 10: t head with flat top + concave flank (wedge sweep); g lower loop/link notch and upper-bowl shoulder; merge drops needle spikes; 0 overrides |
| 10 | 2026-11-01 | `m` | 0.9589 | 11.7 | 1.6 | +0 | 0 | 0 | cut, bracketed, wedge | 10: t head with flat top + concave flank (wedge sweep); g lower loop/link notch and upper-bowl shoulder; merge drops needle spikes; 0 overrides |
| 10 | 2026-11-01 | `u` | 0.9715 | 7.2 | 1.2 | +0 | 0 | 0 | cut, wedge | 10: t head with flat top + concave flank (wedge sweep); g lower loop/link notch and upper-bowl shoulder; merge drops needle spikes; 0 overrides |
| 10 | 2026-11-01 | `b` | 0.9594 | 12.0 | 1.8 | -16 | 0 | 0 | lobe, flat, wedge | 10: t head with flat top + concave flank (wedge sweep); g lower loop/link notch and upper-bowl shoulder; merge drops needle spikes; 0 overrides |
| 10 | 2026-11-01 | `r` | 0.9587 | 8.9 | 1.4 | +30 | 0 | 0 | bracketed, flat, wedge | 10: t head with flat top + concave flank (wedge sweep); g lower loop/link notch and upper-bowl shoulder; merge drops needle spikes; 0 overrides |
| 10 | 2026-11-01 | `f` | 0.9649 | 12.3 | 1.2 | +64 | 0 | 0 | contour, bracketed, flat | 10: t head with flat top + concave flank (wedge sweep); g lower loop/link notch and upper-bowl shoulder; merge drops needle spikes; 0 overrides |
| 10 | 2026-11-01 | `t` | 0.9575 | 16.5 | 1.8 | +67 | 0 | 0 | contour, flat, wedge | 10: t head with flat top + concave flank (wedge sweep); g lower loop/link notch and upper-bowl shoulder; merge drops needle spikes; 0 overrides |
| 10 | 2026-11-01 | `i` | 0.9888 | 6.1 | 0.5 | +47 | 0 | 0 | contour, bracketed, wedge | 10: t head with flat top + concave flank (wedge sweep); g lower loop/link notch and upper-bowl shoulder; merge drops needle spikes; 0 overrides |
| 11 | 2026-11-01 | `o` | 0.9840 | 1.8 | 0.8 | -14 | 0 | 0 | bowl | 11: stroke ends -- ends metric + ends objective; diagonal brackets, head inset at joins, t tail cut angle, spur shape controls; s and a hood as plain cuts (no beak artifacts); 0 overrides |
| 11 | 2026-11-01 | `n` | 0.9728 | 8.7 | 1.1 | +1 | 0 | 0 | cut, bracketed, wedge | 11: stroke ends -- ends metric + ends objective; diagonal brackets, head inset at joins, t tail cut angle, spur shape controls; s and a hood as plain cuts (no beak artifacts); 0 overrides |
| 11 | 2026-11-01 | `H` | 0.9949 | 1.8 | 0.4 | -10 | 0 | 0 | contour, bracketed | 11: stroke ends -- ends metric + ends objective; diagonal brackets, head inset at joins, t tail cut angle, spur shape controls; s and a hood as plain cuts (no beak artifacts); 0 overrides |
| 11 | 2026-11-01 | `e` | 0.9506 | 7.5 | 1.8 | -25 | 0 | 0 | bowl, contour, cut | 11: stroke ends -- ends metric + ends objective; diagonal brackets, head inset at joins, t tail cut angle, spur shape controls; s and a hood as plain cuts (no beak artifacts); 0 overrides |
| 11 | 2026-11-01 | `a` | 0.9586 | 13.9 | 1.8 | +5 | 0 | 0 | lobe, flat, wedge | 11: stroke ends -- ends metric + ends objective; diagonal brackets, head inset at joins, t tail cut angle, spur shape controls; s and a hood as plain cuts (no beak artifacts); 0 overrides |
| 11 | 2026-11-01 | `s` | 0.9466 | 7.7 | 1.7 | +2 | 0 | 0 | flat | 11: stroke ends -- ends metric + ends objective; diagonal brackets, head inset at joins, t tail cut angle, spur shape controls; s and a hood as plain cuts (no beak artifacts); 0 overrides |
| 11 | 2026-11-01 | `g` | 0.9456 | 15.4 | 1.8 | +66 | 0 | 0 | bowl, contour, cut | 11: stroke ends -- ends metric + ends objective; diagonal brackets, head inset at joins, t tail cut angle, spur shape controls; s and a hood as plain cuts (no beak artifacts); 0 overrides |
| 11 | 2026-11-01 | `v` | 0.9732 | 16.3 | 0.9 | +7 | 0 | 0 | cut, bracketed, flat | 11: stroke ends -- ends metric + ends objective; diagonal brackets, head inset at joins, t tail cut angle, spur shape controls; s and a hood as plain cuts (no beak artifacts); 0 overrides |
| 11 | 2026-11-01 | `h` | 0.9661 | 10.8 | 1.3 | -2 | 0 | 0 | cut, bracketed, wedge | 11: stroke ends -- ends metric + ends objective; diagonal brackets, head inset at joins, t tail cut angle, spur shape controls; s and a hood as plain cuts (no beak artifacts); 0 overrides |
| 11 | 2026-11-01 | `m` | 0.9624 | 11.5 | 1.5 | +0 | 0 | 0 | cut, bracketed, wedge | 11: stroke ends -- ends metric + ends objective; diagonal brackets, head inset at joins, t tail cut angle, spur shape controls; s and a hood as plain cuts (no beak artifacts); 0 overrides |
| 11 | 2026-11-01 | `u` | 0.9767 | 7.2 | 1.0 | -2 | 0 | 0 | cut, wedge | 11: stroke ends -- ends metric + ends objective; diagonal brackets, head inset at joins, t tail cut angle, spur shape controls; s and a hood as plain cuts (no beak artifacts); 0 overrides |
| 11 | 2026-11-01 | `b` | 0.9593 | 16.0 | 1.7 | -20 | 0 | 0 | lobe, flat, wedge | 11: stroke ends -- ends metric + ends objective; diagonal brackets, head inset at joins, t tail cut angle, spur shape controls; s and a hood as plain cuts (no beak artifacts); 0 overrides |
| 11 | 2026-11-01 | `r` | 0.9760 | 8.1 | 0.9 | +33 | 0 | 0 | cut, bracketed, flat, wedge | 11: stroke ends -- ends metric + ends objective; diagonal brackets, head inset at joins, t tail cut angle, spur shape controls; s and a hood as plain cuts (no beak artifacts); 0 overrides |
| 11 | 2026-11-01 | `f` | 0.9734 | 12.3 | 0.9 | +64 | 0 | 0 | contour, bracketed, flat | 11: stroke ends -- ends metric + ends objective; diagonal brackets, head inset at joins, t tail cut angle, spur shape controls; s and a hood as plain cuts (no beak artifacts); 0 overrides |
| 11 | 2026-11-01 | `t` | 0.9589 | 9.7 | 1.7 | +74 | 0 | 0 | contour, flat, wedge | 11: stroke ends -- ends metric + ends objective; diagonal brackets, head inset at joins, t tail cut angle, spur shape controls; s and a hood as plain cuts (no beak artifacts); 0 overrides |
| 11 | 2026-11-01 | `i` | 0.9918 | 10.0 | 0.4 | +47 | 0 | 0 | contour, bracketed, wedge | 11: stroke ends -- ends metric + ends objective; diagonal brackets, head inset at joins, t tail cut angle, spur shape controls; s and a hood as plain cuts (no beak artifacts); 0 overrides |
| 12 | 2026-11-01 | `o` | 0.9840 | 1.8 | 0.8 | -14 | 0 | 0 | bowl | 12: g link/loop notch (cut); s leaning extremes, per-node tension, per-end weight; 0 overrides |
| 12 | 2026-11-01 | `n` | 0.9728 | 8.7 | 1.1 | +1 | 0 | 0 | cut, bracketed, wedge | 12: g link/loop notch (cut); s leaning extremes, per-node tension, per-end weight; 0 overrides |
| 12 | 2026-11-01 | `H` | 0.9949 | 1.8 | 0.4 | -10 | 0 | 0 | contour, bracketed | 12: g link/loop notch (cut); s leaning extremes, per-node tension, per-end weight; 0 overrides |
| 12 | 2026-11-01 | `e` | 0.9506 | 7.5 | 1.8 | -25 | 0 | 0 | bowl, contour, cut | 12: g link/loop notch (cut); s leaning extremes, per-node tension, per-end weight; 0 overrides |
| 12 | 2026-11-01 | `a` | 0.9586 | 13.9 | 1.8 | +5 | 0 | 0 | lobe, flat, wedge | 12: g link/loop notch (cut); s leaning extremes, per-node tension, per-end weight; 0 overrides |
| 12 | 2026-11-01 | `s` | 0.9544 | 10.1 | 1.5 | +3 | 0 | 0 | flat | 12: g link/loop notch (cut); s leaning extremes, per-node tension, per-end weight; 0 overrides |
| 12 | 2026-11-01 | `g` | 0.9615 | 16.2 | 1.4 | +66 | 0 | 0 | bowl, contour, cut | 12: g link/loop notch (cut); s leaning extremes, per-node tension, per-end weight; 0 overrides |
| 12 | 2026-11-01 | `v` | 0.9732 | 16.3 | 0.9 | +7 | 0 | 0 | cut, bracketed, flat | 12: g link/loop notch (cut); s leaning extremes, per-node tension, per-end weight; 0 overrides |
| 12 | 2026-11-01 | `h` | 0.9661 | 10.8 | 1.3 | -2 | 0 | 0 | cut, bracketed, wedge | 12: g link/loop notch (cut); s leaning extremes, per-node tension, per-end weight; 0 overrides |
| 12 | 2026-11-01 | `m` | 0.9624 | 11.5 | 1.5 | +0 | 0 | 0 | cut, bracketed, wedge | 12: g link/loop notch (cut); s leaning extremes, per-node tension, per-end weight; 0 overrides |
| 12 | 2026-11-01 | `u` | 0.9767 | 7.2 | 1.0 | -2 | 0 | 0 | cut, wedge | 12: g link/loop notch (cut); s leaning extremes, per-node tension, per-end weight; 0 overrides |
| 12 | 2026-11-01 | `b` | 0.9593 | 16.0 | 1.7 | -20 | 0 | 0 | lobe, flat, wedge | 12: g link/loop notch (cut); s leaning extremes, per-node tension, per-end weight; 0 overrides |
| 12 | 2026-11-01 | `r` | 0.9760 | 8.1 | 0.9 | +33 | 0 | 0 | cut, bracketed, flat, wedge | 12: g link/loop notch (cut); s leaning extremes, per-node tension, per-end weight; 0 overrides |
| 12 | 2026-11-01 | `f` | 0.9734 | 12.3 | 0.9 | +64 | 0 | 0 | contour, bracketed, flat | 12: g link/loop notch (cut); s leaning extremes, per-node tension, per-end weight; 0 overrides |
| 12 | 2026-11-01 | `t` | 0.9589 | 9.7 | 1.7 | +74 | 0 | 0 | contour, flat, wedge | 12: g link/loop notch (cut); s leaning extremes, per-node tension, per-end weight; 0 overrides |
| 12 | 2026-11-01 | `i` | 0.9918 | 10.0 | 0.4 | +47 | 0 | 0 | contour, bracketed, wedge | 12: g link/loop notch (cut); s leaning extremes, per-node tension, per-end weight; 0 overrides |
| 13 | 2026-11-01 | `o` | 0.9840 | 1.8 | 0.8 | -14 | 0 | 0 | bowl | 13: refit the rest -- unpinned ranges, e walls/bar, coupled-move and multi-glyph search (n/h/m arches), metric guard against consistent skia errors; 0 overrides |
| 13 | 2026-11-01 | `n` | 0.9817 | 6.8 | 0.8 | +1 | 0 | 0 | cut, bracketed, wedge | 13: refit the rest -- unpinned ranges, e walls/bar, coupled-move and multi-glyph search (n/h/m arches), metric guard against consistent skia errors; 0 overrides |
| 13 | 2026-11-01 | `H` | 0.9949 | 1.8 | 0.4 | -10 | 0 | 0 | contour, bracketed | 13: refit the rest -- unpinned ranges, e walls/bar, coupled-move and multi-glyph search (n/h/m arches), metric guard against consistent skia errors; 0 overrides |
| 13 | 2026-11-01 | `e` | 0.9692 | 8.3 | 1.2 | -25 | 0 | 0 | bowl, contour, cut | 13: refit the rest -- unpinned ranges, e walls/bar, coupled-move and multi-glyph search (n/h/m arches), metric guard against consistent skia errors; 0 overrides |
| 13 | 2026-11-01 | `a` | 0.9611 | 12.8 | 1.7 | +5 | 0 | 0 | lobe, flat, wedge | 13: refit the rest -- unpinned ranges, e walls/bar, coupled-move and multi-glyph search (n/h/m arches), metric guard against consistent skia errors; 0 overrides |
| 13 | 2026-11-01 | `s` | 0.9550 | 10.1 | 1.5 | +3 | 0 | 0 | flat | 13: refit the rest -- unpinned ranges, e walls/bar, coupled-move and multi-glyph search (n/h/m arches), metric guard against consistent skia errors; 0 overrides |
| 13 | 2026-11-01 | `g` | 0.9664 | 16.2 | 1.3 | +66 | 0 | 0 | bowl, contour, cut | 13: refit the rest -- unpinned ranges, e walls/bar, coupled-move and multi-glyph search (n/h/m arches), metric guard against consistent skia errors; 0 overrides |
| 13 | 2026-11-01 | `v` | 0.9757 | 16.4 | 0.9 | +9 | 0 | 0 | cut, bracketed, flat | 13: refit the rest -- unpinned ranges, e walls/bar, coupled-move and multi-glyph search (n/h/m arches), metric guard against consistent skia errors; 0 overrides |
| 13 | 2026-11-01 | `h` | 0.9743 | 8.6 | 1.1 | -2 | 0 | 0 | cut, bracketed, wedge | 13: refit the rest -- unpinned ranges, e walls/bar, coupled-move and multi-glyph search (n/h/m arches), metric guard against consistent skia errors; 0 overrides |
| 13 | 2026-11-01 | `m` | 0.9692 | 9.6 | 1.3 | +0 | 0 | 0 | cut, bracketed, wedge | 13: refit the rest -- unpinned ranges, e walls/bar, coupled-move and multi-glyph search (n/h/m arches), metric guard against consistent skia errors; 0 overrides |
| 13 | 2026-11-01 | `u` | 0.9809 | 14.6 | 1.0 | -4 | 0 | 0 | cut, wedge | 13: refit the rest -- unpinned ranges, e walls/bar, coupled-move and multi-glyph search (n/h/m arches), metric guard against consistent skia errors; 0 overrides |
| 13 | 2026-11-01 | `b` | 0.9752 | 16.0 | 1.2 | -12 | 0 | 0 | lobe, flat, wedge | 13: refit the rest -- unpinned ranges, e walls/bar, coupled-move and multi-glyph search (n/h/m arches), metric guard against consistent skia errors; 0 overrides |
| 13 | 2026-11-01 | `r` | 0.9795 | 7.3 | 1.0 | +29 | 0 | 0 | cut, bracketed, flat, wedge | 13: refit the rest -- unpinned ranges, e walls/bar, coupled-move and multi-glyph search (n/h/m arches), metric guard against consistent skia errors; 0 overrides |
| 13 | 2026-11-01 | `f` | 0.9776 | 12.3 | 0.9 | +70 | 0 | 0 | contour, bracketed, flat | 13: refit the rest -- unpinned ranges, e walls/bar, coupled-move and multi-glyph search (n/h/m arches), metric guard against consistent skia errors; 0 overrides |
| 13 | 2026-11-01 | `t` | 0.9793 | 9.7 | 1.5 | +76 | 0 | 0 | contour, flat, wedge | 13: refit the rest -- unpinned ranges, e walls/bar, coupled-move and multi-glyph search (n/h/m arches), metric guard against consistent skia errors; 0 overrides |
| 13 | 2026-11-01 | `i` | 0.9918 | 10.0 | 0.4 | +47 | 0 | 0 | contour, bracketed, wedge | 13: refit the rest -- unpinned ranges, e walls/bar, coupled-move and multi-glyph search (n/h/m arches), metric guard against consistent skia errors; 0 overrides |
| 14 | 2026-11-01 | `o` | 0.9840 | 1.8 | 0.8 | -14 | 0 | 0 | bowl | 14: pen angle varies along strokes (node pen_angle / dp override); s and a hood refitted; 0 overrides |
| 14 | 2026-11-01 | `n` | 0.9817 | 6.8 | 0.8 | +1 | 0 | 0 | cut, bracketed, wedge | 14: pen angle varies along strokes (node pen_angle / dp override); s and a hood refitted; 0 overrides |
| 14 | 2026-11-01 | `H` | 0.9949 | 1.8 | 0.4 | -10 | 0 | 0 | contour, bracketed | 14: pen angle varies along strokes (node pen_angle / dp override); s and a hood refitted; 0 overrides |
| 14 | 2026-11-01 | `e` | 0.9692 | 8.3 | 1.2 | -25 | 0 | 0 | bowl, contour, cut | 14: pen angle varies along strokes (node pen_angle / dp override); s and a hood refitted; 0 overrides |
| 14 | 2026-11-01 | `a` | 0.9620 | 12.8 | 1.7 | +5 | 0 | 0 | lobe, flat, wedge | 14: pen angle varies along strokes (node pen_angle / dp override); s and a hood refitted; 0 overrides |
| 14 | 2026-11-01 | `s` | 0.9598 | 11.1 | 1.5 | +3 | 0 | 0 | flat | 14: pen angle varies along strokes (node pen_angle / dp override); s and a hood refitted; 0 overrides |
| 14 | 2026-11-01 | `g` | 0.9664 | 16.2 | 1.3 | +66 | 0 | 0 | bowl, contour, cut | 14: pen angle varies along strokes (node pen_angle / dp override); s and a hood refitted; 0 overrides |
| 14 | 2026-11-01 | `v` | 0.9757 | 16.4 | 0.9 | +9 | 0 | 0 | cut, bracketed, flat | 14: pen angle varies along strokes (node pen_angle / dp override); s and a hood refitted; 0 overrides |
| 14 | 2026-11-01 | `h` | 0.9744 | 8.1 | 1.1 | -2 | 0 | 0 | cut, bracketed, wedge | 14: pen angle varies along strokes (node pen_angle / dp override); s and a hood refitted; 0 overrides |
| 14 | 2026-11-01 | `m` | 0.9692 | 9.6 | 1.3 | +0 | 0 | 0 | cut, bracketed, wedge | 14: pen angle varies along strokes (node pen_angle / dp override); s and a hood refitted; 0 overrides |
| 14 | 2026-11-01 | `u` | 0.9809 | 14.6 | 1.0 | -4 | 0 | 0 | cut, wedge | 14: pen angle varies along strokes (node pen_angle / dp override); s and a hood refitted; 0 overrides |
| 14 | 2026-11-01 | `b` | 0.9752 | 16.0 | 1.2 | -12 | 0 | 0 | lobe, flat, wedge | 14: pen angle varies along strokes (node pen_angle / dp override); s and a hood refitted; 0 overrides |
| 14 | 2026-11-01 | `r` | 0.9795 | 7.3 | 1.0 | +29 | 0 | 0 | cut, bracketed, flat, wedge | 14: pen angle varies along strokes (node pen_angle / dp override); s and a hood refitted; 0 overrides |
| 14 | 2026-11-01 | `f` | 0.9776 | 12.3 | 0.9 | +70 | 0 | 0 | contour, bracketed, flat | 14: pen angle varies along strokes (node pen_angle / dp override); s and a hood refitted; 0 overrides |
| 14 | 2026-11-01 | `t` | 0.9793 | 9.7 | 1.5 | +76 | 0 | 0 | contour, flat, wedge | 14: pen angle varies along strokes (node pen_angle / dp override); s and a hood refitted; 0 overrides |
| 14 | 2026-11-01 | `i` | 0.9918 | 10.0 | 0.4 | +47 | 0 | 0 | contour, bracketed, wedge | 14: pen angle varies along strokes (node pen_angle / dp override); s and a hood refitted; 0 overrides |
