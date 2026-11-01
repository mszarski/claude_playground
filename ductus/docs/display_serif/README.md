# Ductus Display — iteration log

An **original** high-contrast display serif (`examples/display_serif/`): same
genre as the benchmark, but every value is a design decision made by eye and
never fitted to a reference. Sheets: `NN.png` (regenerate with
`python examples/display_serif/make_sheet.py NN`).

## 01 — `o`, `O` from the bowl primitive

Choices: contrast 0.85 (hairline ≈ 15% of stem), 6° oblique stress for a
little warmth, rounds 5% heavier than stems, squarer outer contour than
counter, tight round sidebearings (0.36 × stem).

Critique:
- **Contrast** — reads as Didone-adjacent; good at 120 px. At 18 px the 14-unit
  hairlines are ~0.25 px and vanish: expected for display, an M4 concern.
- **Proportions** — `o` too condensed (468 wide); `O` (1.36 × o) too narrow
  next to it, reads as an ellipse, not a round.
- **Curves** — the rounder counter (0.52 fullness) pinches at top and bottom:
  the counter looks almond-shaped. The lumpiest thing on the sheet.
- **Colour / spacing** — can't judge yet with only rounds.

## 02 — proportions and counter

Changes: `o` 468 → 500; `O` now `0.95 × capheight` (≈ 675) instead of a ratio
of `o`; counter top/bottom fullness 0.52 → 0.56.

Critique:
- `O` now reads round and sits well against `o`; the pinch in the counter is
  gone and the hairlines at top/bottom stay crisp.
- The counter is still narrow (≈ 0.59 of its height) — characteristic of the
  genre (the benchmark has the same counter width), keep for now and revisit
  once `n` sets the colour of the straights.
- The 6° stress shows as a slight diagonal lean of the thin points: subtle at
  120 px, invisible at 48 px. Decide once there are straights to compare.
- Next: `n` with the junction primitive, then judge evenness of colour.

## 03 — the whole test word, from the serif kit

The design moved onto the shared serif kit (`ductus/kits/serif.py`) with its
own values and the **Didone terminal set**: flat hairline serifs with a
whisper of bracket, ball terminals, vertical hairline beards — a different
direction from the benchmark's sharp bracketed/wedge serifs. All of
`hamburgefonstiv` and `H O A G E S` exist.

Critique (sheet `03.png`):
- **Bugs, not design:** stray serifs halfway up every n/h/m leg (the leg
  stroke inherited the stem's default `foot` tag at its top); the space was
  missing (glyph filter only matched single characters).
- **s / S** — mangled: terminals started travelling vertically, so beak cuts
  ran along the stroke and projected far away (spikes).
- **G** — upper arm stopped low and thick; beard floating off the bowl.
- **Ball terminals** barely visible: balls sat back inside the stroke, flush
  with the wrong edge.
- E, A, H, O already credible.

## 04 — terminals and the s

Changes: kit fixes (leg tags, space, cut conditioning); s/S terminals now
leave horizontally so their faces are vertical with hairline beards; spine
swell concentrated mid-stroke (`s_spread`); balls hang flush with the lower
edge and slightly past the end; G's aperture steps around its beard.

Critique (sheet `04.png`):
- **Contrast** — consistent Didone contrast across lower and upper case; the
  hairlines hold at 36 px and up, vanish at 14 px (expected; M4).
- **Serif consistency** — feet, heads, A/v top serifs all read as one family.
  The `u` cup reads spidery next to the `n` arch. (First guess was that the
  rotation put the hairline in the wrong place — wrong: thin join at the
  bottom-right and thick left wall *is* the Didone u. The cause is the
  narrow, tall arch proportion, shared with n.)
- **Terminals** — balls on a, r, f give the face its character. The `e`
  terminal is a blob; it wants a thin upturned hairline.
- **Colour / spacing** — `a`→`m` and `m`→`b` are too tight (ball and bowl
  against serifs), `r`→`g` too open. Straight sidebearing 34 is too small
  next to 62-unit serifs.
- **Lumpy curves** — s spine is smoother but still slightly calligraphic;
  g's lower loop is heavy at the bottom.

## 05 — g, e, spacing

Changes: g rebuilt in the kit (lower loop is now a contour bowl with its own
walls, link is a hairline stroke leaving the bowl's bottom); e aperture edge
bowed (`e_term_sag`) so the terminal tapers; straight sidebearing 34 → 46,
round 30 → 34; a's hood always ends inside the stem (there was a white gap).

Critique (sheet `05.png`):
- **g** — now a credible Didone two-storey g: hairline link, loop with thick
  sides and hairline top/bottom, ball ear. The ear ball is a little small.
- **e** — terminal tapers to an upturned point; reads right at 120 px.
- **Spacing / colour** — much more even: `ham`, `mbu` no longer collide. `r`
  still leaves a hole before `g`; `t`→`i` slightly tight. Straights vs rounds
  look balanced at 72 px.
- **Still weak** — m arches are narrow (m_width 330 against n 400) and read
  condensed; the u cup still looks lighter than the n arch; S/s spine is
  smooth but the terminals' beards are thin and short.
- **Next** — widen m, beef up the u cup's thick side via `n_bias` per glyph,
  longer s beards; then caps colour (E arms vs H bar).

## 06 — u, m, beards (and a kit-wide spacing fix)

Changes: `u` gets its own bias (`u_bias` 0.3) so the cup carries weight on
the left wall like the n's arch does on the right; m arches 330 → 360;
longer s/S beards. Kit fix found by the benchmark: side detection used to
treat serif tips as a glyph's sides, so serifed straights got *round*
sidebearings; now only the skeleton and contour parts decide.

Critique (sheet `06.png`):
- **Colour** — u and n now match; m no longer reads condensed. The word
  "hamburgefonstiv" at 120 px has even colour except the heavy `g` loop.
- **Spacing** — `m`→`b` is a touch tight after the side fix (both straight
  now); `r`→`g` still open. Next: per-pair kerning is M5, but `r` wants a
  smaller right sidebearing (per-glyph `sb`).
- **Serifs** — consistent hairline serifs across h, m, n, u, r, i, H, E, A, v.
- **Terminals** — balls (a, r, f), beards (s, S, G), the e's tapered
  terminal: a coherent Didone set, distinct from the benchmark's wedges.
- **Lumpy curves** — the s/S spine is now smooth; G's upper terminal is a
  little blunt where the aperture cut meets the thick wall.
- **Verdict** — credible display type at 72–120 px for this word set; not yet
  release quality (spacing, g loop weight, G terminal).

## 07–08 — on the join primitive

The kit's arches (n h m r u) and the a's bowl are now built from the `join`
primitive and the new `lobe` part, so the original font picked those up
too: arches use `n_join_at/thin/angle`; the a's bowl hangs on its stem at a
hairline top join and a heavier bottom join. Bug fixes from the benchmark
(bounded terminal cuts) also apply here.

Critique (sheet `08.png`):
- **a** — the lobe gives it a proper Didone a: hairline top join into the
  stem, weight on the left wall, a narrow counter. Better than any earlier
  iteration.
- **n h m r u** — unchanged in appearance (the join reproduces the old
  construction's crotch), which is what a refactor should do.
- **Remaining** — spacing (`r`→`g`), g's loop weight, G's blunt upper
  terminal. Not release quality yet; credible display type at 72–120 px.

## 09 — the turning pen

The language can now turn the nib along a stroke (a node's `pen_angle`),
so the display serif uses it as a design tool rather than a fit. Three nib
programs were compared on `hamburgefonstiv` (fixed nib / "tightening" /
"calligraphic", the last two mirror images):

- *Calligraphic* (the nib turns toward the stroke at the tops of curves, away
  in the descents): early, heavy shoulders and a lumpy, twisting s. Lively,
  but it fights the hairline serifs and ball terminals.
- *Tightening* (the opposite): the arch leaves the stem as a true hairline
  across the shoulder and swells late into the descent — the Bodoni arch.
  The rest of the face (vertical stress, hairline serifs, balls) already
  speaks that language. **Chosen, at 1.5× the first trial's strength.**

Values (degrees, added to the pen's angle at that node): arch and cup
shoulders −30, legs +15; s arcs (top/bottom) −30, sides +15; f hook top −30,
end +15; r ear end −30; t tail bottom −30, end −15; a hood top −30, terminal
+15. Stems and bowls are untouched (the stems stay exactly `stem` wide —
the normalisation is per sample).

Critique (sheet `09.png`, vs `08.png`):
- **n h m** — the clearest gain: hairline shoulders, the weight arriving in
  the descent; the word at 120 px reads crisper and more Didone.
- **u** — the cup now thins toward its bottom before rising into the right
  stem, matching the arch's logic inverted.
- **r f a** — ear, hook and hood tops finer before their balls; the balls
  read as the heaviest point of each terminal, which is right.
- **s S** — the spine keeps its weight while the arcs thin; but the S's
  outer edge still wobbles (it did in 08 too: the spine's width profile,
  not the nib) — next to fix.
- **Remaining** — the S wobble, the b's stem/bowl junction (reads as "lb"
  at 120 px), r→g spacing, g's loop weight. Not release quality yet.
