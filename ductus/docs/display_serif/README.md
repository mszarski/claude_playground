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
