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

## 10 — the S wobble, the b, the g loop, r spacing

All four remaining issues from 09 fixed (sheet `10.png`):

- **S / s wobble** — two causes, one in the engine, one in the design.
  *Engine:* a stroke's width profile was placed along the stroke by bezier
  parameter, which doesn't move at constant speed, and node deltas (width,
  pen angle) were blended linearly per segment — both made the width's slope
  jump at every node, a kink in the outline. Profiles now follow arc
  length, and node deltas ease between nodes (zero slope at each node).
  *Design:* the middle of the spine was told to run at −48°, steeper than
  the curve wanted, so both halves curled to meet it — a wiggle that the
  high contrast turned into bumps. Now −24°, with the swell (1.5) kept to
  the straight part of the spine (spread 0.2): a heavier spine that stays
  smooth where it bends.
- **b** — read as "lb": a separate bowl laid its thick wall beside the
  stem. The b is now built like the a: the stem *is* the bowl's left side,
  with hairline joins at the crown and the foot (the kit's lobe), a bowl as
  wide and full as the o's.
- **n h m u** — a small step showed where a leg stem's flat end stood
  beside the arch (or the u's cup) before it had turned vertical; the stems
  now stop just past where the stroke is vertical (`n_leg_top`,
  `u_leg_top` 0.55).
- **g** — the lower loop lighter (walls 0.75 → 0.6 / 0.9 → 0.72, bottom
  20 → 14): it no longer outweighs the upper bowl.
- **r** — right sidebearing −20 (`r_rsb`): its ball overhangs; r→g and
  r→a close up, r→n stays clear.

Critique: at 120 px "hamburgefonstiv" now reads as one even Didone word —
no "lb", no dark g, smooth s. Still to do for release: kerning (M5), the
G's upper terminal (blunt where the aperture cut meets the thick wall),
and the beards on s/S are a little ragged at the cut.

## 11 — fixing what looked weird (word level, not zoomed letters)

After 09–10 the face looked worse at word size than its letters did zoomed in
(see `history.png`: the same text re-rendered from every commit). This round
was judged only on the set word at 120 px against the previous iteration
(`11_vs_10.png`). What read as "weird", and the fixes:

- **s / S** — arms ending in chunky triangular beaks, thick right to the end,
  and a kink mid-spine. Now: hairline arms rising into **ball terminals**
  (the face's terminals are balls everywhere else), and the spine's middle
  direction is **solved** instead of imposed. New in the engine: a path can
  ask for its free node directions to be solved for continuous curvature
  (`path(..., smooth=True)`, Hobby's/METAFONT's mock-curvature equations);
  an imposed direction made both halves curl to meet it, and with this much
  contrast that dip in direction became a kink in the width.
- **arches (n h m, u)** — the hairline left the stem low on a straight 50°
  diagonal, which read italic. It now leaves high (`n_join_at` 0.84, angle
  30) and curves away; the nib turns are gentler (−10 / +5).
- **t** — the tail was a short downward stub. Now it curves into a rising
  hairline tail with a ball (`t_end_dir` is a parameter now); a longer tail
  was tried first and its ball read as a full stop ("st.iv").
- **e** — the lower terminal ended in a slanted spike; the aperture cut now
  leaves it level (180°), a fine flat end.

Still to do: spacing is loose overall (M5 kerning and a tighter default),
the G's upper terminal, and a second look at the u's bottom join and the
a's bowl at large sizes.

## 12 — heavier hairlines and serifs, tighter spacing; G, t, s again

Comparison: `12_vs_11.png` (set text, 96 px, unscaled).

- **Weight** — hairline 14 → 22, serif tips 14 → 26, brackets 12/14 → 18/20,
  serifs a little longer (62 → 74, caps 72 → 84, heads 62 → 70). At 14 units
  the serifs vanished whenever a proof was scaled down, and the face read as
  a sans with odd swellings. (Measured at 96 px they are now 3 px thick.)
- **Spacing** — straight / round sidebearings 46/34 → 36/26.
- **s** — its spine was the thinnest stroke: the solved spine runs at −19°,
  nearly horizontal, which a vertical-stress pen draws as a hairline. The
  swell now carries it (2.2 over a narrower spread): the spine is the s's
  heaviest stroke, as a Didone s's should be.
- **t** — the ball on the tail read as a stuck-on hook (and, further out, as
  a full stop). The tail is now long and low and ends in a plain hairline,
  the nib turned a little at the bottom so the curve keeps weight.
- **G** — the arm ended in a staircase (a 26-wide beard hanging from a
  110-wide arm), and the jaw was a full-weight stem plus a long bar: a dark
  block. New G controls: beard width/length (`G_beard_dw`, `G_beard_k`) and
  jaw weight (`G_spur_w`). Now the arm ends flat and the jaw is lighter
  (0.75 × cap stem) with a short bar.
- Proofs render unhinted now (the fonts have no hints; FreeType's autohinter
  was snapping the hairline serifs).

## 13 — t bottom, G jaw

Comparison: `13_vs_12.png` (set text, 110 px).

- **t** — the tail rose at 25° and ended in a square cut at 0.3 of the
  width: with vertical stress the flat bottom is a hairline, the rising part
  thickens, and the end cut was thicker again — a wave. Now the tail stays
  nearly level (end direction 10°, ending at y 35) and thins to 0.2, below
  the bottom's hairline: one clean sweep out of the stem, the weight carried
  round the outside of the curve.
- **G** — the jaw was a full-weight stem standing beside the bowl, a dark
  block. Tried and rejected: stopping the jaw inside the bowl (it floated as
  a separate block, since the bowl curves in below the bar), drawing it as
  a pen stroke along the bowl (left slivers of the bowl inside it), and
  letting the bowl's wall be the jaw (it read as a C). Kept: a slim jaw
  (0.55 × cap stem) to the baseline with its small spur, short bar, flat arm.
  The kit gained `G_spur_bot`, `G_jaw_in` and a `G_spur` terminal tag for
  these experiments (defaults keep the old G).

## 14 — spacing, kerning, the u and a joins

Comparison: `14_vs_13.png` (set text, 96 px).

- **Engine** — fonts can now carry a spacing table (`font.spacing = {glyph:
  (left, right)}`, added to that glyph's sidebearings) and kerning
  (`font.kern(left, right, value)`), emitted as GPOS pair kerning plus a
  legacy `kern` table (which FreeType's proofs apply).
- **Spacing** — set between n's and o's at 72 px. An automatic area-based
  measure was tried as a cross-check and set aside: with these arches the
  result depended mostly on the depth clamp. Rounds were tight (o's nearly
  touching): round sidebearing 26 → 30. The f left a hole after it (hook and
  bar overhang): right side −60, then kerned back out where the ball would
  hit a dot, an ascender or another f (f+i 25, f+f 20, f+b / f+h 30).
  r+g −20. (fi still nearly touches at large sizes: an fi ligature is the
  real fix.)
- **u** — the cup's hairline hooked where it left the bottom: the cup's
  bottom height is set from the unturned pen, so the −10° nib turn there
  made the stroke thinner than planned and the hairline dipped. The u's cup
  now takes no nib turns.
- **a** — the bowl met the stem with a step at its bottom join (at 115, 48
  thick, 55°); now it tapers in low (60, 30, 25°).

## 15 — fi ligature

Comparison: `15_vs_14.png` (set text, 120 px).

The f's ball came within a hair of the i's dot at display sizes; kerning
can only trade that for a hole. Now an fi ligature: the f's hook reaches
over the i (by `fi_dx` 175, which keeps the i stem at its usual distance from
the f) and its ball *is* the i's dot; the f's bar runs across to the dotless
i stem, which stops at the bar's top.

Engine: `font.ligature(glyph, components)`, emitted as a GSUB `liga`
substitution; proofs apply the font's own GSUB ligatures (no HarfBuzz
needed). Kit: the f is built by one helper shared with `f_i`, which is
defined whenever f and i both are. `ductus bench` now skips glyphs the
reference lacks when no glyphs are named (the Geist benchmark has an f_i
too, but Geist Serif has no glyph to compare it with).

## 16 — the f's ball

Comparison: `16_vs_15.png`.

The f's hook didn't end in its ball: it arrived at full weight (end width
1.0, the nib turned +15° to thicken it) heading down-right at −60°, with the
ball beside it, so the flat-cut end stuck out past the ball as a wedge. And
the hook humped at the top: drawn with the arch's 25° nib it went thick
(up the stem), thin (up-right, along the nib), thicker (across the top).
Now the hook has a vertical nib like the rest of the face (`f_angle_d`, a new
kit offset from the arch's angle), no nib turns, and it thins (0.6) as it
turns down (−80°) into the ball — the r's ear works the same way. The fi
ligature's long hook gets the same clean hairline.

## 17 — the s, again

Comparison: `17_vs_16.png`.

Next to the o, the s was too dark with nearly closed counters: the spine,
solved for smooth curvature, ran at about −19° — nearly flat — and the 2.2
swell turned it into a heavy bar across the middle. The s's left and right
extremes now sit further apart vertically (0.78 / 0.22 of the height, were
0.73 / 0.25), which steepens the solved spine into a proper diagonal, and
the swell is 1.8. Its colour now matches the o, e and n; the S follows.

## 18 — toward Geist Serif

Sheet: `18.png`. Comparison: `18_vs_17.png`.

Asked to get "much closer" to Geist Serif. Next to it at the same size the
Didone read as a different face. Its letters were 80–300 u wider, the
contrast was much higher, the joins and hairlines thin, the serifs short
flat hairlines, and it had ball terminals where Geist has wedges and flags.
This iteration gives up the Didone. The face now uses the serif kit's
bracketed terminal set (wedge heads, bracketed feet, the g's flag ear and
flat band). Its values were set group by group (widths and spacing, weight
and contrast, joins, serifs, then each letter) from side-by-side overlays
with the bench's IoU against Geist as the guide, and rounded. A short
coordinate search placed the capitals, which the benchmark never fitted. No
outlines are copied: every glyph is still the kit's strokes, pens and
terminals. Mean IoU against Geist went from 0.556 to 0.912. The lowercase is
mostly 0.88–0.98; A, E, G and S are 0.77–0.85.

Kit additions:
- `E_serifs="wedge"`: the E's arm serifs as wedges sloping back `E_wedge`
  along the arm.
- `S_swell_k`: the S's spine weight relative to the s's.
- `G_jaw_dx`: the G's jaw can stand inside the bowl's edge.
- The G's spur has its own `G_spur_len/rise/drop`, so tuning the u's spur no
  longer moves it.

Side bearings come from the glyphs' measured side bearings, with a few
kerns (r against o, e and s; f against o).

Still visibly different: the G's jaw (Geist's stem stands free of the bowl
down to a point at the baseline), the S's ends (Geist's have vertical
beards), the g's link (Geist cuts a notch where it meets the band), and the
bottom of the e (heavier in Geist).
