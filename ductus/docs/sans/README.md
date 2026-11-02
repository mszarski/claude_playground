# Ductus Sans

A grotesque sans-serif built with the same compiler and the same kit
(`ductus/kits/serif.py`) as Ductus Display. It is matched to Geist Sans
(OFL-1.1, Vercel): values are tuned from overlays and the bench's IoU, then
rounded. No outlines are copied.

Source: `examples/sans/font.py`. Sheet: `docs/sans/NN.png`
(`python examples/sans/make_sheet.py NN`). Kerning: `ductus kern
examples/sans/font.py` writes `examples/sans/kerning.json`. Reference:
`python benchmarks/geist_sans/fetch_reference.py`; it is git-ignored and
used for measurement only.

## 01 — a sans from the serif kit

The kit's constructions are style-neutral: arches, a and b lobes, bowls,
the s spine, diagonals, the cap bowls, composites, spacing and auto-kerning.
The serifs live in the **terminal set**. So the sans is mostly three
changes:

1. **`kit.sans_set(P)`**, a new terminal set with no serifs. Stems end
   square; diagonals, apices and vertices are cut flat (horizontally).
   Curved strokes' ends are cut at `sans_cut_*` angles.
2. **Monoline values**, measured off Geist Sans: stem 84, horizontals and
   former hairlines about 76, round sides × 1.05, and the joins and lobes
   at near-stem weight. Several of the kit's ranges were widened to allow
   them.
3. **Grotesque options**, new structure choices in `kit.define`:
   - `g_style="single"`: the single-storey g (the d's lobe on a stem that
     runs into a hook).
   - `l_style="tail"`: the l turns into a tail, as the t does.
   - `J_style="base"`: the J turns along the baseline and back up.
   - `R_leg="curve"`: the leg bends down to stand upright.
   - `Q_style="bar"`: a straight bar across the bowl's lower right.
   - `dot_square`: rounded-square dots.
   - `colon_dy`: the colon's upper dot set lower.
   - `N_dy`: the N's diagonal and right stem stand on the baseline.

The values were then fitted by the same coordinate searches as the serif.
Each glyph's parameters were searched against Geist Sans: weights first,
then the lowercase, capitals, figures, punctuation, symbols, accents and
Æ Œ Ø ß. Side bearings come from Geist Sans's, and the word space is
250 u. Kerning comes from the optical kerner with its default settings
(6,590 pairs; no style weights yet).

**A kit bug found on the way.** The B's two bowls each kept only half a
hairline at the waist, so their counters met and the waist pinched to a
point at the stem. Each bowl now keeps a full hairline wall. That took the
sans B from 0.89 to 0.95 and the serif's from 0.93 to 0.95.

Against Geist Sans: all 175 glyphs 0.912, the 52 letters 0.937. Weakest:
- standalone cedilla 0.38 (ç and Ç are 0.89–0.90);
- £ 0.61, ~ 0.72, 7 0.76, ^ 0.77;
- @, ? and g 0.79–0.80;
- { } and & 0.81–0.82; 8 0.82.

The sans font is linted, compatibility-swept and checked at its range
extremes in the test suite, alongside the serif.
