# Ductus

A parametric font design language and compiler: glyphs are described as
named strokes and parts driven by shared parameters (`font.py`), hand
adjustments live as deltas in `overrides.yaml`, and the compiler emits a CFF
`.otf`.

![preview](docs/preview.png)

![contact sheet](docs/contact_sheet.png)

## Usage

```bash
pip install -e ".[proof,validate,serve,test]"

ductus serve examples/test_font/font.py                          # live editor at http://127.0.0.1:8765/

ductus build examples/test_font/font.py -o out/                 # static .otf (validated with fontTools + OTS)
ductus build examples/test_font/font.py --set stem=120           # one-off parameter values
ductus build examples/test_font/font.py --sweep stem=40,160      # + point-compatibility check
ductus test  examples/test_font/font.py                          # all lints, default sweep = every ranged param at both ends
ductus proof examples/test_font/font.py --setting "Bold:stem=150" -o proof.png
python docs/make_proofs.py                                       # regenerate docs/ proofs
pytest
```

## The preview editor (`ductus serve`)

- **Sliders** for every ranged parameter (number fields for the rest),
  per-parameter reset, presets saved in the browser, and *Copy* to paste the
  current values back into `font.py` as Python. Parameter values in the
  editor are session state; the editor never rewrites Python.
- **Canvas**: the glyph in context (`context` field, `#` = the glyph being
  edited), metrics, skeleton, fill/outline views. Click a skeleton point
  to get handles:
  - drag the point: `dx`/`dy`
  - drag the ◆ handle: rotate the stroke direction there (`da`, ⇧ snaps to 15°)
  - drag the □ handles: change the stroke width there (`dw`)
  - `,` / `.` change curve tension (`dt`)
  - turn on *Outline pts* to drag expanded outline points (`stage: outline`)
  - drag the dashed advance lines to change sidebearings (`<glyph>.lsb/.rsb`)
- **Inspector**: resolved position/direction/width/tension of the selected
  point, numeric fields for every delta, the glyph's overrides list and
  stale-override warnings with one-click removal.
- **Undo/redo** (⌘Z / ⇧⌘Z); a whole drag is one undo step.
- **Text** typed in the panel is set in the freshly compiled `.otf` via the
  browser's `FontFace` API (with a waterfall view and a download button).
- Edits go to `overrides.yaml`; `font.py` and `overrides.yaml` are watched,
  so editing either in your editor updates the preview. A broken `font.py`
  shows an error banner and keeps the last good result.

Timings on the example font: full compile ≈ 65–90 ms (parameter changes),
single-glyph compile ≈ 10 ms (dragging); .otf emit ≈ 6–10 ms.

## Overrides

```yaml
n.arch.shoulder:      { dx: 12, dy: 0 }                     # move a skeleton point
n.arch.join:          { dx: 0, dy: 0, dw: -18, da: -8 }     # thinner, steeper join
o.bowl.top:           { dx: 0, dy: 0, dt: 0.15 }            # flatter curve through the top
m.arch2.shoulder_out: { dx: -6, dy: 4, stage: outline }     # move both expanded points
o.lsb:                { dx: 6, dy: 0 }                      # sidebearing adjustment
```

`dw` is interpolated linearly along the segments next to the node; `da`
rotates the direction of travel; `dt` changes the Hobby tension on both
sides of the node. None of them change point structure, so they are safe
for interpolation.

## Pipeline

`params → skeletons (parts) → skeleton overrides → broad-nib expansion →
terminals → outline overrides → overlap removal (skia-pathops), direction,
rounding → spacing → .otf`. `compile_font(font, params, overrides)` is a pure
function returning JSON-serialisable `CompiledFont` (outlines, skeletons,
named points, metrics).

**Pen model.** `w(θ) = hair + k·soft|sin(θ − angle)|`, `hair = stem·(1−contrast)`,
with `k` chosen so vertical strokes are exactly `stem` (that's what the stem
lint measures). Each side of a stroke is offset along the normal and refitted
with a **fixed 4 cubics per skeleton curve** (1 line per skeleton line), so
point structure never depends on parameter values. Worst-case fitting error is
~2 units at stem 180 / contrast 0.9 (tested < 3).

**Stress (`Pen(inside=...)`).** For closed strokes, `inside` is the share of
the thick/thin modulation put on the counter side. 0.5 offsets symmetrically
(the outer contour of a bowl becomes egg-shaped with an oblique pen); higher
values keep the outer contour elliptical and put the stress into the counter.
The example font uses `stress = 0.72`.

## Decisions on the open questions

- **DSL ergonomics:** decorators (`@part`, `@glyph`) over builder objects.
  Parts are deferred calls, so `g.inherit("n"); g["stem"].top = ascender`
  edits the inherited part's arguments before it runs. Inherited glyphs are
  re-scoped (`h.arch.shoulder`); **overrides are not inherited**.
- **Naming:** `glyph.instance[.stroke].node`; a part drawing a single stroke
  collapses to `glyph.instance.node`. Outline points add `.l` / `.r`.
  Duplicate names are a compile error.
- **Skeleton overrides** move points freely in x/y (not constrained to the
  curve); curve handles are re-derived from node directions afterwards, so
  a moved node stays smooth.
- **Grid rounding:** only at the end (after overlap removal).
- **Points created by overlap removal** are unnamed and not overridable.
  Merged-outline topology can change across a sweep (the stroke/arch junction
  slides between fitted sub-segments); `test` reports this as info. The
  pre-merge stroke outlines are strictly compatible and are what a
  variable-font build (M2) will interpolate.
- **Spacing:** sides are `straight` if a vertical skeleton line sits at the
  extreme, else `round`; sidebearings come from `font.sidebearings`.
