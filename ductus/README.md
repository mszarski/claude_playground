# Ductus

A parametric font design language and compiler: glyphs are described as
named strokes and parts driven by shared parameters (`font.py`), hand
adjustments live as deltas in `overrides.yaml`, and the compiler emits a CFF
`.otf`.

![contact sheet](docs/contact_sheet.png)

*Milestone 1 status: compiler, `build`, `test` and proofs are done;
`ductus serve` (live preview) is next.*

## Usage

```bash
pip install -e ".[proof,validate,test]"

ductus build examples/test_font/font.py -o out/                 # static .otf (validated with fontTools + OTS)
ductus build examples/test_font/font.py --set stem=120           # one-off parameter values
ductus build examples/test_font/font.py --sweep stem=40,160      # + point-compatibility check
ductus test  examples/test_font/font.py                          # all lints, default sweep = every ranged param at both ends
ductus proof examples/test_font/font.py --setting "Bold:stem=150" -o proof.png
python docs/make_proofs.py                                       # regenerate docs/ proofs
pytest
```

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
