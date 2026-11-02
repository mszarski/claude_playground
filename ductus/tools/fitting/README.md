# Fitting scripts (as used, not polished)

These are the working scripts that fitted Ductus Sans (`examples/sans/font.py`)
to Geist Sans, copied from the session scratchpad so they are not lost. The
display serif was fitted with near-identical copies pointed at
`examples/display_serif/font.py` and Geist Serif.

**They have hard-coded absolute paths.** Before using them, edit the paths at
the top of each file:
- the repo: `/home/user/claude_playground/ductus`
- the scratch directory: `/tmp/claude-0/.../scratchpad` (shown as `N`, `SP` or `sys.path[0]`)

They expect these in the scratch directory:
- `sw/`, an empty directory for temporary font variants. `cd2.py` uses
  `sw/` beside itself; `lintvar.py` uses `<scratch>/sans/sw/`. Both write
  one file per evaluation and never clean up; it reached about 8,700 files.
  Keep `sw/` out of git.
- `sans/geist_sans.otf`: Geist Sans converted from the fetched woff2:
  ```python
  from fontTools.ttLib import TTFont
  f = TTFont("benchmarks/geist_sans/reference/Geist-Regular.woff2")
  f.flavor = None
  f.save("geist_sans.otf")
  ```
- `vs/geist.otf`: Geist Serif converted the same way. Used by `exp.py`; see below.

The reference fonts are for measurement and overlays only (SPEC §8). Never
commit them, renders of them, or data extracted from them, such as
`geist_kern.json`.

| Script | What it does |
|---|---|
| `cd2.py GLYPHS p:step ...` | Coordinate descent on the font file's `kit.params(...)` values, scoring mean IoU over `GLYPHS`. `GLYPHS` is either one string of single-letter glyph names (`"eS"`) or comma-separated names (`seven,` / `at,`). Prints `end <iou> {changes}`; it does not write the font. |
| `apply.py "{'k': v}"` | Writes those values into the font file: replaces `k=` if present, else inserts the parameter beside others with the same prefix. |
| `tidy.py` | Re-wraps runs of one-parameter lines in the font file. |
| `exp.py font.py TAG` | Per-glyph IoU plus `MEAN IoU`, and writes `TAG.otf` and `TAG.png` (words image). **Bug:** the words image compares against `vs/geist.otf`, which is Geist *Serif*, even for the sans. |
| `respace.py built.otf` | Moves side bearings in `font.spacing.update({...})` toward the reference's. Run `exp.py` → `respace.py` until it reports no changes (usually two passes). |
| `ov.sh TAG glyphs` | Overlay grid (ours vs reference) → `ov_TAG.png`. Needs a `TAG.py` copy of the font in the scratch directory. |
| `lintvar.py glyphs "k=v,..."` | Lint errors and IoU for parameter variants, without editing the font. |
| `kcal.py` | Reads Geist Sans's GPOS kerning for a 68-character set → `sans/geist_kern.json` (scratch only). |
| `kstyle.py [lambda]` | Fits per-glyph kerning style weights (alternating least squares, regularised toward 1) → `sans/kern_weights.json`; copy it to `examples/sans/kern_style.json`. |
| `kagree.py kerning.json ...` | Correlation and mean absolute error of a `kerning.json` against Geist's kerning. |
| `cmpw.py "line§line" out.png SIZE` | Words rendered in ours and in Geist Sans, alternating rows. |
| `weak.sh` | Example batch: a `cd2` + `apply` run per glyph group, then `tidy` and `exp`. |
