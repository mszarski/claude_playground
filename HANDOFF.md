# Handoff: Ductus (parametric font compiler)

These notes are for a fresh session picking up the work. All of the work
described here lives in `ductus/`. The repo root also holds an unrelated
`trajectory_classifier` package that this work never touched. When `ductus/`
moves to its own repo, take this file with it; paths below are relative to
`ductus/` unless they start with `/`.

Last state: branch `claude/quirky-darwin-eqmcsa`, 105 tests passing, working
tree clean.

---

## 1. Goal

**What it is.** Ductus is a Python-embedded language for describing
typefaces as strokes, parts and shared parameters. A compiler turns that
description into a real OpenType (CFF) font, and a live browser editor lets
you design with it.

The idea: generate the 90% from shared parameters, hand-own the 10% with
named-point overrides, and treat a font like software (it compiles, it has
lints and tests).

**Where the spec came from.** The user uploaded `SPEC.md` twice (rev 1, then
rev 2). **It is not in the repo**; it only existed as an upload to this
session. If the new repo should have it, ask the user to add it. Its key
points:

- **M1, foundations and editor.** Glyphs `n h m o H O space`; build a valid
  .otf; point compatibility across a `stem` 40–160 sweep; overrides stored as
  deltas on semantic names; the `ductus serve` editor; `ductus test`.
- **M2, representability.** Make the DSL expressive enough to represent a
  high-contrast display serif in the genre of Geist Serif (Vercel, OFL).
  - Metrics: UPM 1000, cap height 710, x-height 530, ascender 750,
    descender −200.
  - A benchmark harness: IoU, Hausdorff and override count per glyph.
  - Editor additions: onion-skin overlay, error heat map, stage inspector,
    grouped overrides, primitive handles.
  - An original display serif in `examples/display_serif/`.
- **Later milestones** (not started as such): M3 variable fonts plus a full
  character set; M4 size-aware compilation (opsz, hairline rasterisation
  tests); M5 kerning and spacing from shape; M6 rule-based derived styles
  (italics); M7 browser-only mode via Pyodide.
- **Licensing rule (SPEC §8).** Geist is used **only for measurement and
  overlays**. Never copy, trace or import its outlines. Anything fitted
  closely to it is a derivative work under the OFL, so it must stay an
  internal benchmark under `benchmarks/` and never be released.

**What success looked like.**
- M1 done, M2 done: every benchmark glyph at IoU ≥ 0.93, Hausdorff ≤ 20 u,
  ≤ 2 overrides.
- Then, at the user's direction: a full Latin-1 display serif, pulled
  visually close to Geist Serif.
- Then a second style from the same compiler and kit (Ductus Sans, matched to
  Geist Sans).
- The user judges success **by eye on set words**, not by IoU alone.

## 2. Current state

### What works
- **Compiler pipeline.** Params → parts/glyphs → Hobby skeletons → pen
  expansion (turning pen, width profiles) → junctions and terminals →
  skia-pathops merge (union minus cuts) → spacing → OTF with GPOS `kern`,
  GSUB `liga` and a legacy `kern` table.
- **CLI** (`ductus.cli`): `build`, `test`, `serve`, `kern`, `bench`, `fit`,
  `proof`.
- **Editor** (`ductus serve`): every M1 and M2 editor feature, including
  primitive handles and the reference/heat views.
- **Optical auto-kerner** (`ductus/kerning.py`, `ductus kern`), with
  per-glyph style weights.
- **Kit** (`ductus/kits/serif.py`, about 2,300 lines, one file): every construction
  plus three terminal sets, `bracketed_set`, `didone_set` and `sans_set`.
  Style switches on `kit.define(...)`:
  - `g_style`, `l_style`, `J_style`, `R_leg`, `Q_style`, `seven_style`
  - `g_ear`, `g_band`, `b_style`, `E_serifs`, `G_jaw`
- **Composites and combines**: `font.composite` for accented letters,
  `font.combine` for æ and œ, `font.slant`.

### Fonts

| Font | File | Glyphs | Status |
|---|---|---|---|
| Benchmark | `benchmarks/geist_serif/font.py` + `fitted.yaml` | 16 (`hamburgefonstiv` + `H`) plus the `f_i` ligature | Latest row (iteration 15, `docs/representability.md`): every glyph 0.96–0.995 IoU, 0 overrides. The README quotes mean 0.977 |
| Ductus Display (serif) | `examples/display_serif/font.py` | 181 | Iterations 01–32 in `docs/display_serif/README.md`. Letters ≈ 0.944 mean IoU vs Geist Serif; accented ≈ 0.94; symbols ≈ 0.90; digits ≈ 0.89. Kerning: 5,187 pairs, style-fitted (letter-pair correlation with Geist 0.77, letter–punctuation 0.75) |
| Ductus Sans | `examples/sans/font.py` | 181 (175 scored) | Iterations 01–02 in `docs/sans/README.md`. Mean IoU 0.916 (letters ≈ 0.94). Kerning: 9,094 pairs, clearance 60, min-abs 5. Correlation with Geist Sans: all pairs 0.71, letter pairs 0.68 |
| Test font | `examples/test_font/font.py` | 7 | The M1 font |

### Ductus Sans weakest glyphs (IoU)

| Glyph | IoU |
|---|---|
| standalone cedilla | 0.46 (ç 0.87, Ç 0.89) |
| ~ | 0.74 |
| ^ | 0.77 |
| @ | 0.80 |
| g | 0.81 |
| { } | 0.81 |
| tilde mark | 0.82 |
| ß | 0.83 |
| = | 0.83 |
| & | 0.83 |
| £ | 0.84 |
| idieresis | 0.85 |
| 9 | 0.85 |
| period | 0.85 |
| # | 0.85 |
| ellipsis | 0.85 |
| a | 0.86 |
| J | 0.86 |

Overlays show ~ ^ @ { } already close by eye.

### Ductus Sans kerning against Geist Sans

| Pair | Ours | Geist |
|---|---|---|
| AV | −105 | −106 |
| P. | −110 | −110 |
| To | −70 | −80 |
| TA | −65 | −90 |
| LT | −95 | −140 |
| rn | 0 | −40 |
| Fl | 0 | −20 |

### Half-finished or not started
- **SPEC M3–M7 have not been started**: no variable-font build, no opsz, no
  italics.
- **Display serif kerning outliers.** “A is −30 against −140, LT −50 against
  −140, AV −160 against −110.
- **Sans glyph and kerning gaps**: the weakest glyphs above, and pairs the
  profile kerner can't see (rn, Fl).

## 3. Decisions and reasons
- **Open questions in SPEC §13** are answered in `README.md` ("Decisions on
  the open questions"):
  - named-extrema bowls, not superellipse formulas;
  - Hobby skeletons, not Spiro (tension is more direct to expose);
  - width profiles as small knot curves with smoothstep between knots;
  - decorators, not a builder;
  - naming `glyph.instance[.stroke].node`;
  - overrides are not inherited.
- **Benchmark target: IoU ≥ 0.93, Hausdorff ≤ 20 u, ≤ 2 overrides.** Set
  after `o`/`n`. IoU is exact on outlines (pathops), aligned at the best
  offset, with guards because pathops once returned IoU 1.007.
- **Prefer extending the language over overrides** (SPEC principle 7). The
  benchmark reached 0 overrides that way: the `h` overrides were later
  replaced by parameters.
- **Serifs live in the terminal set, not in the constructions.** That is why
  a sans needed only `sans_set`, monoline values and a few structural style
  switches, not a new kit. Making a "sans kit" was rejected.
- **The turning pen.** The pen angle varies along strokes (node `pen_angle`,
  the `dp` override), clamped to ±80°. Profiles are mapped by arc length with
  smoothstep-eased node deltas, which removed kinks at nodes.
- **Open strokes are oriented by construction** (built clockwise), not by
  signed area, because area-based orientation flipped `r` at the all-max
  sweep corner.
- **Kerning is optical and computed from our outlines** (`ductus/kerning.py`):
  - The distance is a power mean of side-profile gaps (p = −32, depth 500),
    compared against a neutral flat edge `|` (edge 40).
  - Serifs are discounted with a morphological opening (height 80, reach
    100).
  - 2-D clearance only limits tightening; it never pushes apart.
  - The *house style* is per-glyph weights `[as left, as right]`, fitted to
    the reference's kerning by alternating least squares regularised toward
    1 (`kern_style.json`).
  - Rejected: copying the reference's kern values (licensing, and the shapes
    differ), a neutral "H" partner (its serifs polluted the comparison), and
    1-D clearance.
- **Proofs read GPOS kerning themselves** (`proof.gpos_kerning`), because
  FreeType ignores the legacy `kern` table in CFF fonts. All proofs before
  that fix were unkerned.
- **The user decides by eye.** Display changes are judged on whole words at
  80–120 px against the previous version, with before/after images, not on
  zoomed single letters.
- **Display serif direction.** At first it was meant to be an *original*
  design (SPEC). The user later asked to pull it "much closer" to Geist Serif
  ("it really seems like there's a large perceptual difference"), and the
  values were fitted toward Geist from iteration 18 onward. **See the open
  question in §7.**
- **Ductus Sans.** The user picked "Sans-serif (grotesque)" and "Match a
  reference" (Geist Sans).

## 4. Dead ends (don't repeat)
- **"Sliver traps."** Changes that raise IoU through thin hairline extensions
  but look wrong:
  - the A's toe;
  - v top serifs;
  - the Œ's bowl hairline showing inside the E.
  These are now constrained: `A_foot_dl` −20, `v top_serif_len`,
  `OE_dw ≤ 0`. Always look at the overlay, not just the number.
- **Judging from zoomed letters made the display serif worse.** Around
  iteration 10 the user said "that looks worse than a few turns ago". The
  fix was to judge on words and keep a history sheet
  (`docs/display_serif/history.png`).
- **`g_full` 0.57** made the g overlap itself. Reverted.
- **Inner-bracket experiments and S-angle experiments** on the serif didn't
  help. Reverted (one revert broke the parens; fixed).
- **Wrong arch leg-end formula** (side convention reversed). Reverted; added
  `n_leg_top` instead.
- **Spiking pen widths when the pen angle neared 90°.** Hence the ±80°
  clamp.
- **Œ trim cut removed the E arms.** The bowl's right wall is now the stem,
  with no cut.
- **Blindly merging fit results**, like the polish fits for several glyphs
  at once, made glyphs worse. `ductus fit --only-improve` now saves only if
  every glyph improves and no on-target glyph falls off.
- **Sans `7` face wedge.** The kit's wedge serif under the bar caused an
  overlap-lint failure with `seven_style="curve"` and doesn't belong on a
  sans. It is skipped for the curved style. Moving the stroke's start point
  did not fix it.

## 5. Gotchas
- **FreeType + CFF ignore the legacy `kern` table.** Use
  `proof.gpos_kerning` / `render_text`. `gpos_kerning` is cached by
  path + mtime.
- **`proof._shape` applies every GSUB ligature lookup (type 4), whatever the
  feature.** That is probably why Geist Sans rendered `{g}` with
  corner-bracket glyphs in a comparison. Treat odd reference renders with
  suspicion.
- **`pyproject.toml` lists `packages = ["ductus", "ductus.serve",
  "ductus.bench"]` and omits `ductus.kits`.** Editable installs work; a
  regular install would miss the kit. It should be fixed.
- **Tiny-contour filters.** Pinhole contours under 40 u² are dropped after
  merge (and under 4 u² earlier). A legitimately tiny mark could vanish.
- **The stems lint is skipped for slanted glyphs** (`font.slants`), and for
  profiled strokes and strokes with node `dw`.
- **Composites inherit their base's kerning weights; combines (æ œ) take
  their outer parts'.** Marks are drawn at the lowercase position and lifted
  by cap height minus x-height on capitals.
- **Spacing table.** `font.spacing.update({...})` holds per-glyph `(l, r)`
  deltas. Empty glyphs (space) are widened by it too. `respace.py` once
  keyed Æ by its character instead of `AE`; fixed with
  `c.isascii() and c.isalpha()`.
- **`N_dy`** applies to the right stem and the clip; the sans needs
  `N_dy=12` or the N's tick drops below the baseline.
- **The sans inherited `at_slant 14`** from defaults; it is set to 0. Check
  inherited serif defaults when adding sans glyphs.
- **The cedilla needs `ced_w 55`** in the sans, or lint fails on its
  direction and overlaps.
- **Fitting scripts** (`tools/fitting/`) have hard-coded absolute paths and
  write thousands of temp fonts into `sw/`. `exp.py`'s words image compares
  against Geist **Serif**, even for the sans.
- **Process handling:**
  - Don't `pkill -f`; it kills its own shell.
  - `pgrep -f` waiters match themselves; use
    `ps -eo args | grep "^/usr/local/bin/python3.13 /usr/local/bin/ductus fit"`.
  - Shell chains can hide a lint FAIL; check exit codes.
  - Don't `git stash` while fits run.
- **This environment's stop hook demands a clean, pushed tree** at the end
  of each turn. Commit after background jobs finish, not mid-run.
- **The repo-level CLAUDE.md describes `trajectory_classifier`**, not Ductus.
  Its `bd` issue-tracker instructions were never used for this work.

## 6. The user's preferences
- **Show, don't just tell.** Send sheets and overlays as images
  (SendUserFile) after each step. They often ask "show me the sheet".
- **They judge visually.** Words at about 80–120 px, compared with the
  previous version and with the reference. Numbers support the judgment but
  don't replace it. When asked to "fix X", they mean it should *look* right.
- **Keep going without check-ins** on clear iterative asks ("keep going
  until you fix them all"). When a request is open-ended (e.g. "try a
  different style font"), offer choices first; they picked from options for
  the sans.
- **Mid-turn questions** (e.g. "what branch are we on?") get answered
  inline, then the work continues.
- **Corrections they gave:**
  - "somehow that looks worse than a few turns ago?": keep a history and
    judge on words.
  - "Joins all look thin, g is quite different up top": they want real
    perceptual closeness, not IoU gains.
  - Earlier, a figure was misstated (0.975 for 0.976) and a commit was
    pushed with a failing test. Report numbers exactly and push only when
    green.
- **Licensing.** Geist renders and data stay out of the repo (scratch only).
  A comparison image containing Geist Sans renders was committed by mistake
  in `179a97a` and removed in `3ac61e9`; it is still in history.
- **Git.**
  - Work on `claude/quirky-darwin-eqmcsa`. No PRs unless asked.
  - Every commit ends with the session's `Co-Authored-By` and
    `Claude-Session` trailers, and model names appear nowhere else.
  - Don't edit CLAUDE.md (it is updated separately).
- **Docs.** Each iteration gets a numbered sheet `docs/<font>/NN.png` and a
  README section with a short self-critique and before/after numbers.

## 7. Next steps (priority order)
1. **Open question for the user: licensing status of the two "matched"
   fonts.** SPEC §8 says anything fitted closely to Geist is an internal
   benchmark that must never be released, and that `examples/` holds
   *original* designs. Yet:
   - Ductus Display was fitted toward Geist Serif from iteration 18 on;
   - Ductus Sans was fitted to Geist Sans, including side bearings copied
     toward Geist's (`respace.py`) and kerning style fitted to Geist's.
   Ask whether to move both under `benchmarks/` (or mark them
   internal-only), or to diverge them deliberately before any release.
2. **Fix `pyproject.toml`** to include `ductus.kits`, or use package
   discovery.
3. **Ductus Sans glyphs:** ~ ^ @ g { } & £ ß = and the cedilla mark. Use
   `tools/fitting/cd2.py` per group, check overlays (`ov.sh`), lint
   (`lintvar.py`, then `pytest tests`), then respace twice.
4. **Sans kerning:** close the TA and LT gaps and pairs the profile kerner
   misses (rn, Fl, Ke). Perhaps a per-pair or class exceptions file next to
   `kern_style.json`, or a deeper `depth` for the arm of L.
5. **Display serif kerning outliers:** “A, LT, AV.
6. **SPEC M3, variable fonts.** Point compatibility is already enforced by
   construction and tested at range extremes, so a weight axis via
   `fontTools.varLib` from param ranges is the natural next milestone.
   Ask the user before starting a new milestone.
7. **Make `tools/fitting/` path-independent** (repo-relative paths, a
   `--font` argument, a `--reference` argument) if fitting continues.

**Other open questions:**
- Should `SPEC.md` be added to the new repo?
- Is the next style (italics via M6 rules, a text cut via M4, or a bolder
  weight) the user's priority over polishing the sans?

## 8. Pointers

### Commands (from `ductus/`)
```bash
pip install -e ".[proof,validate,serve,bench,test]"
python -m pytest tests -q            # 105 tests, ~50 s
python benchmarks/geist_serif/fetch_reference.py   # Geist Serif -> benchmarks/geist_serif/reference/ (git-ignored)
python benchmarks/geist_sans/fetch_reference.py    # Geist Sans 1.9.0 (npm `geist`) -> benchmarks/geist_sans/reference/Geist-Regular.woff2
ductus serve examples/display_serif/font.py --reference benchmarks/geist_serif/reference/GeistSerif-Regular.woff2
ductus bench benchmarks/geist_serif/font.py --glyphs o --note "what changed"
ductus fit   benchmarks/geist_serif/font.py --glyph o --only-improve
ductus kern  examples/display_serif/font.py                                # clearance 90, min-abs 10
ductus kern  examples/sans/font.py --clearance 60 --min-abs 5
python examples/display_serif/make_sheet.py 33   # next display sheet -> docs/display_serif/33.png
python examples/sans/make_sheet.py 03            # next sans sheet -> docs/sans/03.png
```

### Key files
- **Pipeline:** `ductus/compile.py`, `ductus/pen.py`, `ductus/skeleton.py`,
  `ductus/parts.py`, `ductus/contours.py`, `ductus/junctions.py`,
  `ductus/terminals.py`, `ductus/emit.py`, `ductus/lint.py`.
- **Kerning and proofs:** `ductus/kerning.py`, `ductus/proof.py`.
- **Benchmark:** `ductus/bench/` (`metrics.py`, `run.py`, `reference.py`,
  `report.py`).
- **Kit:** `ductus/kits/serif.py`, with `DEFAULTS` as
  `name: (default, (lo, hi))`.
- **Fonts:** `examples/display_serif/` and `examples/sans/`, each with
  `font.py`, `kerning.json`, `kern_style.json` and `make_sheet.py`.
- **Benchmark font:** `benchmarks/geist_serif/` (`font.py`, `fitted.yaml`,
  `overrides.yaml`).
- **Docs:**
  - `README.md` (usage, primitives, decisions)
  - `docs/representability.md` (benchmark table, iterations 01–15)
  - `docs/display_serif/README.md` (iterations 01–32) and `NN.png`
  - `docs/sans/README.md` (01–02)
  - `docs/iterations/NN/` (benchmark overlays)
- **Fitting scripts:** `tools/fitting/` (see its README).
- **Tests:** `tests/`. Fonts are parametrised in `test_contours.py`,
  `test_junctions.py` and `test_primitives.py`; `test_ends.py` covers
  punctuation, composites, slant, combine, autokern and GPOS in proofs.

### Commits (branch `claude/quirky-darwin-eqmcsa`)
| Commit | What |
|---|---|
| `cdfc98a` | Display serif iteration 31 (kerning in Geist's style) |
| `5fcd3af` | Ductus Sans 01 |
| `21135a4` | Sans weakest-glyph fixes |
| `179a97a` | Sans kerning style and sheet 02 |
| `a63c122` | `tools/fitting` saved |
| `3ac61e9` | Geist render removed from docs |

### External resources
- **Geist fonts:** npm package `geist`, pinned to 1.9.0 in
  `benchmarks/geist_sans/fetch_reference.py`; https://registry.npmjs.org/geist.
  SIL OFL 1.1. Measurement only.
- **This session's transcript**, only while this container exists:
  `/root/.claude/projects/-home-user-claude-playground/9fa36f9b-b558-5a88-94f2-47889dc83674.jsonl`.
- **The scratchpad**
  (`/tmp/claude-0/-home-user-claude-playground/9fa36f9b-b558-5a88-94f2-47889dc83674/scratchpad`)
  holds the Geist .otf conversions, comparisons, overlays and the serif
  fitting scripts (`near/`). It disappears with the container.
