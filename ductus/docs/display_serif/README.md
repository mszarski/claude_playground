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
