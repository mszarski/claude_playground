# datagen: procedural engineering parts as CadQuery programs

Goal: millions of *realistic* single-part CAD programs that use the features real parts have (revolves, holes,
counterbores/countersinks, fillets, chamfers, shells, pockets, slots, patterns, ribs) and that our current training
data (DeepCAD + CAD-Recode: sketch + extrude only) never contains. The programs are training targets for a model that
reads renders of a part and writes the program, so the *code itself* is what the model learns to write.

## A family module
`datagen/families/<name>.py` defines `generate(rng: random.Random) -> str`, returning one program. Reference:
`flange.py`. Helpers in `common.py`: `f(x)` number formatting, `orient(rng)` optional final rotation, `on_circle(...)`.

## Rules for the emitted code
1. **Literal CadQuery only.** First line `import cadquery as cq`, then assignments; the last assigns `r`. No loops,
   functions, comprehensions, imports other than cadquery, math, randomness: the generator computes every number
   and writes it out. Numbers through `f()` (integers or one decimal). Keep it as short as the part allows: one
   method chain is ideal; a few named intermediate workplanes/solids (`w0=...`, `b=...`) are fine.
2. **Build in the XY frame**, so face selectors such as `faces('>Z')`, `faces('<X')` mean what they say. Orient the
   finished part with `orient(rng)` (one final `.rotate(...)` about a coordinate axis), never by building on YZ/XZ.
3. **Edge selection for fillets/chamfers: `near((x,y,z), ...)`** (defined in `mm/cad_helpers.py`, available in every
   runtime): picks, for each point, the edge truly closest to it. Give a point *on* the intended edge (for circles use
   `on_circle(r, z, cx, cy)`, which avoids the cylinder seam). Simple string selectors (`edges('|Z')`,
   `edges('>Z')`, `faces('>Z')`) are also fine when they select exactly what you mean.
4. **Canonical frame:** centred at the origin (bounding-box centre within ~10 of 0), longest bounding-box side
   between ~100 and 200 (use the whole range).
5. **One valid solid.** `datagen/check.py` enforces: literal code, builds, exactly one solid, BRepCheck valid, size and
   centre limits, not a sliver, deterministic rebuild.
6. **Realistic and varied.** Proportions engineers would draw; features optional with probabilities so a family
   yields many distinct topologies; feature sizes consistent with the part (holes clear of edges, fillet radii smaller
   than the faces they blend, wall thicknesses sensible). Use a mix of op styles (hole/cboreHole/cskHole,
   rarray/polarArray/pushPoints, cutBlind/cutThruAll, revolve, shell, slot2D, rect/circle/polygon sketches).
7. **Fast:** < ~1 s per part to build.

## Quality bar for a family (check before handing in)
- `python -m datagen.check <family> 200` from `gencad/`: pass_rate >= 0.9, sec_per_part < 1.
- `python -m datagen.preview <family> --n 24 --out <png>` and *look at it*: parts must read as plausible engineering
  parts, show the intended features, and vary. Fix anything degenerate (zero-width walls, fillets swallowing faces,
  holes breaking through edges, near-duplicates).
