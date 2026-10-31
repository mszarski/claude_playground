"""Ductus Display -- an original high-contrast display serif (M2).

Same genre as the benchmark (sharp, high-contrast, near-vertical stress),
but every value here is a design decision made by eye, not fitted to any
reference. Iteration notes live in docs/display_serif/.
"""
from ductus import Font, Pen, glyph, param, part

font = Font("Ductus Display", upm=1000)

# -- vertical metrics (SPEC M2 targets) -----------------------------------------
xheight   = param(530, range=(490, 570))
capheight = param(710, range=(670, 750))
ascender  = param(750, range=(710, 790))
overshoot = param(11,  range=(0, 25))

# -- weight and contrast ------------------------------------------------------------
stem      = param(94,   range=(60, 140))
contrast  = param(0.85, range=(0.6, 0.95))     # hairline = stem * (1 - contrast)
hairline  = stem * (1 - contrast)
round_gain = param(1.05, range=(0.95, 1.15))   # rounds a touch heavier than stems

font.pen = Pen(angle=0, width=stem, contrast=contrast)
font.xheight, font.capheight, font.ascender, font.descender = xheight, capheight, ascender, -200

# -- spacing (tight display spacing) -------------------------------------------------
font.sidebearings = {"straight": stem * 0.42, "round": stem * 0.36}

# -- rounds ------------------------------------------------------------------------------
stress      = param(6,    range=(-10, 20))     # degrees; slight oblique counter
outer_full  = param(0.60, range=(0.5, 0.8))   # outer: a little squarer than an ellipse
outer_full_v = param(0.56, range=(0.45, 0.8))
inner_full  = param(0.56, range=(0.45, 0.8))   # counter: round enough to keep hairlines crisp, not pinched
inner_full_v = param(0.62, range=(0.45, 0.8))  # ...with full sides for a strong vertical
o_width     = param(500, range=(400, 560))


@part
def bowl(p, width, height, stress=stress):
    p.bowl(0, width, -overshoot, height + overshoot,
           thick=stem * round_gain, thin=hairline,
           stress=stress,
           fullness=(outer_full, outer_full_v), inner_fullness=(inner_full, inner_full_v))


@glyph("o")
def o(g):
    g += bowl(width=o_width, height=xheight)


@glyph("O")
def O(g):
    g += bowl(width=capheight * 0.95, height=capheight)


@glyph("space", advance=stem * 2.3)
def space(g):
    pass
