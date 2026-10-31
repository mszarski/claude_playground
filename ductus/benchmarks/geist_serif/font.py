"""INTERNAL BENCHMARK ONLY -- never build or release this font.

A Ductus font whose parameters are tuned (by hand and by `ductus fit`) to
measure how well the language can represent Geist Serif. Fitted values are
derived from the reference and so this file must stay in benchmarks/
(SPEC section 8, licensing rule). Original designs live in examples/.
"""
from ductus import Font, Pen, glyph, param, part

font = Font("Ductus Bench Serif", upm=1000)

# -- vertical metrics (genre targets, SPEC M2) ------------------------------
xheight   = param(530, range=(480, 580))
capheight = param(710, range=(660, 760))
ascender  = param(750, range=(700, 800))
overshoot = param(12.26,  range=(0, 25))

# -- weight and contrast -----------------------------------------------------
stem      = param(100.5,  range=(60, 140))    # vertical stem
hairline  = param(33.5,  range=(8, 60))      # thinnest horizontal parts
round_gain = param(1.01, range=(0.85, 1.25))  # optical rule: rounds a bit heavier

font.pen = Pen(angle=0, width=stem, contrast=1 - hairline / stem)
font.xheight, font.capheight, font.ascender, font.descender = xheight, capheight, ascender, -200

# -- spacing -------------------------------------------------------------------
straight_sb = param(40, range=(10, 90))
round_sb    = param(48.7, range=(10, 90))
font.sidebearings = {"straight": straight_sb, "round": round_sb}

# -- o -------------------------------------------------------------------------
o_width          = param(507.3,  range=(380, 620))
o_full_h         = param(0.705, range=(0.4, 0.85))   # outer, handles at top/bottom
o_full_v         = param(0.415, range=(0.4, 0.85))   # outer, handles at the sides
o_in_full_h      = param(0.43, range=(0.4, 0.85))   # counter
o_in_full_v      = param(0.817, range=(0.4, 0.85))
o_stress         = param(-0.27,  range=(-20, 20))


@part
def round_bowl(p, width, height, stress, fullness, inner_fullness, thin):
    p.bowl(0, width, -overshoot, height + overshoot,
           thick=stem * round_gain, thin=thin,
           stress=stress, fullness=fullness, inner_fullness=inner_fullness)


@glyph("o")
def o(g):
    g += round_bowl(width=o_width, height=xheight, stress=o_stress,
                    fullness=(o_full_h, o_full_v), inner_fullness=(o_in_full_h, o_in_full_v),
                    thin=hairline, name="bowl")


@glyph("space", advance=240)
def space(g):
    pass
