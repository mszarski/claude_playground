"""Ductus Test -- the milestone-1 glyph set: n h m o H O space."""
from ductus import Font, Pen, glyph, param, part, serifs

font = Font("Ductus Test", upm=1000)

# --- shared parameters (ranges drive preview sliders and, later, axes) ----
stem      = param(80,   range=(30, 180))
contrast  = param(0.45, range=(0.0, 0.9))    # 0 = monoline, 0.9 = hairline thins
xheight   = param(530,  range=(450, 600))
capheight = param(710,  range=(620, 780))
ascender  = param(740,  range=(680, 820))
overshoot = param(12,   range=(0, 25))
counter   = param(300,  range=(200, 420))    # inner width of the n
roundness = param(1.06, range=(0.9, 1.25))   # o counter relative to n counter

font.pen       = Pen(angle=param(20, range=(0, 45)), width=stem, contrast=contrast)
font.ascender  = ascender
font.descender = -220
font.xheight   = xheight
font.capheight = capheight
font.terminals = serifs.none


# --- parts ---------------------------------------------------------------

@part
def stem_(p, x, top, bottom=0):
    p.line((x, bottom), (x, top), name="stem", names=("bottom", "top"))


@part
def arch(p, x, width, top=xheight):
    """From inside the stem, up over the shoulder, down to the baseline.
    Vertical positions keep the *outer* edge on x-height + overshoot."""
    pen = p.pen
    shoulder_y = top + overshoot - pen.width_at(0) / 2
    drop = width * 0.42
    (p.path(name="arch")
        .start((x, shoulder_y - drop * 1.25), name="join", dir=68, terminal=False)
        .curve_to((x + width * 0.53, shoulder_y), name="shoulder", dir=0)
        .curve_to((x + width, shoulder_y - drop), name="shoulder_out", dir=-90)
        .line_to((x + width, 0), name="leg"))


@part
def bowl(p, x, width, height):
    """Closed elliptical bowl whose outer edge spans x..x+width and
    -overshoot..height+overshoot (vertical edges carry the full stem)."""
    pen = p.pen
    half_v = pen.width_at(0) / 2       # horizontal travel at top/bottom
    half_h = pen.width_at(90) / 2      # vertical travel at the sides
    left, right = x + half_h, x + width - half_h
    bottom, top = -overshoot + half_v, height + overshoot - half_v
    cx, cy = (left + right) / 2, (bottom + top) / 2
    (p.path(name="bowl")
        .start((cx, bottom), name="bottom", dir=0)
        .curve_to((right, cy), name="right", dir=90)
        .curve_to((cx, top), name="top", dir=180)
        .curve_to((left, cy), name="left", dir=270)
        .close())


@part
def bar(p, x0, x1, y):
    p.line((x0, y), (x1, y), name="bar", names=("left", "right"), terminal=False)


# --- glyphs --------------------------------------------------------------

n_width = counter + stem    # stem centre to stem centre

@glyph("n")
def n(g):
    g += stem_(x=0, top=xheight)
    g += arch(x=0, width=n_width)


@glyph("h")
def h(g):
    g.inherit("n")
    g["stem"].top = ascender


@glyph("m")
def m(g):
    w = (counter * 0.82 + stem)
    g += stem_(x=0, top=xheight)
    g += arch(x=0, width=w, name="arch1")
    g += arch(x=w, width=w, name="arch2")


@glyph("o")
def o(g):
    g += bowl(x=0, width=counter * roundness + stem * 2, height=xheight)


@glyph("H")
def H(g):
    w = counter * 1.45 + stem
    g += stem_(x=0, top=capheight, name="left")
    g += stem_(x=w, top=capheight, name="right")
    g += bar(x0=0, x1=w, y=capheight * 0.52)


@glyph("O")
def O(g):
    g += bowl(x=0, width=capheight * 0.86 + stem * 0.6, height=capheight)


@glyph("space", advance=counter * 0.6 + stem * 0.5)
def space(g):
    pass
