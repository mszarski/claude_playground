"""Render an iteration sheet for Ductus Display:
    python examples/display_serif/make_sheet.py 01
-> docs/display_serif/NN.png"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ductus.compile import compile_font
from ductus.emit import build_otf
from ductus.loader import load_font
from ductus.overrides import load_overrides
from ductus.proof import contact_sheet, render_text

HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[1] / "docs" / "display_serif"
it = sys.argv[1] if len(sys.argv) > 1 else "01"
font = load_font(HERE / "font.py")
ovs = load_overrides(HERE / "overrides.yaml")
build = DOCS / "_build"
build.mkdir(parents=True, exist_ok=True)

glyphs = [g for g in font.glyphs if g != "space"]
rows = [("defaults", compile_font(font, {}, ovs)),
        ("contrast 0.65", compile_font(font, {"contrast": 0.65}, ovs)),
        ("stress 0 · stem 120", compile_font(font, {"stress": 0, "stem": 120}, ovs))]
contact_sheet(rows, build / "construction.png")

otf = build_otf(rows[0][1], build / "DuctusDisplay.otf")
text = "".join(ch for ch in "oO o OoO oo" if any(g.unicode == ord(ch) for g in rows[0][1].glyphs.values()) or ch == " ")
parts = [Image.open(build / "construction.png").convert("L")]
for size in (120, 48, 18):
    parts.append(Image.open(render_text(otf, text, build / f"t{size}.png", size=size)))
W = max(p.width for p in parts)
sheet = Image.new("L", (W, sum(p.height for p in parts) + 30), 255)
ImageDraw.Draw(sheet).text((10, 8), f"Ductus Display -- iteration {it}", fill=0)
y = 30
for p in parts:
    sheet.paste(p, (0, y)); y += p.height
out = DOCS / f"{it}.png"
sheet.save(out)
print("wrote", out)
