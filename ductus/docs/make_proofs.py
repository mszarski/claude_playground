"""Regenerate the proofs in docs/:  python docs/make_proofs.py"""
from pathlib import Path

from ductus.compile import compile_font
from ductus.emit import build_otf
from ductus.loader import load_font
from ductus.overrides import load_overrides
from ductus.proof import contact_sheet, render_text
from ductus.terminals import serifs

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "examples" / "test_font"

SETTINGS = [
    ("Light\nstem 40 · contrast .2\nx-height 500", {"stem": 40, "contrast": 0.2, "xheight": 500}),
    ("Regular\n(defaults)\nstem 80 · contrast .45", {}),
    ("Heavy\nstem 160 · contrast .6\nx-height 570", {"stem": 160, "contrast": 0.6, "xheight": 570}),
]

font = load_font(SRC / "font.py")
ovs = load_overrides(SRC / "overrides.yaml")
rows = [(label, compile_font(font, params, ovs)) for label, params in SETTINGS]
contact_sheet(rows, HERE / "contact_sheet.png")
contact_sheet([("Regular\nserifs.slab()", compile_font(font, {}, ovs, terminals=serifs.slab()))],
              HERE / "contact_sheet_slab.png")

# text set from the compiled .otf files through FreeType (checks the binary)
from PIL import Image
lines = []
for label, params in SETTINGS:
    otf = build_otf(compile_font(font, params, ovs), HERE / "_build" / f"{label.split()[0]}.otf")
    lines.append(Image.open(render_text(otf, "nohm Hommo OHO mhn", HERE / "_build" / f"{label.split()[0]}.png", size=72)))
W = max(im.width for im in lines)
sheet = Image.new("L", (W, sum(im.height for im in lines)), 255)
y = 0
for im in lines:
    sheet.paste(im, (0, y)); y += im.height
sheet.save(HERE / "otf_text.png")
print("wrote", HERE / "contact_sheet.png", HERE / "contact_sheet_slab.png", HERE / "otf_text.png")
