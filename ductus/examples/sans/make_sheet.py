"""Render an iteration sheet for Ductus Sans:
    python examples/sans/make_sheet.py NN
-> docs/sans/NN.png (+ NN_glyphs.png construction view)"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from ductus.compile import compile_font
from ductus.emit import build_otf
from ductus.loader import load_font
from ductus.overrides import load_overrides
from ductus.proof import contact_sheet, render_text

HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[1] / "docs" / "sans"
it = sys.argv[1] if len(sys.argv) > 1 else "01"
font = load_font(HERE / "font.py")
cf = compile_font(font, {}, load_overrides(HERE / "overrides.yaml"))
build = DOCS / "_build"
build.mkdir(parents=True, exist_ok=True)

lower = [g for g in "abcdefghijklmnopqrstuvwxyz" if g in cf.glyphs]
order = list(dict.fromkeys(lower)) + [g for g in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if g in cf.glyphs]
half = len(order) // 2 + 1 if len(order) < 40 else 26
contact_sheet([("", cf)], build / "c1.png", glyph_order=order[:half], cell=1.4)
contact_sheet([("", cf)], build / "c2.png", glyph_order=order[half:], cell=1.4)
glyph_view = [Image.open(build / "c1.png").convert("L"), Image.open(build / "c2.png").convert("L")]

otf = build_otf(cf, build / "DuctusSans.otf")
lines = [("hamburgefonstiv", 120), ("HOAGES  hamburgefonstiv", 72), ("Hamburgefonstiv OHAGES ishmon", 36),
         ("the shoe is on the other font", 24), ("hamburgefonstiv HOAGES", 14)]
if "z" in cf.glyphs:
    lines[1:1] = [("the quick brown fox jumps over a lazy dog", 72)]
if "Z" in cf.glyphs:
    lines[2:2] = [("THE QUICK BROWN FOX JUMPS OVER A LAZY DOG", 60),
                  ("Bold Ideas, Weird Mazes, Zany Jokes", 72)]
if "nine" in cf.glyphs:
    lines[3:3] = [("0123456789  1984 – 2026, 37/50 (8:15)", 72)]
if "eacute" in cf.glyphs:
    lines[4:4] = [("Ça déjà vu: naïve façade, crème brûlée — El Niño, Ångström, Zoë", 60),
                  ("Æsop’s Œuvre: søster, Straße, encyclopædia, œuvre — ØRE", 60),
                  ("ÀÁÂÃÄÅ ÇÈÉÊË ÌÍÎÏ ÑÒÓÔÕÖ ÙÚÛÜÝ àáâãäå çèéêë ìíîï ñòóôõö ùúûü ýÿ", 48)]
if "ampersand" in cf.glyphs:
    lines[6:6] = [("& * % # $ + < = > @ [ \\ ] ^ _ { | } ~ … • · ° × € £", 60)]
if "comma" in cf.glyphs:     # punctuation, set in a sentence and on its own
    lines[1:1] = [("“Ah,” she sighs; “is it (so) — or not?” Oh! it is: ’tis/isn’t.", 60),
                  (". , : ; - – — ' \" ‘ ’ “ ” ! ? ( ) /", 72)]
text = [Image.open(render_text(otf, s, build / f"t{i}.png", size=sz)) for i, (s, sz) in enumerate(lines)]
parts = text + glyph_view
W = max(p.width for p in parts)
sheet = Image.new("L", (W, sum(p.height for p in parts) + 30), 255)
ImageDraw.Draw(sheet).text((10, 8), f"Ductus Sans -- iteration {it}", fill=0)
y = 30
for p in parts:
    sheet.paste(p, (0, y)); y += p.height
out = DOCS / f"{it}.png"
sheet.save(out)
print("wrote", out, f"({len(cf.glyphs)} glyphs, {cf.compile_ms:.0f} ms)")
