# usage: python exp.py font.py tag  -> per-glyph IoU, mean, and a words image vs Geist
import sys, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from ductus.loader import load_font; from ductus.compile import compile_font
from ductus.emit import build_otf; from ductus.proof import render_text
from ductus.bench.reference import Reference; from ductus.bench.run import score_font
R = Path('/home/user/claude_playground/ductus')
SP = Path(__file__).parent
fp, tag = sys.argv[1], sys.argv[2]
f = load_font(fp); cf = compile_font(f)
ref = Reference(R / 'benchmarks/geist_sans/reference/Geist-Regular.woff2')
names = [n for n, g in cf.glyphs.items() if g.contours and ref.name_for(g.unicode, n) is not None]
sc = score_font(cf, ref, names)
print(" ".join(f"{g}:{s.iou:.2f}" for g, s in sc.items()))
print(f"MEAN IoU {np.mean([s.iou for s in sc.values()]):.4f}  mean|dadv| {np.mean([abs(s.advance_delta) for s in sc.values()]):.0f}")
otf = build_otf(cf, SP / f"{tag}.otf"); geist = SP.parent / 'vs/geist.otf'
ims = []
for t in ("hamburgefonstiv", "HOAGES the shoe is on"):
    for o in (otf, geist):
        ims.append(Image.open(render_text(o, t, SP / "_w.png", size=96)).convert("L"))
W = max(i.width for i in ims); H = sum(i.height for i in ims)
out = Image.new("L", (W, H), 255); y = 0
for i in ims: out.paste(i, (0, y)); y += i.height
out.save(SP / f"{tag}.png")
