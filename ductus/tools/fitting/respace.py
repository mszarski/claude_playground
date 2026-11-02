import sys, re
from fontTools.ttLib import TTFont
from fontTools.pens.boundsPen import BoundsPen
N='/tmp/claude-0/-home-user-claude-playground/9fa36f9b-b558-5a88-94f2-47889dc83674/scratchpad/sans'
def sb(path):
    t=TTFont(path); gs=t.getGlyphSet(); cm=t.getBestCmap(); out={}
    for c in "hamburgefonstivHOAGEScdjklpqwxyzBCDFIJKLMNPQRTUVWXYZ0123456789" + "ÆŒØæœøß" + "&*%·#$+<=>@[\\]_{|}~…•°×€" + ".,:;-–—'\"‘’“”!?()/":
        if ord(c) not in cm: continue
        g=gs[cm[ord(c)]]; p=BoundsPen(gs); g.draw(p); b=p.bounds; out[c]=(b[0], g.width-b[2], b[2]-b[0])
    return out
a=sb(sys.argv[1]); b=sb(N+'/../sans/geist_sans.otf')
FP='/home/user/claude_playground/ductus/examples/sans/font.py'
src=open(FP).read()
from fontTools.agl import UV2AGL
def gname(c): return c if (c.isascii() and c.isalpha()) else UV2AGL.get(ord(c), c)
for c in a:
    if c not in b: continue
    if abs(a[c][2]-b[c][2])>25: print("ink width differs", c, round(a[c][2]), round(b[c][2]))
    dl, dr = round((b[c][0]-a[c][0])/2)*2, round((b[c][1]-a[c][1])/2)*2
    if abs(dl)<2 and abs(dr)<2: continue
    nm=gname(c)
    m=re.search(rf'    "{re.escape(nm)}": \((-?\d+), (-?\d+)\),', src)
    l0,r0=(int(m.group(1)),int(m.group(2))) if m else (0,0)
    new=f'    "{nm}": ({l0+dl}, {r0+dr}),'
    print(c, (l0,r0), '->', (l0+dl, r0+dr))
    src = src.replace(m.group(0), new) if m else src.replace('font.spacing.update({\n', 'font.spacing.update({\n'+new+'\n')
open(FP,'w').write(src)
