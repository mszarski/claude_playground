# fit per-glyph style weights (left-glyph right-side, right-glyph left-side) to Geist's kerning
import sys, json, numpy as np
sys.path.insert(0, '/home/user/claude_playground/ductus')
from ductus.loader import load_font; from ductus.compile import compile_font
from ductus.kerning import Kerner
from fontTools.agl import UV2AGL
N='/tmp/claude-0/-home-user-claude-playground/9fa36f9b-b558-5a88-94f2-47889dc83674/scratchpad'
d=json.load(open(N+'/sans/geist_kern.json')); pairs=[tuple(p) for p in d["pairs"]]; gk=np.array(d["gk"])
f=load_font('/home/user/claude_playground/ductus/examples/sans/font.py'); f.kerning={}
cf=compile_font(f)
nm=lambda c: c if c.isascii() and c.isalpha() else UV2AGL.get(ord(c))
keep=[i for i,(a,b) in enumerate(pairs) if a.isalpha() or b.isalpha()]       # no punct-punct
P=[(nm(pairs[i][0]), nm(pairs[i][1])) for i in keep]; G=gk[keep]
kr=Kerner(cf.glyphs, strength=0.6, punct_strength=0.3, clearance=90)
R=np.array([kr.correction(a,b) for a,b in P])
names=sorted({x for p in P for x in p}); ix={n:i for i,n in enumerate(names)}
wl=np.ones(len(names)); wr=np.ones(len(names))       # wl: as left glyph, wr: as right glyph
L=np.array([ix[a] for a,_ in P]); Rr=np.array([ix[b] for _,b in P])
lam=float(sys.argv[1]) if len(sys.argv)>1 else 2000.0
for it in range(30):
    for (w, other, own, oth) in ((wl, wr, L, Rr), (wr, wl, Rr, L)):
        x = R * other[oth]
        num = np.bincount(own, x*G, len(names)) + lam*1.0
        den = np.bincount(own, x*x, len(names)) + lam
        w[:] = np.clip(num/den, 0.0, 2.5)
fit = R*wl[L]*wr[Rr]
print("fit MAE (raw, no clearance)", round(np.mean(np.abs(fit-G)),1), "corr", round(np.corrcoef(fit,G)[0,1],3))
W={n:[round(float(wl[i]),2), round(float(wr[i]),2)] for n,i in ix.items()}
json.dump(W, open(N+'/sans/kern_weights.json','w'), indent=0)
print({n:W[n] for n in ["T","r","K","v","y","A","V","quotedblleft","quoteright","L","P","F","period","comma","o","a"]})
