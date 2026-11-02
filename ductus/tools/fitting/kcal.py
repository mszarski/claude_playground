import sys, itertools, numpy as np
sys.path.insert(0, '/home/user/claude_playground/ductus')
from fontTools.ttLib import TTFont
from ductus.loader import load_font; from ductus.compile import compile_font
from ductus.kerning import Kerner
N='/tmp/claude-0/-home-user-claude-playground/9fa36f9b-b558-5a88-94f2-47889dc83674/scratchpad'
t=TTFont(N+'/sans/geist_sans.otf'); cm=t.getBestCmap(); G=t['GPOS'].table
def geist_kern(A, B):
    for fr in G.FeatureList.FeatureRecord:
        if fr.FeatureTag!='kern': continue
        for idx in fr.Feature.LookupListIndex:
            lk=G.LookupList.Lookup[idx]
            for st in lk.SubTable:
                if lk.LookupType==9: st=st.ExtSubTable
                if getattr(st,'LookupType',2)!=2 and lk.LookupType not in (2,9): continue
                if not hasattr(st,'Coverage') or A not in st.Coverage.glyphs: continue
                if st.Format==1:
                    ps=st.PairSet[st.Coverage.glyphs.index(A)]
                    for pv in ps.PairValueRecord:
                        if pv.SecondGlyph==B: return (pv.Value1.XAdvance or 0) if pv.Value1 else 0
                else:
                    c1=st.ClassDef1.classDefs.get(A,0); c2=st.ClassDef2.classDefs.get(B,0)
                    v=st.Class1Record[c1].Class2Record[c2].Value1
                    if v is not None and getattr(v,'XAdvance',0): return v.XAdvance
    return 0
chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;-'\"‘’“”!?()/"
f=load_font('/home/user/claude_playground/ductus/examples/sans/font.py')
f.kerning={}
cf=compile_font(f)
from fontTools.agl import UV2AGL
ours={c:(c if c.isascii() and c.isalpha() else UV2AGL.get(ord(c))) for c in chars}
pairs=[(a,b) for a in chars for b in chars]
gk=np.array([geist_kern(cm[ord(a)], cm[ord(b)]) for a,b in pairs], float)
print("geist pairs kerned:", int((gk!=0).sum()), "of", len(pairs))
import json; json.dump({"pairs": pairs, "gk": gk.tolist()}, open(N+'/sans/geist_kern.json','w'))
