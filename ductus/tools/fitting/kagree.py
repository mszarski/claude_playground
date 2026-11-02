import json, sys, numpy as np
from fontTools.agl import UV2AGL
N='/tmp/claude-0/-home-user-claude-playground/9fa36f9b-b558-5a88-94f2-47889dc83674/scratchpad'
d=json.load(open(N+'/sans/geist_kern.json')); pairs=[tuple(p) for p in d["pairs"]]; gk=np.array(d["gk"])
nm=lambda c: c if c.isascii() and c.isalpha() else UV2AGL.get(ord(c))
for path in sys.argv[1:]:
    raw=json.load(open(path)); K={}
    items = raw["pairs"]
    for it in items:
        if isinstance(it, (list,tuple)) and len(it)==3: K[(it[0],it[1])]=it[2]
        else:
            k,v=it; a,b=(k.split(" ",1) if isinstance(k,str) else k); K[(a,b)]=v
    ok=np.array([K.get((nm(a),nm(b)),0) for a,b in pairs],float)
    let=np.array([a.isalpha() and b.isalpha() for a,b in pairs])
    for lab,m in (("letters",let),("all",np.ones_like(let))):
        print(path.split('/')[-1], lab, "corr %.3f MAE %.1f  kerned ours %d geist %d" % (np.corrcoef(ok[m],gk[m])[0,1], np.mean(np.abs(ok[m]-gk[m])), (ok[m]!=0).sum(), (gk[m]!=0).sum()))
