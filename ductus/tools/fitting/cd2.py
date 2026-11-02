# cd2.py glyphs p:step ... : coordinate descent on the current display font file; prints best changes
import sys, uuid, re
from ductus.loader import load_font; from ductus.compile import compile_font
from ductus.bench.reference import Reference; from ductus.bench.run import score_font
from ductus.kits.serif import DEFAULTS
ref = Reference('/home/user/claude_playground/ductus/benchmarks/geist_sans/reference/Geist-Regular.woff2')
PUNCT = 'Oslash oslash AE OE ae oe germandbls zero one two three four five six seven eight nine period comma colon semicolon quoteright quoteleft quotedblright quotedblleft quotesingle quotedbl hyphen endash emdash exclam question parenleft parenright slash'.split()
FP = '/home/user/claude_playground/ductus/examples/sans/font.py'
src = open(FP).read(); gl = [x for x in sys.argv[1].split(",") if x] if ("," in sys.argv[1] or sys.argv[1] in PUNCT) else list(sys.argv[1])
ps = [(a.split(':')[0], float(a.split(':')[1])) for a in sys.argv[2:]]
def build(ch):
    s = src.replace("P = kit.params(", f"P = kit.params(**{ch!r}, **{{k: v for k, v in dict(", 1).replace("\n)\nkit.define", "\n).items() if k not in " + repr(list(ch)) + "})\nkit.define", 1)
    fn = f'{sys.path[0]}/sw/_c{uuid.uuid4().hex[:8]}.py'; open(fn, 'w').write(s); return fn
def score(ch):
    try:
        cf = compile_font(load_font(build(ch)), glyphs=gl); sc = score_font(cf, ref, gl)
        return sum(x.iou for x in sc.values()) / len(sc)
    except Exception: return -1
f0 = load_font(FP)
def cur(k):
    m = re.search(rf"\b{k}=([-0-9.]+)", src)
    if m: return float(m.group(1))
    d = DEFAULTS[k]; return d[0] if isinstance(d, tuple) else d
B = {}; best = score(B); print("start", round(best, 4))
for rnd in range(3):
    imp = False
    for k, st in ps:
        for sg in (1, -1):
            while True:
                lo, hi = DEFAULTS[k][1]
                v = round(B.get(k, cur(k)) + sg * st, 4)
                if not lo <= v <= hi: break
                s = score({**B, k: v})
                if s > best + 1e-4: best = s; B[k] = v; imp = True
                else: break
    if not imp: break
print("end", round(best, 4), B)
