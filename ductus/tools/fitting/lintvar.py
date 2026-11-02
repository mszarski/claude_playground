# lintvar.py glyphs "k=v,k=v" ... : lint errors + IoU for each variant
import sys, re, uuid
sys.path.insert(0, '/home/user/claude_playground/ductus')
from ductus.loader import load_font; from ductus.compile import compile_font
from ductus.lint import check_self_intersections, check_direction, check_stems
from ductus.bench.reference import Reference; from ductus.bench.run import score_font
from ductus.params import evaluating
ref = Reference('/home/user/claude_playground/ductus/benchmarks/geist_sans/reference/Geist-Regular.woff2')
src = open('/home/user/claude_playground/ductus/examples/sans/font.py').read()
gl = sys.argv[1].split(',')
for var in sys.argv[2:]:
    s = src
    ch = dict(kv.split('=') for kv in var.split(',')) if var else {}
    for k, v in ch.items():
        s = re.sub(rf"\b{k}=[-0-9.]+", f"{k}={v}", s, count=1) if re.search(rf"\b{k}=", s) else s.replace("P = kit.params(", f"P = kit.params({k}={v},", 1)
    fn = f'/tmp/claude-0/-home-user-claude-playground/9fa36f9b-b558-5a88-94f2-47889dc83674/scratchpad/sans/sw/_l{uuid.uuid4().hex[:6]}.py'
    open(fn, 'w').write(s)
    f = load_font(fn); cf = compile_font(f, glyphs=gl)
    def exp(st):
        if st.profile is not None or any(n.dw for n in st.nodes): return None
        return (st.pen or f.pen).width_at(90)
    errs = []
    with evaluating(cf.params):
        for g in gl:
            cg = cf.glyphs[g]
            errs += [f"{g}:{x.check}" for x in check_self_intersections(cg) + check_direction(cg) + check_stems(cg, exp) if x.level == "error"]
    sc = score_font(cf, ref, [g for g in gl if g in cf.glyphs and ref.name_for(cf.glyphs[g].unicode, g)])
    print(var or "current", errs, {g: round(s.iou, 3) for g, s in sc.items()})
