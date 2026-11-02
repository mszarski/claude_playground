# apply.py "{'k': v, ...}" : set params in the display font file (replace or insert into its group line)
import sys, re, ast
FP = '/home/user/claude_playground/ductus/examples/sans/font.py'
s = open(FP).read(); ch = ast.literal_eval(sys.argv[1])
for k, v in ch.items():
    v = int(v) if float(v).is_integer() and abs(v) >= 20 else v
    if re.search(rf"\b{k}=", s):
        s = re.sub(rf"\b{k}=[-0-9.]+", f"{k}={v!r}", s, count=1)
    else:
        pre = k.split('_')[0] + '_'
        m = list(re.finditer(rf"\n(    (?:[^\n]*\b{re.escape(pre)}[a-z0-9_]+=[^\n]*))", s[:s.index('\n)\nkit.define')]))
        at = m[-1].end() if m else s.index('\n)\nkit.define')
        s = s[:at] + f"\n    {k}={v!r}," + s[at:]
open(FP, 'w').write(s)
