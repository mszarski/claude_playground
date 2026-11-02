# tidy.py: merge runs of one-param lines in the display font's params into wrapped lines
import re
FP='/home/user/claude_playground/ductus/examples/sans/font.py'
s=open(FP).read()
def merge(m):
    items=[l.strip() for l in m.group(0).splitlines() if l.strip()]
    out=[]; row="   "
    for it in items:
        if len(row)+1+len(it)>112: out.append(row); row="   "
        row+=" "+it
    out.append(row); return "\n".join(out)+"\n"
s=re.sub(r"(?:^    [A-Za-z_][A-Za-z_0-9]*=[-0-9.]+,\n){2,}", merge, s, flags=re.M)
open(FP,'w').write(s)
