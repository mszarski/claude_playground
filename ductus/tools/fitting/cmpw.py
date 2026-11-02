# cmpw.py text out.png : ours vs Geist rows
import sys
from ductus.loader import load_font; from ductus.compile import compile_font; from ductus.emit import build_otf; from ductus.proof import render_text
from PIL import Image
N='/tmp/claude-0/-home-user-claude-playground/9fa36f9b-b558-5a88-94f2-47889dc83674/scratchpad'
cf=compile_font(load_font('/home/user/claude_playground/ductus/examples/sans/font.py'))
for w in cf.warnings[:8]: print("warn:", w)
o=build_otf(cf, N+'/near/cur.otf')
ims=[]
for t in sys.argv[1].split('§'):
    for f in (o, N+'/sans/geist_sans.otf'):
        ims.append(Image.open(render_text(f, t, N+'/near/_r.png', size=int(sys.argv[3]) if len(sys.argv)>3 else 120)).convert('L'))
out=Image.new('L',(max(i.width for i in ims), sum(i.height for i in ims)),255); y=0
for i in ims: out.paste(i,(0,y)); y+=i.height
out.save(sys.argv[2])
