#!/bin/bash
# ov.sh tag : overlay grid for variant
N=/tmp/claude-0/-home-user-claude-playground/9fa36f9b-b558-5a88-94f2-47889dc83674/scratchpad/sans
cd /home/user/claude_playground/ductus
rm -rf $N/docs_$1; cp $N/$1.py $N/f_$1.py
python -m ductus.cli bench $N/f_$1.py --reference benchmarks/geist_sans/reference/Geist-Regular.woff2 --docs $N/docs_$1 --iteration 1 ${2:+--glyphs $2} >/dev/null 2>&1
python -c "
from PIL import Image; import glob
fs=sorted(glob.glob('$N/docs_$1/iterations/01/*.png')); ims=[Image.open(f) for f in fs]
w=max(i.width for i in ims); h=max(i.height for i in ims); cols=min(6,len(ims)); rows=(len(ims)+cols-1)//cols
out=Image.new('RGB',(w*cols,h*rows),'white')
for k,i in enumerate(ims): out.paste(i,((k%cols)*w,(k//cols)*h))
out.save('$N/ov_$1.png')"
