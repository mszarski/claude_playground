"""Download Geist Serif (SIL OFL 1.1) from the `geist` npm package into
./reference/ for measurement and overlays ONLY.

Licensing rule (SPEC section 8): never copy, trace or import its outlines
into a Ductus font. Anything fitted to it stays in benchmarks/ and is never
released. The reference directory is git-ignored.

    python benchmarks/geist_serif/fetch_reference.py
"""
import io
import json
import tarfile
import urllib.request
from pathlib import Path

VERSION = "1.9.0"
WANT = {"package/dist/fonts/geist-serif/GeistSerif-Regular.woff2", "package/LICENSE.txt"}

here = Path(__file__).resolve().parent / "reference"
here.mkdir(exist_ok=True)
meta = json.load(urllib.request.urlopen(f"https://registry.npmjs.org/geist/{VERSION}"))
data = urllib.request.urlopen(meta["dist"]["tarball"]).read()
with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
    for m in tf.getmembers():
        if m.name in WANT or m.name.endswith("/LICENSE.txt") and "geist-serif" in m.name:
            out = here / Path(m.name).name
            out.write_bytes(tf.extractfile(m).read())
            print("wrote", out)
