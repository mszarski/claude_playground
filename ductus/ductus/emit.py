"""Write a static CFF-flavoured .otf with fontTools.fontBuilder."""
from __future__ import annotations

import re
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2CharStringPen import T2CharStringPen

from .compile import CompiledFont


def ps_name(cf: CompiledFont) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", cf.family) + "-" + re.sub(r"[^A-Za-z0-9]", "", cf.style)


def build_otf(cf: CompiledFont, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = {k: round(v) for k, v in cf.metrics.items()}
    upm = cf.upm
    order = [".notdef"] + list(cf.glyphs)
    fb = FontBuilder(upm, isTTF=False)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap({g.unicode: g.name for g in cf.glyphs.values() if g.unicode is not None})

    charstrings, metrics = {}, {}
    # .notdef: an open box
    nd_w = round(upm * 0.5)
    pen = T2CharStringPen(nd_w, None)
    t = 50
    for x0, y0, x1, y1, cw in ((50, 0, nd_w - 50, m["capheight"], False),
                              (50 + t, t, nd_w - 50 - t, m["capheight"] - t, True)):
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        if cw:
            pts = pts[::-1]
        pen.moveTo(pts[0]); [pen.lineTo(p) for p in pts[1:]]; pen.closePath()
    charstrings[".notdef"] = pen.getCharString()
    metrics[".notdef"] = (nd_w, 50)
    for g in cf.glyphs.values():
        pen = T2CharStringPen(g.advance, None)
        for c in g.contours:
            c.draw(pen)
        charstrings[g.name] = pen.getCharString()
        metrics[g.name] = (g.advance, int(round(g.bounds[0])) if g.bounds else 0)

    fb.setupCFF(ps_name(cf), {"FullName": f"{cf.family} {cf.style}", "FamilyName": cf.family,
                              "Weight": cf.style}, charstrings, {})
    fb.setupHorizontalMetrics(metrics)
    asc, desc = m["ascender"], m["descender"]
    fb.setupHorizontalHeader(ascent=asc, descent=desc, lineGap=0)
    fb.setupNameTable({"familyName": cf.family, "styleName": cf.style,
                       "uniqueFontIdentifier": f"Ductus;{ps_name(cf)}",
                       "fullName": f"{cf.family} {cf.style}", "psName": ps_name(cf),
                       "version": "Version 0.001"})
    fb.setupOS2(sTypoAscender=asc, sTypoDescender=desc, sTypoLineGap=0,
                usWinAscent=max(asc, 0) + 20, usWinDescent=abs(desc) + 20,
                sxHeight=m["xheight"], sCapHeight=m["capheight"], achVendID="NONE",
                fsType=0)
    fb.setupPost()
    fb.setupHead(unitsPerEm=upm)
    fb.save(str(path))
    return path


def validate_otf(path) -> list[str]:
    """Round-trip through fontTools and run OTS if installed. Returns problems."""
    from fontTools.ttLib import TTFont
    problems = []
    try:
        f = TTFont(str(path))
        for tag in ("CFF ", "cmap", "head", "hhea", "hmtx", "maxp", "name", "OS/2", "post"):
            if tag not in f:
                problems.append(f"missing table {tag!r}")
        cs = f["CFF "].cff.topDictIndex[0].CharStrings
        for name in f.getGlyphOrder():
            cs[name].decompile()
    except Exception as e:
        problems.append(f"fontTools round-trip failed: {e}")
    try:
        import ots
    except ImportError:
        return problems
    r = ots.sanitize(str(path), "/dev/null", capture_output=True)
    if r.returncode != 0:
        problems.append("ots-sanitize: " + (r.stderr or r.stdout).decode(errors="replace").strip())
    return problems
