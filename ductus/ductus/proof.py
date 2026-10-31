"""Visual proofs: render compiled glyphs to PNG (matplotlib) and render text
with the emitted .otf (FreeType) to check the binary itself."""
from __future__ import annotations

from pathlib import Path

import numpy as np

INK = "#1d1d1f"
SKEL = "#d1495b"
GUIDE = "#9aa5b1"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch
    return plt, MPath, PathPatch


def glyph_patch(cg, **kw):
    plt, MPath, PathPatch = _mpl()
    verts, codes = [], []
    for c in cg.contours:
        pts, types = c.pts, c.types
        verts.append(pts[0]); codes.append(MPath.MOVETO)
        n = len(pts)
        i = 1
        while i <= n:
            j = i % n
            if types[j] == "off":
                verts += [pts[j], pts[(i + 1) % n], pts[(i + 2) % n]]
                codes += [MPath.CURVE4] * 3
                i += 3
            else:
                verts.append(pts[j]); codes.append(MPath.LINETO)
                i += 1
        verts.append(pts[0]); codes.append(MPath.CLOSEPOLY)
    if not verts:
        return None
    return PathPatch(MPath(np.array(verts), codes), **kw)


def draw_glyph(ax, cg, metrics, skeleton=True, points=True, guides=True):
    if guides:
        for y in (0, metrics["xheight"], metrics["capheight"], metrics["ascender"]):
            ax.axhline(y, color=GUIDE, lw=0.4, zorder=0)
        ax.axvline(0, color=GUIDE, lw=0.4, ls=":")
        ax.axvline(cg.advance, color=GUIDE, lw=0.4, ls=":")
    patch = glyph_patch(cg, facecolor=INK, edgecolor="none", zorder=1)
    if patch is not None:
        ax.add_patch(patch)
    if skeleton:
        for s in cg.skeleton:
            for P in s["segments"]:
                P = np.array(P)
                t = np.linspace(0, 1, 30)[:, None]
                B = (1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3]
                ax.plot(B[:, 0], B[:, 1], color=SKEL, lw=0.8, zorder=2)
    if points:
        xy = np.array(list(cg.points.values())) if cg.points else np.zeros((0, 2))
        if len(xy):
            ax.scatter(xy[:, 0], xy[:, 1], s=6, color=SKEL, zorder=3)
            for c in cg.contours:
                on = np.array([p for p, k in zip(c.pts, c.types) if k == "on"])
                ax.scatter(on[:, 0], on[:, 1], s=2.5, color="#3c91e6", zorder=3)


def contact_sheet(rows: list[tuple[str, object]], out: Path, glyph_order=None, cell=1.6):
    """``rows`` = [(label, CompiledFont), ...]."""
    plt, *_ = _mpl()
    order = glyph_order or list(rows[0][1].glyphs)
    order = [g for g in order if g != "space"] + (["space"] if "space" in order else [])
    nrow, ncol = len(rows), len(order)
    widths = [max(cf.glyphs[g].advance for _, cf in rows) for g in order]
    height = nrow * cell * 1.15 + 0.35
    fig, axes = plt.subplots(nrow, ncol, figsize=(sum(widths) / 1000 * cell * 1.15 + 1.8, height),
                             gridspec_kw={"width_ratios": widths}, squeeze=False)
    for r, (label, cf) in enumerate(rows):
        for c, g in enumerate(order):
            ax = axes[r][c]
            cg = cf.glyphs[g]
            draw_glyph(ax, cg, cf.metrics)
            ax.set_xlim(-30, widths[c] + 30)
            ax.set_ylim(-260, 880)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(g, fontsize=9, color="#555")
            if c == 0:
                ax.set_ylabel(label, fontsize=7.5, color="#333", rotation=0, ha="right", va="center", labelpad=6)
    fig.subplots_adjust(left=0.13, right=0.99, top=1 - 0.3 / height, bottom=0.02, wspace=0.05, hspace=0.08)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    return out


def render_text(otf: Path, text: str, out: Path, size: int = 96, margin: int = 24):
    """Rasterise ``text`` with the compiled font through FreeType."""
    import freetype
    from PIL import Image
    face = freetype.Face(str(otf))
    face.set_pixel_sizes(0, size)
    asc = face.size.ascender >> 6
    desc = -(face.size.descender >> 6)
    pen_x, glyphs = 0, []
    for ch in text:
        face.load_char(ch, freetype.FT_LOAD_RENDER)
        g = face.glyph
        bmp = g.bitmap
        arr = np.array(bmp.buffer, dtype=np.uint8).reshape(bmp.rows, bmp.width) if bmp.rows else None
        glyphs.append((pen_x + g.bitmap_left, asc - g.bitmap_top, arr))
        pen_x += g.advance.x >> 6
    W, H = pen_x + 2 * margin, asc + desc + 2 * margin
    canvas = np.zeros((H, W), dtype=np.uint8)
    for x, y, arr in glyphs:
        if arr is None:
            continue
        x += margin; y += margin
        h, w = arr.shape
        canvas[y:y + h, x:x + w] = np.maximum(canvas[y:y + h, x:x + w], arr)
    Image.fromarray(255 - canvas).save(out)
    return out
