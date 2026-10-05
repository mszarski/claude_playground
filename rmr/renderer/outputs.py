"""Videos, single-motion contact sheets, and labelled grid videos.

Reference: ``renderer/outputs.py`` in pham-tuan-binh/reachy-motion-generator (Apache-2.0).
"""
import os

import numpy as np

from ..motion import FPS
from .sim import Sim


def _label(img, text, h=16):
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, h], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 0))
    return np.asarray(im)


def _write(path, frames):
    import imageio.v2 as iio

    iio.mimwrite(path, frames, fps=FPS, quality=8, macro_block_size=1)


def videos(paths, outdir, width=520, height=420, log=print):
    os.makedirs(outdir, exist_ok=True)
    s, out = Sim(width, height), []
    for p in paths:
        fr = s.play_file(p)
        nm = os.path.basename(p)[:-5]
        o = os.path.join(outdir, nm + ".mp4")
        _write(o, fr)
        log(f"  {nm:48s} {len(fr) / FPS:5.1f}s -> {o}" + (f"  ({s.ik_fail} unreachable frames held)" if s.ik_fail else ""))
        out.append(o)
    return out


def sheet(paths, out, ncols=12, tile=170, log=print):
    """One row per motion, ``ncols`` evenly spaced frames: high temporal resolution for ONE prompt."""
    from PIL import Image

    s, rows = Sim(tile, tile), []
    for p in paths:
        fr = s.play_file(p)
        idx = np.linspace(0, len(fr) - 1, ncols).astype(int)
        rows.append(_label(np.concatenate([fr[i] for i in idx], 1),
                           f"{os.path.basename(p)[:-5][:60]}  [{len(fr) / FPS:.1f}s]", 12))
    Image.fromarray(np.concatenate(rows, 0)).save(out)
    log(f"sheet -> {out}")
    return out


def grid(paths, out, ncols=6, tile=260, log=print):
    """Tile several motions into one labelled video; shorter clips loop."""
    s, clips = Sim(tile, int(tile * 420 / 520)), []
    for p in paths:
        lab = os.path.basename(p)[:-5].replace("_", " ")
        clips.append(np.stack([_label(f, lab) for f in s.play_file(p)]))
    ncols = min(ncols, len(clips))
    T, (h, w) = max(len(c) for c in clips), clips[0].shape[1:3]
    nrow = -(-len(clips) // ncols)
    frames = np.zeros((T, nrow * h, ncols * w, 3), np.uint8)
    for i, c in enumerate(clips):
        r, k = divmod(i, ncols)
        for t in range(T):
            frames[t, r * h:(r + 1) * h, k * w:(k + 1) * w] = c[t % len(c)]
    _write(out, list(frames))
    log(f"grid ({len(clips)} motions) -> {out}")
    return out
