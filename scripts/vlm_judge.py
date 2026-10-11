"""A rater that watches the videos: a vision-language model sees frame strips of every answer and picks best / worst.

  python scripts/vlm_judge.py --items runs/rating_clean/set/items.json --videos runs/rating_clean/set/videos \
      --model moonshotai/Kimi-K3 --out runs/rating_clean/vlm_kimi.json [--only-rated runs/rating_clean/ratings_export]
  python scripts/judge_ratings.py compare --items ... --judge runs/rating_clean/vlm_kimi.json --ratings ...

The text-only judge (scripts/judge_ratings.py) agrees with people at chance: it cannot see the motion. Here each
item becomes one image: a row per answer (A, B, ... in a shuffled order), ``--frames`` frames evenly spread over the
clip, with the time under each frame. The model gets what the rating page shows besides the video: the earlier
lines, the words, the voice reading, and Reachy's reading of the person per answer. Votes are written in the
page's format, so ``judge_ratings.py compare`` measures agreement with the human ratings.
"""
import argparse
import base64
import io
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from judge_ratings import SCHEMA  # noqa: E402
from rating_agreement import load_votes  # noqa: E402

PROMPT = """Someone said something to Reachy Mini, a small desktop robot with a head that tilts, nods and turns, two
antenna ears and no face, which answers with body language only. The image shows {k} candidate responses, one per
row ({letters}); each row is {f} frames from its video, left to right, with the time under each frame. Pick the
response that answers BEST how the person feels and what they said, and the one that answers WORST. A good response
reads the person right and answers like a kind, emotionally intelligent companion: gentle and slow for sadness,
calm and steady for anger, reassuring for worry, sharing the joy for happiness, attentive and curious for neutral
talk. Watch the motion itself: how the head and ears move over time, how big and how fast.

{context}They said: "{text}"
Their voice sounded {voice}.

What Reachy took from them, per row:
{readings}

Reply with JSON: {{"best": "<letter>", "worst": "<letter>", "why": "<one sentence>"}}. Best and worst must differ."""


DENSE = PROMPT.replace(
    "The image shows {k} candidate responses, one per\nrow ({letters}); each row is {f} frames from its video, left to "
    "right, with the time under each frame.",
    "Each of the {k} candidate responses ({letters}) has its own image: its video\nas frames every "
    "{fps:g}th of a second, left to right and top to bottom, with the time under each frame, and below them its "
    "motion traces on fixed scales (head angles, ears, height over time), so you can compare size, speed and timing "
    "between responses.")


def strip_image(paths, frames, width):
    """One PNG: a labelled row of ``frames`` frames per video."""
    import imageio
    import numpy as np
    from PIL import Image, ImageDraw

    rows = []
    for p in paths:
        vid = np.stack(imageio.mimread(p, memtest=False))       # (T, H, W, 3), via imageio-ffmpeg
        w = vid.shape[2]
        vid = vid[:, :, w // 4: 3 * w // 4]                      # the robot stands in the middle half
        idx = [int(round(i * (len(vid) - 1) / (frames - 1))) for i in range(frames)]
        tiles = [Image.fromarray(vid[i]).resize((width, int(width * vid.shape[1] / vid.shape[2]))) for i in idx]
        rows.append((tiles, [i / 25 for i in idx]))
    tw, th, pad, lab = rows[0][0][0].width, rows[0][0][0].height, 16, 28
    img = Image.new("RGB", (lab + frames * tw, len(rows) * (th + pad)), "white")
    d = ImageDraw.Draw(img)
    for r, (tiles, ts) in enumerate(rows):
        y = r * (th + pad)
        d.text((6, y + th // 2 - 6), chr(65 + r), fill="black")
        for c, (t, s) in enumerate(zip(tiles, ts)):
            img.paste(t, (lab + c * tw, y))
            d.text((lab + c * tw + 2, y + th + 1), f"{s:.1f}s", fill="gray")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


_GEN = {}


def motion(recipe, video, ckpt):
    """The exact motion in a rating video, regenerated as scripts/rating_set.py rendered it (seed from the recipe
    hash): ``(T, 7)`` posture in plan units (earR earL pitch roll yaw z body; rmr.plan.posture)."""
    from rmr.generator.sample import generate_batch, load
    from rmr.motion import move_to_traj, traj_to_move
    from rmr.plan import posture
    from rmr.reach import Reach
    from rmr.recipe import variants

    if not _GEN:
        _GEN["gen"], _GEN["reach"] = load(ckpt), Reach()
    net, stats, dev = _GEN["gen"]
    seed = int(video[:8], 16) % 100000
    A = generate_batch(net, stats, variants(recipe, 1, seed=seed, fc=2.0, kdt=0.25), dev, seeds=[seed])[0]
    return posture(move_to_traj(_GEN["reach"].project(traj_to_move(A, ""))[0]))


def trace_image(P, width, seconds=15.0):
    """Motion traces on fixed scales, so answers compare at a glance: head pitch / roll / turn, ears, height."""
    import numpy as np
    from PIL import Image, ImageDraw

    panels = [("head (deg): pitch +down red, tilt green, turn blue", [(P[:, 2], "red"), (P[:, 3], "green"),
                                                                      (P[:, 4], "blue")], (-30, 30)),
              ("ears (deg): 0 up, 150 drooped; right red, left blue", [(P[:, 0], "red"), (P[:, 1], "blue")], (-20, 170)),
              ("height (mm)", [(P[:, 5], "black")], (-25, 25))]
    ph, lab = 70, 14
    img = Image.new("RGB", (width, len(panels) * (ph + lab) + 14), "white")
    d = ImageDraw.Draw(img)
    T = len(P)
    xs = np.arange(T) / 25 / seconds * (width - 1)
    for k, (name, lines, (lo, hi)) in enumerate(panels):
        y0 = k * (ph + lab) + lab
        d.text((2, y0 - lab + 1), name, fill="gray")
        d.rectangle([0, y0, width - 1, y0 + ph], outline="#cccccc")
        if lo < 0 < hi:
            yz = y0 + ph - (0 - lo) / (hi - lo) * ph
            d.line([(0, yz), (width - 1, yz)], fill="#e6e6e6")
        for v, col in lines:
            ys = y0 + ph - (np.clip(v, lo, hi) - lo) / (hi - lo) * ph
            d.line(list(zip(xs.tolist(), ys.tolist())), fill=col, width=2)
    for s in range(0, int(seconds) + 1, 2):                 # time ticks under the last panel
        x = s / seconds * (width - 1)
        d.text((min(x, width - 18), len(panels) * (ph + lab)), f"{s}s", fill="gray")
    return img


def answer_image(path, recipe, ckpt, fps, max_frames, tile, traces, head=False, cols=8):
    """One answer: its frames at ``fps`` (at most ``max_frames``, then evenly spread) in a grid of 8 columns, with
    the time under each frame, and (``traces``) its motion traces under the grid."""
    import imageio
    import numpy as np
    from PIL import Image, ImageDraw

    vid = np.stack(imageio.mimread(path, memtest=False))
    h, w = vid.shape[1], vid.shape[2]
    vid = vid[:, : int(h * 0.42), int(w * 0.3): int(w * 0.7)] if head else vid[:, :, w // 4: 3 * w // 4]
    n = min(max_frames, max(2, int(len(vid) / 25 * fps) + 1))
    idx = [int(round(i * (len(vid) - 1) / (n - 1))) for i in range(n)]
    th = int(tile * vid.shape[1] / vid.shape[2])
    rows = (n + cols - 1) // cols
    grid = Image.new("RGB", (cols * tile, rows * (th + 12)), "white")
    d = ImageDraw.Draw(grid)
    for k, i in enumerate(idx):
        x, y = (k % cols) * tile, (k // cols) * (th + 12)
        grid.paste(Image.fromarray(vid[i]).resize((tile, th)), (x, y))
        d.text((x + 2, y + th), f"{i / 25:.2f}s", fill="gray")
    if not traces:
        return grid
    tr = trace_image(motion(recipe, os.path.basename(path), ckpt), grid.width)
    out = Image.new("RGB", (grid.width, grid.height + tr.height), "white")
    out.paste(grid, (0, 0))
    out.paste(tr, (0, grid.height))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--model", default="moonshotai/Kimi-K3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--width", type=int, default=150)
    ap.add_argument("--only-rated", help="ratings export: judge only the items a person rated (to measure agreement)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-images", help="folder to keep the strips in (to look at)")
    ap.add_argument("--dense", action="store_true",
                    help="one image per answer: frames at --fps (not --frames over the clip) plus motion traces")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--no-traces", action="store_true")
    ap.add_argument("--head", action="store_true", help="crop the frames to the head and antennas (denser frames)")
    ap.add_argument("--tile", type=int, default=110, help="frame width in pixels")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--ckpt", default="checkpoints/generator_v2.pt", help="the generator the videos were made with")
    a = ap.parse_args()
    from rmr.planner.llm import _post, parse_json

    items = json.load(open(a.items))
    if a.only_rated:
        rated = {i for d in load_votes(a.only_rated) for i in ((d.get("data", d)).get("votes") or {})}
        items = [it for it in items if it["id"] in rated]
    part = a.out + ".part"
    done = dict(json.loads(x) for x in open(part)) if os.path.exists(part) else {}

    def pngs(it, order):
        """The images for one item, built up front (the generator is not thread-safe)."""
        if not a.dense:
            return [strip_image([os.path.join(a.videos, it["answers"][j]["video"]) for j in order], a.frames,
                                a.width)]
        out = []
        for j in order:
            x = it["answers"][j]
            img = answer_image(os.path.join(a.videos, x["video"]), x["recipe"], a.ckpt, a.fps, a.max_frames, a.tile,
                               not a.no_traces, head=a.head, cols=a.cols)
            buf = io.BytesIO()
            img.save(buf, "PNG")
            out.append(buf.getvalue())
        return out

    def one(job):
        it, order, images = job
        k = len(it["answers"])
        if a.save_images:
            os.makedirs(a.save_images, exist_ok=True)
            for n, png in enumerate(images):
                open(os.path.join(a.save_images, f"{it['id']}_{n}.png"), "wb").write(png)
        ctx = "".join(f'Earlier: "{c}"\n' for c in (it.get("context") or [])[-2:])
        readings = "\n".join(f"{chr(65 + p)}. {it['answers'][j]['reading']}" for p, j in enumerate(order))
        text = (DENSE if a.dense else PROMPT).format(k=k, letters=", ".join(chr(65 + p) for p in range(k)), f=a.frames,
                                                     fps=a.fps, context=ctx, text=it["text"], voice=it["voice"],
                                                     readings=readings)
        content = []
        for n, png in enumerate(images):
            if a.dense:
                content.append({"type": "text", "text": f"Response {chr(65 + n)}:"})
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64,"
                                                                      + base64.b64encode(png).decode()}})
        msg = [{"role": "user", "content": content + [{"type": "text", "text": text}]}]
        for _ in range(3):
            try:
                out = parse_json(_post({"model": a.model, "messages": msg, "temperature": 0.0,
                                        "response_format": {"type": "json_schema", "json_schema": {
                                            "name": "pick", "schema": SCHEMA, "strict": True}}},
                                       timeout=120)["choices"][0]["message"]["content"])
                b, w = (ord(str(out[key]).strip().upper()[:1]) - 65 for key in ("best", "worst"))
                if 0 <= b < k and 0 <= w < k and b != w:
                    return it["id"], {"best": order[b], "worst": order[w], "why": out.get("why", "")}
            except Exception as e:
                print(it["id"], type(e).__name__, str(e)[:120], flush=True)
        return it["id"], None

    votes = dict(done)
    with ThreadPoolExecutor(a.workers) as ex, open(part, "a") as f:
        futs = []
        for it in items:
            if it["id"] not in done:
                order = random.Random(f"vlm{it['id']}").sample(range(len(it["answers"])), len(it["answers"]))
                futs.append(ex.submit(one, (it, order, pngs(it, order))))
        for fut in as_completed(futs):
            i, v = fut.result()
            if v:
                votes[i] = v
                f.write(json.dumps([i, v]) + "\n")
                f.flush()
    with open(a.out, "w") as f:
        json.dump({"rater": a.model, "votes": votes}, f, indent=1)
    print(f"{len(votes)} of {len(items)} items judged -> {a.out}")


if __name__ == "__main__":
    main()
