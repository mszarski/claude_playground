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
    a = ap.parse_args()
    from rmr.planner.llm import _post, parse_json

    items = json.load(open(a.items))
    if a.only_rated:
        rated = {i for d in load_votes(a.only_rated) for i in ((d.get("data", d)).get("votes") or {})}
        items = [it for it in items if it["id"] in rated]
    part = a.out + ".part"
    done = dict(json.loads(x) for x in open(part)) if os.path.exists(part) else {}

    def one(it):
        k = len(it["answers"])
        order = random.Random(f"vlm{it['id']}").sample(range(k), k)
        png = strip_image([os.path.join(a.videos, it["answers"][j]["video"]) for j in order], a.frames, a.width)
        if a.save_images:
            os.makedirs(a.save_images, exist_ok=True)
            open(os.path.join(a.save_images, it["id"] + ".png"), "wb").write(png)
        ctx = "".join(f'Earlier: "{c}"\n' for c in (it.get("context") or [])[-2:])
        readings = "\n".join(f"{chr(65 + p)}. {it['answers'][j]['reading']}" for p, j in enumerate(order))
        text = PROMPT.format(k=k, letters=", ".join(chr(65 + p) for p in range(k)), f=a.frames, context=ctx,
                             text=it["text"], voice=it["voice"], readings=readings)
        msg = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
            {"type": "text", "text": text}]}]
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
        for fut in as_completed([ex.submit(one, it) for it in items if it["id"] not in done]):
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
