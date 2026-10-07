"""Tune the listening style from your own side-by-side ratings, instead of from a licensed dataset.

  python scripts/listen_rating.py render --clips runs/listen_rating/clips --out runs/listen_rating \
      [--listener-model runs/listener/listener.json]          # the learned (CC-BY-NC) model, as a yardstick only
  python scripts/listen_rating.py pairs --out runs/listen_rating
  python scripts/listen_rating.py page --urls runs/listen_rating/urls.json      # after uploading pairs/*.mp4
  python scripts/listen_rating.py fit --out runs/listen_rating --ratings runs/listen_rating/ratings.json

``render`` renders every candidate style in ``STYLES`` (rmr.listen.STYLE overrides) listening to every clip, with no
captions. ``pairs`` puts two styles side by side on the same clip, with the speech as the soundtrack: every pair of
styles once (up to ``--n``), clips assigned in turn, sides random. ``fit`` reads the rating page's votes and ranks the
styles with a Bradley-Terry model (each style has a strength; P(a beats b) = s_a / (s_a + s_b); a tie counts half
to each), with bootstrap intervals, so you can see which differences the votes actually support.

The clips come from the AMI meeting corpus (CC-BY 4.0; scripts in the docs say how they were cut), so nothing here
depends on non-commercial data unless you add the learned model as a yardstick.
"""
import argparse
import itertools
import json
import os
import random

STYLES = {
    "current": {},
    "still": {"nod_deg": 0.0, "perk": 0.0},
    "subtle": {"nod_deg": 4.0, "double_talk": 99.0, "sway_deg": 1.0},
    "lively": {"nod_deg": 9.0, "double_talk": 2.5, "sway_deg": 1.5, "glances": 2.0},
    "sway only": {"nod_deg": 0.0, "sway_deg": 1.5, "glances": 1.0},
    "frequent": {"nod_deg": 5.0, "pause": 0.2, "min_talk": 0.6, "sway_deg": 0.8},
    "sparse": {"nod_deg": 7.0, "pause": 0.5, "min_talk": 2.5, "double_talk": 99.0, "sway_deg": 0.8, "glances": 1.0},
    "leaning": {"lean": 1.8, "nod_deg": 5.0, "sway_deg": 1.0},
    "humanlike": {"nod_deg": 4.0, "min_talk": 1.5, "double_talk": 99.0, "sway_deg": 1.8, "glances": 3.0},
}
LEARNED = "learned"


def slug(name):
    return name.replace(" ", "_")


def _render_one(job):
    from rmr.listen import render

    clip, name, style, weights, path = job
    if os.path.exists(path):
        return path
    head = None
    if weights:
        from rmr.listen_model import LearnedHead
        head = LearnedHead(json.load(open(weights)), seed=1)
    render(clip, path, width=320, height=280, head=head, style=style, seed=1, labels=False)
    return path


def render_all(a):
    from multiprocessing import Pool

    clips = sorted(f for f in os.listdir(a.clips) if f.endswith(".wav"))
    os.makedirs(os.path.join(a.out, "renders"), exist_ok=True)
    styles = dict(STYLES)
    jobs = []
    for c in clips:
        for name, style in styles.items():
            jobs.append((os.path.join(a.clips, c), name, style, None,
                         os.path.join(a.out, "renders", f"{c[:-4]}__{slug(name)}.mp4")))
        if a.listener_model:
            jobs.append((os.path.join(a.clips, c), LEARNED, {"nod_deg": 0.0}, a.listener_model,
                         os.path.join(a.out, "renders", f"{c[:-4]}__{LEARNED}.mp4")))
    with Pool(a.workers) as pool:
        for k, p in enumerate(pool.imap_unordered(_render_one, jobs)):
            print(f"{k + 1}/{len(jobs)} {os.path.basename(p)}", flush=True)


def pairs(a):
    import subprocess

    import imageio_ffmpeg

    rd = os.path.join(a.out, "renders")
    names = sorted({f.split("__")[1][:-4] for f in os.listdir(rd) if f.endswith(".mp4")})
    clips = sorted({f.split("__")[0] for f in os.listdir(rd) if f.endswith(".mp4")})
    rng = random.Random(0)
    combos = list(itertools.combinations(names, 2))
    rng.shuffle(combos)
    combos = combos[:a.n]
    os.makedirs(os.path.join(a.out, "pairs"), exist_ok=True)
    items = []
    for i, (x, y) in enumerate(combos):
        left, right = (x, y) if rng.random() < 0.5 else (y, x)
        clip = clips[i % len(clips)]
        out = os.path.join(a.out, "pairs", f"p{i:02d}.mp4")
        if not os.path.exists(out):
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                            "-i", os.path.join(rd, f"{clip}__{left}.mp4"), "-i", os.path.join(rd, f"{clip}__{right}.mp4"),
                            "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]", "-map", "0:a",
                            "-c:v", "libx264", "-crf", "26", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags",
                            "+faststart", out], check=True)
        items.append({"id": f"p{i:02d}", "clip": clip, "left": left, "right": right, "video": os.path.basename(out)})
        print(f"{i + 1}/{len(combos)} {clip}: {left} | {right}", flush=True)
    with open(os.path.join(a.out, "pairs.json"), "w") as f:
        json.dump(items, f, indent=1)


def bradley_terry(names, games, iters=500):
    """games: (winner, loser, weight). Minorisation-maximisation (Hunter 2004), with a weak prior (one tie against
    a virtual average opponent) so a style that never won doesn't go to zero."""
    s = {n: 1.0 for n in names}
    for _ in range(iters):
        new = {}
        for n in names:
            wins = 0.5 + sum(w for a, b, w in games if a == n)
            denom = 1.0 / (s[n] + 1.0)
            for a, b, w in games:
                if n in (a, b):
                    denom += w / (s[a] + s[b])
            new[n] = wins / denom
        g = sum(new.values()) / len(new)
        s = {n: v / g for n, v in new.items()}
    return s


def fit(a):
    import numpy as np

    items = {it["id"]: it for it in json.load(open(os.path.join(a.out, "pairs.json")))}
    raw = json.load(open(a.ratings))
    docs = raw if isinstance(raw, list) else raw.get("documents", [])
    games = []
    for d in docs:
        for pid, v in ((d.get("data", d)).get("votes") or {}).items():
            it = items.get(pid)
            if not it:
                continue
            if v["choice"] == "same":
                games += [(it["left"], it["right"], 0.5), (it["right"], it["left"], 0.5)]
            else:
                win = it[v["choice"]]
                lose = it["right"] if win == it["left"] else it["left"]
                games.append((win, lose, 1.0))
    names = sorted({n for it in items.values() for n in (it["left"], it["right"])})
    s = bradley_terry(names, games)
    rng = np.random.default_rng(0)
    boots = {n: [] for n in names}
    decisive = [g for g in games if g[2] == 1.0]
    ties = [g for g in games if g[2] == 0.5]
    for _ in range(200):
        sample = [decisive[i] for i in rng.integers(0, len(decisive), len(decisive))] if decisive else []
        bs = bradley_terry(names, sample + ties, iters=200)
        for n in names:
            boots[n].append(np.log(bs[n]))
    ranked = sorted(names, key=lambda n: -s[n])
    print(f"{len(decisive)} decisive votes, {len(ties) // 2} ties\n")
    print(f"{'style':12s} {'strength':>9s} {'90% interval':>18s}  win rate vs the average style")
    rows = []
    for n in ranked:
        lo, hi = np.exp(np.percentile(boots[n], [5, 95]))
        p = s[n] / (s[n] + 1)
        rows.append({"style": n, "strength": round(s[n], 3), "lo": round(lo, 3), "hi": round(hi, 3),
                     "params": STYLES.get(n, "learned model")})
        print(f"{n:12s} {s[n]:9.2f} {lo:8.2f} - {hi:6.2f}   {p:.0%}")
    with open(os.path.join(a.out, "fit.json"), "w") as f:
        json.dump(rows, f, indent=1)


def page(a):
    clips = {c["clip"]: c for c in json.load(open(os.path.join(a.clips, "clips.json")))}
    urls = json.load(open(a.urls))
    items = [{"id": it["id"], "url": urls[it["video"]], "text": clips[it["clip"]]["text"],
              "left_style": it["left"], "right_style": it["right"]}
             for it in json.load(open(os.path.join(a.out, "pairs.json"))) if it["video"] in urls]
    tpl = open(os.path.join(os.path.dirname(__file__), "..", "deploy", "listen_rating_page.html")).read()
    with open(a.html, "w") as f:
        f.write(tpl.replace("__ITEMS__", json.dumps(items).replace("</", "<\\/")))
    print(f"{len(items)} comparisons -> {a.html}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--clips", default="runs/listen_rating/clips")
    r.add_argument("--out", default="runs/listen_rating")
    r.add_argument("--listener-model")
    r.add_argument("--workers", type=int, default=os.cpu_count())
    p = sub.add_parser("pairs")
    p.add_argument("--out", default="runs/listen_rating")
    p.add_argument("--n", type=int, default=40)
    g = sub.add_parser("page")
    g.add_argument("--out", default="runs/listen_rating")
    g.add_argument("--clips", default="runs/listen_rating/clips")
    g.add_argument("--urls", required=True, help="video file name -> asset URL")
    g.add_argument("--html", default="runs/listen_rating/reachy_listening.html")
    f = sub.add_parser("fit")
    f.add_argument("--out", default="runs/listen_rating")
    f.add_argument("--ratings", required=True)
    a = ap.parse_args()
    {"render": render_all, "pairs": pairs, "page": page, "fit": fit}[a.cmd](a)


if __name__ == "__main__":
    main()
