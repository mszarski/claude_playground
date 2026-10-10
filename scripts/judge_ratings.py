"""An automatic rater for the response rating page: an LLM marks the best and the worst answer on every item.

  python scripts/judge_ratings.py judge --items runs/rating_clean/set/items.json --out runs/rating_clean/judge.json
  python scripts/judge_ratings.py compare --items runs/rating_clean/set/items.json \
      --judge runs/rating_clean/judge.json --ratings runs/rating_clean/ratings_export

``judge`` asks a different open model from the teacher (GLM-5.3 by default, so the teacher's answers are not graded
by their author) to make the same choice a person makes on deploy/rating_page.html. It sees what the person sees,
except the videos: the earlier lines, the words, the voice reading, and for each answer Reachy's reading of them, its
response sentence and a plain description of the motion computed from the recipe (how long, how fast the head
moves, nods, tilts, turns, leaning in or rising, the ears). Answers are shown in a shuffled order. It writes its
votes in the page's own format (``{"votes": {item id: {"best": i, "worst": j}}}``, indices into the item's answers),
so scripts/rating_agreement.py and scripts/dpo_from_ratings.py can read it like a rater, and prints the per-model
scores it implies on the test items.

``compare`` (after people have rated): how often the judge's best / worst match the human's, and over the pairs a
best-worst judgement orders, how often the judge orders them the same way (chance 50%). If it agrees well, it can
rate thousands more items for preference tuning; if not, the disagreements show where.
"""
import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from rating_agreement import load_votes  # noqa: E402

SCHEMA = {"type": "object", "properties": {"best": {"type": "string"}, "worst": {"type": "string"},
                                           "why": {"type": "string"}},
          "required": ["best", "worst", "why"], "additionalProperties": False}

PROMPT = """Someone said something to Reachy Mini, a small desktop robot with a head that tilts, nods and turns, two
antenna ears and no face, which answers with body language only. Several candidate responses are below. Pick the
one that responds BEST to how the person actually feels, and the one that responds WORST. A good response reads the
person right (words first; the automatic voice reading is often wrong, but real) and answers like a kind,
emotionally intelligent companion: gentle and slow for sadness, calm and steady for anger (never anger back),
reassuring for worry, sharing the joy for happiness, attentive and curious for neutral talk. Judge mostly by the
motion; the robot's reading explains its choice.

{context}They said: "{text}"
Their voice sounded {voice}.

{answers}

Reply with JSON: {{"best": "<letter>", "worst": "<letter>", "why": "<one sentence>"}}. Best and worst must differ."""


def describe(recipe):
    """A plain description of a recipe's motion (rmr.recipe.expand; plan channels rmr.plan.CH)."""
    from rmr.recipe import expand

    F = expand(recipe, np.random.default_rng(0))
    ears, pitch, roll, yaw, z, E = F[:, :2].mean(1), F[:, 2], F[:, 3], F[:, 4], F[:, 5], F[:, 7]
    speed = (np.abs(np.diff(F[:, 2:5], axis=0)) * 25).max() if len(F) > 1 else 0.0
    nods = int(((np.diff(np.sign(np.diff(pitch))) != 0) & (np.abs(np.diff(pitch))[1:] > 0.05)).sum() // 2)
    parts = [f"{len(F) / 25:.1f} s", "head moves " + ("very slowly" if speed < 30 else "slowly" if speed < 80 else
                                                    "briskly" if speed < 180 else "suddenly")]
    if pitch.max() > 8:
        parts.append(f"looks down up to {pitch.max():.0f} deg")
    if pitch.min() < -8:
        parts.append(f"looks up up to {-pitch.min():.0f} deg")
    if nods >= 2:
        parts.append(f"about {nods} nods")
    if np.abs(roll).max() > 8:
        parts.append(f"tilts its head {np.abs(roll).max():.0f} deg")
    if np.abs(yaw).max() > 10:
        parts.append(f"turns {np.abs(yaw).max():.0f} deg")
    if z.max() - z[0] > 5:
        parts.append(f"rises {z.max() - z[0]:.0f} mm")
    if z[0] - z.min() > 5:
        parts.append(f"sinks {z[0] - z.min():.0f} mm")
    # ears: 15 deg is neutral, lower is perked up, higher droops (rmr.recipe.NEUTRAL, LIMITS); energy runs 0..12
    ear = "ears perk up" if ears.min() < 5 else ""
    droop = "ears droop low" if ears.max() >= 80 else "ears lower" if ears.max() >= 40 else ""
    parts.append(" then ".join(x for x in (ear, droop) if x) or "ears at rest")
    parts.append(f"liveliness {E.max():.0f} of 12")
    return ", ".join(parts)


def judge(a):
    from rmr.planner.llm import chat_json

    items = json.load(open(a.items))

    def one(it):
        k = len(it["answers"])
        order = random.Random(f"judge{it['id']}").sample(range(k), k)
        lines = []
        for pos, j in enumerate(order):
            x = it["answers"][j]
            lines.append(f"{chr(65 + pos)}. Reachy reads them as: {x['reading']}\n   Response: {x['response']}\n"
                         f"   Motion: {describe(x['recipe'])}")
        ctx = "".join(f'Earlier: "{c}"\n' for c in (it.get("context") or [])[-2:])
        msg = PROMPT.format(context=ctx, text=it["text"], voice=it["voice"], answers="\n".join(lines))
        try:
            out = chat_json([{"role": "user", "content": msg}], SCHEMA, "pick", model=a.model, temperature=0.0)
            b, w = (ord(str(out[key]).strip().upper()[:1]) - 65 for key in ("best", "worst"))
            if not (0 <= b < k and 0 <= w < k) or b == w:
                raise ValueError(f"bad picks {out}")
            print(it["id"], "done", flush=True)
            return it["id"], {"best": order[b], "worst": order[w], "why": out.get("why", "")}
        except Exception as e:
            print(f"{it['id']}: {type(e).__name__}: {e}", flush=True)
            return it["id"], None

    with ThreadPoolExecutor(a.workers) as ex:
        votes = {i: v for i, v in ex.map(one, items) if v}
    with open(a.out, "w") as f:
        json.dump({"rater": a.model, "votes": votes}, f, indent=1)
    score = {}
    for it in items:
        v = votes.get(it["id"])
        if it["pool"] != "eval" or not v:
            continue
        for j, x in enumerate(it["answers"]):
            s = score.setdefault(x["model"], [0, 0, 0])
            s[0] += v["best"] == j
            s[1] += v["worst"] == j
            s[2] += 1
    print(f"{len(votes)} of {len(items)} items judged -> {a.out}\njudge's scores on the test items "
          "((best - worst) / shown):")
    for m, (b, w, n) in sorted(score.items(), key=lambda kv: -(kv[1][0] - kv[1][1]) / kv[1][2]):
        print(f"  {m:24s} best {b:3d}  worst {w:3d}  shown {n:3d}  score {(b - w) / n:+.2f}")


def orders(v, k):
    """The pairs (better, worse) a best-worst vote orders."""
    b, w = v["best"], v["worst"]
    return {(b, j) for j in range(k) if j != b} | {(j, w) for j in range(k) if j not in (b, w)}


def compare(a):
    items = {it["id"]: it for it in json.load(open(a.items))}
    jv = json.load(open(a.judge))["votes"]
    best = worst = n = agree = pairs = 0
    for d in load_votes(a.ratings):
        for i, v in ((d.get("data", d)).get("votes") or {}).items():
            if v.get("same") or i not in jv or i not in items or "best" not in v:
                continue
            k, j = len(items[i]["answers"]), jv[i]
            n += 1
            best += v["best"] == j["best"]
            worst += v["worst"] == j["worst"]
            hp, jp = orders(v, k), orders(j, k)
            for x, y in hp:                        # pairs both order: same way?
                if (x, y) in jp or (y, x) in jp:
                    pairs += 1
                    agree += (x, y) in jp
    if not n:
        print("no overlapping judgements yet")
        return
    k = len(next(iter(items.values()))["answers"])
    print(f"{n} judgements: same best {best / n:.0%} (chance {1 / k:.0%}), same worst {worst / n:.0%}, "
          f"pairs ordered the same way {agree}/{pairs} = {agree / max(pairs, 1):.0%} (chance 50%)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--items", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--model", default="zai-org/GLM-5.3")
    j.add_argument("--workers", type=int, default=8)
    c = sub.add_parser("compare")
    c.add_argument("--items", required=True)
    c.add_argument("--judge", required=True)
    c.add_argument("--ratings", required=True)
    a = ap.parse_args()
    {"judge": judge, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
