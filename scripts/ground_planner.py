"""Ground the motion planner in how Pollen's animator actually moves: real clips as recipe examples.

  python scripts/ground_planner.py library --out runs/ground/library.json     # transcribe the real clips
  python scripts/ground_planner.py fidelity --library runs/ground/library.json
  python scripts/ground_planner.py eval --library runs/ground/library.json --model moonshotai/Kimi-K3 \
      --samples 4 --out runs/ground/kimi_grounded.json [--plain]

The planner's recipes for the 12 held-out emotions are identified among the real clips 20-28% of the time
(docs/results/planner_zeroshot.md, planner_finetune.md); given the clip's true plan the generator gets 91.7%. The gap
is taste: how an LLM imagines "frustrated" moves versus how Pollen's animator made it move. Here the LLM sees that
taste directly.

* ``library``: every emotion clip *except* the held-out emotions (and any other clip of the same word) becomes a
  recipe: its plan (rmr.plan.extract, as the generator was trained on) transcribed into one ``go`` segment per
  ``--step`` seconds (ears, pitch, roll, yaw, height, body, and the energy of the fast detail), plus its caption.
* ``fidelity``: the same transcription of the held-out clips, through the generator: how much a recipe can carry
  (the ceiling for a planner writing in this style).
* ``eval``: the planner system prompt (rmr.planner.prompt.SYSTEM) plus the transcribed library as examples of how
  Reachy's animator moves; ``--plain`` leaves the library out (the baseline). ``--samples`` recipes per held-out
  emotion at temperature 0.7, scored like rmr.planner.evaluate (5 plan variants each, 100 flow steps, top-1 and
  mean rank among the 12 held-out clips; chance 8.3% / 6.5).
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rmr import library  # noqa: E402
from rmr.recipe import LIMITS, check  # noqa: E402

KEYS = [("eR", "earR"), ("eL", "earL"), ("p", "pitch"), ("r", "roll"), ("y", "yaw"), ("z", "z"), ("b", "body")]


def word(name):
    return re.sub(r"\d+$", "", name.split("@")[0])


def transcribe(plan, step=1.0):
    """Plan -> recipe: one ``go step`` per ``step`` seconds to that key's posture, energy = the max in the window."""
    keys, segs = plan["keys"], []
    every = max(1, int(round(step / (keys[1]["t"] - keys[0]["t"])))) if len(keys) > 1 else 1
    for i in range(every, len(keys), every):
        k = keys[i]
        e = max(x["energy"] for x in keys[i - every + 1:i + 1])
        parts = [f"{a}={np.clip(k[c], *LIMITS[a]):.0f}" for a, c in KEYS]
        segs.append(f"go {k['t'] - keys[i - every]['t']:.1f} " + " ".join(parts) + f" E={min(e, 12):.1f}")
    tail = (len(keys) - 1) % every
    if tail:
        segs.append(f"hold {keys[-1]['t'] - keys[len(keys) - 1 - tail]['t']:.1f}")
    return " | ".join(segs)


def build(a):
    from rmr.motion import move_to_traj
    from rmr.plan import extract

    held = {word(h) for h in library.HELD_OUT_EMOTIONS}
    out = []
    for name, move in library.hub(library.EMOTIONS):
        rec = transcribe(extract(move_to_traj(move)), a.step)
        assert check(rec) is None, (name, check(rec))
        out.append({"name": name, "caption": library.caption(name, move), "recipe": rec,
                    "held_out": name in library.HELD_OUT_EMOTIONS, "same_word_as_held_out": word(name) in held})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{len(out)} clips, {sum(not x['same_word_as_held_out'] for x in out)} usable as examples -> {a.out}")


def score(recipes, ckpt, steps=100):
    """``{held-out name: [recipes]}`` -> ranks (1 = identified)."""
    from rmr.planner.evaluate import real_clip_ranks

    ranks = []
    for s in range(max(len(v) for v in recipes.values())):
        r = real_clip_ranks({h: v[s] if s < len(v) else None for h, v in recipes.items()}, ckpt, steps=steps)
        ranks += list(r)
    r = np.array(ranks)
    return {"top1": float((r == 1).mean()), "mean_rank": float(r.mean()), "n": len(r)}


def fidelity(a):
    lib = json.load(open(a.library))
    rec = {x["name"]: [x["recipe"]] for x in lib if x["held_out"]}
    print("transcribed held-out clips through the generator:", score(rec, a.ckpt))


def grounded_system(lib):
    from rmr.planner.prompt import SYSTEM

    ex = "\n".join(f"{x['caption']}\n  {x['recipe']}" for x in lib if not x["same_word_as_held_out"])
    return SYSTEM.replace("Respond with JSON only.", "") + (
        "\n# How Reachy's animator actually moves\nThese recipes are transcribed from Pollen Robotics' hand-animated "
        "emotion clips for this robot (one segment per second). They are the house style: match their sizes, "
        "timing, use of the ears and energy, and how they settle, rather than copying any one of them.\n"
        + ex + "\n\nRespond with JSON only.")


def evaluate(a):
    from rmr.planner import llm
    from rmr.planner.evaluate import held_out_prompts
    from rmr.planner.prompt import SCHEMA, SYSTEM, user_message

    lib = json.load(open(a.library))
    system = SYSTEM if a.plain else grounded_system(lib)
    prompts = held_out_prompts()

    def one(job):
        h, s = job
        for _ in range(3):
            try:
                out = llm.chat_json([{"role": "system", "content": system},
                                     {"role": "user", "content": user_message([prompts[h]])}],
                                    SCHEMA, "motion_recipes", model=a.model, temperature=0.7)
                m = (out.get("motions") or [{}])[0]
                if isinstance(m.get("recipe"), str) and check(m["recipe"]) is None:
                    return h, m["recipe"]
            except Exception as e:
                print(h, s, type(e).__name__, e, flush=True)
        return h, None

    with ThreadPoolExecutor(a.workers) as ex:
        got = list(ex.map(one, [(h, s) for h in prompts for s in range(a.samples)]))
    recipes = {}
    for h, r in got:
        recipes.setdefault(h, []).append(r)
    res = dict(model=a.model, grounded=not a.plain, valid=sum(r is not None for _, r in got) / len(got))
    with open(a.out, "w") as f:                      # recipes first: scoring is the slow part (``score``)
        json.dump({**res, "recipes": recipes}, f, indent=1)
    if not a.no_score:
        rescore(argparse.Namespace(files=[a.out], ckpt=a.ckpt))


def rescore(a):
    """Score (or re-score) saved recipes: adds top1 / mean_rank to each file."""
    import torch

    torch.set_num_threads(max(1, os.cpu_count() // len(a.files)) if not torch.cuda.is_available() else 4)
    for p in a.files:
        d = json.load(open(p))
        d.update(score(d["recipes"], a.ckpt))
        json.dump(d, open(p, "w"), indent=1)
        print(p, json.dumps({k: d[k] for k in ("model", "grounded", "valid", "top1", "mean_rank", "n")}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("library")
    b.add_argument("--out", required=True)
    b.add_argument("--step", type=float, default=1.0)
    f = sub.add_parser("fidelity")
    f.add_argument("--library", required=True)
    f.add_argument("--ckpt", default="checkpoints/generator.pt")
    e = sub.add_parser("eval")
    e.add_argument("--library", required=True)
    e.add_argument("--model", default="moonshotai/Kimi-K3")
    e.add_argument("--samples", type=int, default=4)
    e.add_argument("--plain", action="store_true")
    e.add_argument("--workers", type=int, default=6)
    e.add_argument("--ckpt", default="checkpoints/generator.pt")
    e.add_argument("--out", required=True)
    e.add_argument("--no-score", action="store_true", help="only write the recipes (score them with ``score``)")
    s = sub.add_parser("score")
    s.add_argument("files", nargs="+")
    s.add_argument("--ckpt", default="checkpoints/generator.pt")
    a = ap.parse_args()
    {"library": build, "fidelity": fidelity, "eval": evaluate, "score": rescore}[a.cmd](a)


if __name__ == "__main__":
    main()
