"""python -m rmr.generator {train,evaluate,sample}

Motion sources (combine freely): --emotions-dir DIR (a local copy of Pollen's emotions library),
--hub (download emotions + dances from the Hugging Face Hub), --dances (procedural dances, no network).
"""
import argparse
import json
import os

from .. import library


def _moves(a):
    moves, held = [], []
    if a.emotions_dir:
        moves += library.local(a.emotions_dir)
        held += library.HELD_OUT_EMOTIONS
    if a.hub:
        moves += library.hub(library.EMOTIONS) + library.hub(library.DANCES)
        held += library.HELD_OUT_EMOTIONS
    if a.dances:
        moves += library.dances()
        if not held:                 # dances only: hold out a few dances instead
            held += library.HELD_OUT_DANCES
    if not moves:
        raise SystemExit("no motion: pass --dances, --emotions-dir DIR and/or --hub")
    return moves, held


def main():
    ap = argparse.ArgumentParser(prog="python -m rmr.generator")
    ap.add_argument("cmd", choices=["train", "evaluate", "sample"])
    ap.add_argument("--emotions-dir")
    ap.add_argument("--hub", action="store_true")
    ap.add_argument("--dances", action="store_true")
    ap.add_argument("--ckpt", default="checkpoints/generator.pt")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--simplify", type=float, default=0.0,
                    help="train: share of samples conditioned on a simplified plan (energy carries the detail)")
    ap.add_argument("--recipes", help="sample: JSON {prompt: recipe}")
    ap.add_argument("--out", default="runs/samples")
    a = ap.parse_args()

    if a.cmd == "train":
        from .train import train
        moves, held = _moves(a)
        train(moves, held, out=a.ckpt, steps=a.steps, bs=a.bs, eval_every=a.eval_every, simplify=a.simplify,
              log=lambda s: print(s, flush=True))
    elif a.cmd == "evaluate":
        from .evaluate import evaluate
        from .sample import load
        net, stats, dev = load(a.ckpt)
        moves, held = _moves(a)
        print(json.dumps(evaluate(net, stats, dev, moves, held, seeds=a.seeds), indent=1))
    else:
        from ..motion import FPS, traj_to_move
        from ..reach import Reach
        from ..recipe import variants
        from .sample import generate_batch, load
        net, stats, dev = load(a.ckpt)
        recipes = json.load(open(a.recipes))
        R = Reach()
        os.makedirs(a.out, exist_ok=True)
        for prompt, recipe in recipes.items():
            plans = variants(recipe, a.seeds)
            for i, A in enumerate(generate_batch(net, stats, plans, dev)):
                move, frac = R.project(traj_to_move(A, prompt))
                name = f"{prompt.split('.')[0].strip().replace(' ', '_')}__{i}"
                with open(os.path.join(a.out, name + ".json"), "w") as f:
                    json.dump(move, f)
                print(f"  {name:40s} {len(A) / FPS:5.1f}s  projected {100 * frac:4.1f}% of frames", flush=True)


if __name__ == "__main__":
    main()
