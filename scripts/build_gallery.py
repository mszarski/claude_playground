"""Build the viewer gallery from saved evaluation outputs (no LLM or GPU needed).

  python scripts/build_gallery.py --ckpt hf://mszarski/reachy-motion-generator/generator.pt
  python -m http.server -d visualizer 8000

Inputs: runs/4b/eval.json (rmr.planner.evaluate --generations, fine-tuned 4B) and
runs/planner_eval_Kimi-K3.json (rmr.planner.evaluate --model moonshotai/Kimi-K3). For the 16 probe prompts and the
12 held-out emotions it takes each planner's first valid recipe, generates one motion per recipe as the pipeline
does, and adds Pollen's real clip of each held-out emotion (resampled to 25 Hz) for comparison.
"""
import argparse
import json
import os

from rmr import library
from rmr.generator.sample import generate_batch, load
from rmr.motion import move_to_traj, traj_to_move
from rmr.planner import slug
from rmr.planner.evaluate import held_out_prompts
from rmr.reach import Reach
from rmr.recipe import variants
from rmr.viewer import main as viewer_main


def first_valid(rs):
    return next((r for r in rs if r), None)


def write_folder(out, recipes, net, stats, dev, R):
    os.makedirs(os.path.join(out, "motions"), exist_ok=True)
    with open(os.path.join(out, "recipes.json"), "w") as f:
        json.dump(recipes, f, indent=1)
    prompts = list(recipes)
    plans = [variants(recipes[p], 1, seed=i, fc=2.0, kdt=0.25)[0] for i, p in enumerate(prompts)]
    for p, A in zip(prompts, generate_batch(net, stats, plans, dev, seeds=list(range(len(plans))))):
        move, _ = R.project(traj_to_move(A, p))
        with open(os.path.join(out, "motions", slug(p) + ".json"), "w") as f:
            json.dump(move, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/generator.pt")
    ap.add_argument("--finetuned", default="runs/4b/eval.json")
    ap.add_argument("--zeroshot", default="runs/planner_eval_Kimi-K3.json")
    ap.add_argument("--work", default="runs/gallery")
    ap.add_argument("--out", default="visualizer/examples/examples.json")
    a = ap.parse_args()
    net, stats, dev = load(a.ckpt)
    R = Reach()
    held = held_out_prompts()
    folders = []
    for name, path, label in [("ft", a.finetuned, "fine-tuned Qwen3.5-4B"), ("zs", a.zeroshot, "zero-shot Kimi-K3")]:
        ev = json.load(open(path))
        probes = {p: first_valid(rs) for p, rs in ev["probe_recipes"].items()}
        emotions = {held[h]: r for h, r in ev["held_recipes"].items() if r}
        for part, recipes in [("probes", probes), ("held-out emotions", emotions)]:
            out = os.path.join(a.work, f"{name}_{part.split()[0]}")
            write_folder(out, {p: r for p, r in recipes.items() if r}, net, stats, dev, R)
            folders.append(f"{out}={label}: {part}")
    real = os.path.join(a.work, "real", "motions")
    os.makedirs(real, exist_ok=True)
    clips = dict(library.hub(library.EMOTIONS))
    for h, p in held.items():
        with open(os.path.join(real, h + ".json"), "w") as f:
            json.dump(traj_to_move(move_to_traj(clips[h]), p), f)
    folders.append(f"{os.path.dirname(real)}=Pollen's real clips: held-out emotions")
    viewer_main(folders + ["--out", a.out])


if __name__ == "__main__":
    main()
