"""Text prompts -> reachable Reachy Mini moves, end to end (no rendering yet).

  python -m rmr.pipeline --recipes examples/demo_recipes.json --out runs/demo            # no LLM
  python -m rmr.pipeline --prompt "startled. A door slams behind you." --out runs/one     # zero-shot LLM planner
  python -m rmr.pipeline --prompts my_prompts.txt --model deepseek-ai/DeepSeek-V4-Pro --out runs/many
  python -m rmr.pipeline --prompt "sneezing." --planner mszarski/reachy-mini-planner-4b --out runs/ft  # fine-tuned

Stages (each writes into --out):
  1. planner   recipes.json   a recipe per prompt: the zero-shot LLM (rmr.planner.write), a fine-tuned planner
                              (--planner, rmr.planner.finetune; a GPU helps), or --recipes
  2. planner   plans.jsonl    each recipe -> N randomised plans (amplitude, tempo, mirror)
  3. generator motions/       plan -> 25 Hz motion (flow model) -> projected onto the reachable set

Plans are expanded as the reference serves them (``fc=2``, ``kdt=0.25``): fast events stay fast.

Reference: ``pipeline.py`` in pham-tuan-binh/reachy-motion-generator (Apache-2.0).
"""
import argparse
import json
import os

from .planner import read_lines, slug


def main():
    ap = argparse.ArgumentParser(prog="python -m rmr.pipeline", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompts", help="text file, one prompt per line")
    src.add_argument("--prompt", action="append")
    src.add_argument("--recipes", help="skip the planner: a {prompt: recipe} JSON file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", help="zero-shot planner model (default: rmr.planner.llm.DEFAULT_MODEL)")
    ap.add_argument("--planner", help="fine-tuned planner instead: a merged model dir or Hub repo id")
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--ckpt", default="checkpoints/generator.pt", help="path or hf://<repo id>/<file>")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rec_path = os.path.join(a.out, "recipes.json")
    if a.recipes:
        recipes = json.load(open(a.recipes))
        with open(rec_path, "w") as f:
            json.dump(recipes, f, indent=1)
    elif a.planner:
        from .planner.finetune import Planner
        prompts = read_lines(a.prompts) if a.prompts else a.prompt
        print(f"[1/3] planner: {len(prompts)} prompts -> recipes ({a.planner})", flush=True)
        P, recipes = Planner(a.planner), {}
        for p in prompts:
            try:
                recipes[p] = P.plan(p)
            except ValueError as e:
                print(f"  FAILED {p[:60]!r}: {e}")
        with open(rec_path, "w") as f:
            json.dump(recipes, f, indent=1)
    else:
        from .planner.write import write_recipes
        prompts = read_lines(a.prompts) if a.prompts else a.prompt
        print(f"[1/3] planner: {len(prompts)} prompts -> recipes ({a.model or 'default model'})", flush=True)
        recipes = write_recipes(prompts, rec_path, model=a.model)

    from .recipe import variants
    print(f"[2/3] planner: {len(recipes)} recipes x {a.variants} variants -> plans", flush=True)
    plans = []
    for i, (prompt, rec) in enumerate(recipes.items()):
        for v, pl in enumerate(variants(rec, a.variants, seed=i, fc=2.0, kdt=0.25)):
            plans.append(dict(name=f"{slug(prompt)}__{v}", prompt=prompt, **pl))
    with open(os.path.join(a.out, "plans.jsonl"), "w") as fh:
        fh.writelines(json.dumps(p) + "\n" for p in plans)

    from .generator.sample import generate_batch, load
    from .motion import FPS, traj_to_move
    from .reach import Reach
    print(f"[3/3] generator: {len(plans)} plans x {a.seeds} seeds -> motions", flush=True)
    net, stats, dev = load(a.ckpt)
    R = Reach()
    mdir = os.path.join(a.out, "motions")
    os.makedirs(mdir, exist_ok=True)
    for s in range(a.seeds):
        trajs = generate_batch(net, stats, plans, dev, seeds=[s * 7919 + k for k in range(len(plans))], cfg=a.cfg)
        for p, A in zip(plans, trajs):
            move, frac = R.project(traj_to_move(A, p["prompt"]))
            name = p["name"] + (f"_s{s}" if a.seeds > 1 else "")
            with open(os.path.join(mdir, name + ".json"), "w") as f:
                json.dump(move, f)
            print(f"  {name:44s} {len(A) / FPS:5.1f}s  projected {100 * frac:4.1f}% of frames", flush=True)
    print(f"done -> {a.out}")


if __name__ == "__main__":
    main()
