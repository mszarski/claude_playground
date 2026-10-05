"""Score a planner on prompts outside its training data.

  python -m rmr.planner.evaluate --model moonshotai/Kimi-K3 --out runs/planner_eval.json      # zero-shot LLM
  python -m rmr.planner.evaluate --generations runs/4b/generations.json --out runs/4b/eval.json  # fine-tuned

  probes  16 out-of-distribution prompts with a physical check each (``probes.PROBES``), ``--samples`` recipes per
          prompt at temperature 0.7: OOD-core (concepts in no training data) and skill pass rates
  real    its recipes for the 12 held-out real emotions -> generator -> identification among Pollen's real clips
          (top-1 and mean rank; chance 8.3% / 6.5), independent of any teacher's taste
  agree   (fine-tuned only) mean per-descriptor Pearson r between the planner's greedy plans and the teacher's on
          the 39 val prompts, next to the teacher's agreement with itself (its second recipe): the ceiling
  valid   share of answers that parse and pass ``recipe.check`` (zero-shot: after up to 2 fix rounds)

Differences under ~0.1 in probe pass rate, and ~7 points in real-clip top-1, are noise.

Reference: ``planner/distill/evaluate.py`` in pham-tuan-binh/reachy-motion-generator (Apache-2.0), which scores
the fine-tuned planners from vLLM directly. Here the fine-tuned planner's answers come from
``finetune generate`` (run on a GPU), and the zero-shot LLM planner is scored through ``write.write_recipes``.
"""
import argparse
import json
import os
import re

import numpy as np

from .. import library
from ..recipe import check, expand, variants
from . import read_lines
from .probes import PROBES, split
from .probes import score as probe_score

EVAL_PROMPTS = os.path.join(os.path.dirname(__file__), "..", "..", "data", "teacher", "eval_prompts.txt")


DESC_NAMES = ["duration", "ear_mean", "ear_min", "ear_max", "pitch_mean", "pitch_min", "pitch_max",
              "z_mean", "z_max", "z_min", "yaw_range", "roll_range", "energy_mean", "energy_max"]


def plan_descriptors(recipe):
    """Deterministic expansion (no jitter, amp = tempo = 1) -> the descriptor vector named by ``DESC_NAMES``."""
    F = expand(recipe, np.random.default_rng(0))
    T, ears = len(F) / 25, F[:, :2].mean(1)
    return np.array([T, ears.mean(), ears.min(), ears.max(), F[:, 2].mean(), F[:, 2].min(), F[:, 2].max(),
                     F[:, 5].mean(), F[:, 5].max(), F[:, 5].min(), np.ptp(F[:, 4]), np.ptp(F[:, 3]),
                     F[:, 7].mean(), F[:, 7].max()])


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return [float(np.corrcoef(a[:, j], b[:, j])[0, 1]) if a[:, j].std() > 1e-6 and b[:, j].std() > 1e-6
            else float("nan") for j in range(a.shape[1])]


def parse_answer(text):
    """A planner answer -> its recipe if it is JSON with a valid recipe, else ``None``."""
    m = re.search(r"\{.*\}", (text or "").split("</think>")[-1], re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    r = d.get("recipe") if isinstance(d, dict) else None
    return r if isinstance(r, str) and not check(r) else None


def agreement(val_recipes, val_file=None):
    """``(planner vs teacher, teacher vs itself)`` mean descriptor correlation on the val prompts."""
    val = [json.loads(l) for l in open(val_file or os.path.join(os.path.dirname(EVAL_PROMPTS), "val.jsonl"))]
    ok = [(v, val_recipes[v["prompt"]]) for v in val if val_recipes.get(v["prompt"])]
    st = corr([plan_descriptors(r) for _, r in ok], [plan_descriptors(v["labels"][0]["recipe"]) for v, _ in ok])
    tt = corr(*zip(*[(plan_descriptors(v["labels"][0]["recipe"]), plan_descriptors(v["labels"][1]["recipe"]))
                     for v in val]))
    return float(np.nanmean(st)), float(np.nanmean(tt)), dict(zip(DESC_NAMES, st))


def held_out_prompts(path=EVAL_PROMPTS):
    """``{"disgusted1": "disgusted. A movement you use when ..."}`` for the 12 held-out emotions."""
    by_word = {p.split(".")[0]: p for p in read_lines(path)}
    return {h: by_word[h.rstrip("0123456789")] for h in library.HELD_OUT_EMOTIONS}


def real_clip_ranks(held_recipes, ckpt, moves=None, n_variants=5, steps=100):
    """Ranks of the generated motions among the held-out real clips (1 = identified), with the settings of the
    reference's published numbers: 5 training-style (1 Hz) plans per recipe, 100 flow steps."""
    from ..generator.evaluate import HeldOut
    from ..generator.sample import generate, load

    moves = moves or library.hub(library.EMOTIONS)
    H = HeldOut(moves, library.HELD_OUT_EMOTIONS)
    net, stats, dev = load(ckpt)
    ranks = []
    for h, rec in held_recipes.items():
        if rec:
            ranks += [H.rank(generate(net, stats, pl, dev, seed=i, steps=steps), h)
                      for i, pl in enumerate(variants(rec, n_variants, seed=1))]
    return np.array(ranks, float)


def main():
    from .write import write_recipes

    ap = argparse.ArgumentParser(prog="python -m rmr.planner.evaluate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="zero-shot LLM to score")
    ap.add_argument("--generations", help="score a fine-tuned planner's answers from finetune generate instead")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--ckpt", default="checkpoints/generator.pt", help="path or hf://<repo id>/<file>")
    ap.add_argument("--no-real", action="store_true", help="skip the generator-based real-clip check")
    ap.add_argument("--out", help="write the full report as JSON")
    a = ap.parse_args()
    quiet = lambda s: None

    probe_prompts = [p for p, _, _ in PROBES]
    gen = json.load(open(a.generations)) if a.generations else None
    if gen:
        per = {p: [parse_answer(t) for t in gen["probes"].get(p, [])] for p in probe_prompts}
        a.samples = gen["samples"]
    else:
        per = {p: [] for p in probe_prompts}
        for _ in range(a.samples):        # one call per sample: a batch must not repeat a prompt
            got = write_recipes(probe_prompts, model=a.model, log=quiet)
            for p in probe_prompts:
                per[p].append(got.get(p))
    _, per_rate = probe_score(per)
    ood, skill = split(per_rate)
    print(f"probes OOD-core {ood:.2f}  skill {skill:.2f}")
    for p, v in per_rate.items():
        print(f"  {v:4.2f}  {p}")

    report = dict(model=gen["model"] if gen else a.model, ood=ood, skill=skill, per_probe=per_rate, probe_recipes=per)
    n_ok, n = sum(r is not None for rs in per.values() for r in rs), a.samples * len(probe_prompts)
    if gen:
        val_recipes = {p: parse_answer(t) for p, t in gen["val"].items()}
        st, tt, per_desc = agreement(val_recipes)
        print(f"agree {st:.2f} (teacher vs itself {tt:.2f})")
        report.update(agree=st, agree_teacher=tt, agree_per_descriptor=per_desc, val_recipes=val_recipes,
                      seconds_per_answer=gen.get("seconds_per_answer"))
        n_ok, n = n_ok + sum(r is not None for r in val_recipes.values()), n + len(val_recipes)
    if not a.no_real:
        held = held_out_prompts()
        if gen:
            held_recipes = {h: parse_answer(gen["held"].get(h)) for h in held}
        else:
            got = write_recipes(list(held.values()), model=a.model, log=quiet)
            held_recipes = {h: got.get(p) for h, p in held.items()}
        r = real_clip_ranks(held_recipes, a.ckpt)
        r = r if len(r) else np.array([np.nan])
        print(f"real clips top-1 {100 * (r == 1).mean():.0f}%  mean rank {r.mean():.2f}")
        report.update(real_top1=float((r == 1).mean()), real_rank=float(r.mean()), held_recipes=held_recipes)
        n_ok, n = n_ok + sum(x is not None for x in held_recipes.values()), n + len(held)
    report["valid"] = n_ok / n
    print(f"valid {100 * n_ok / n:.1f}%")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
