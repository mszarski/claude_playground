"""Planner CLI.

  python -m rmr.planner write  --prompts prompts.txt --out recipes.json    # zero-shot LLM writes recipes
  python -m rmr.planner expand --recipes recipes.json --variants 4 --out plans.jsonl

Reference: ``planner/__main__.py`` in pham-tuan-binh/reachy-motion-generator (Apache-2.0).
"""
import argparse
import json

from . import read_lines, slug


def main():
    ap = argparse.ArgumentParser(prog="python -m rmr.planner", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="zero-shot LLM writes one recipe per prompt")
    src = w.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompts", help="text file, one prompt per line ('word. one sentence.')")
    src.add_argument("--prompt", action="append")
    w.add_argument("--out", required=True)
    w.add_argument("--model")
    w.add_argument("--batch", type=int, default=8)
    w.add_argument("--workers", type=int, default=4)
    e = sub.add_parser("expand", help="recipes -> randomised plans")
    e.add_argument("--recipes", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--variants", type=int, default=4)
    e.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.cmd == "write":
        from .write import write_recipes
        r = write_recipes(read_lines(a.prompts) if a.prompts else a.prompt, a.out, model=a.model, batch=a.batch,
                          workers=a.workers)
        print(f"{len(r)} recipes -> {a.out}")
    else:
        from ..recipe import variants
        recipes, n = json.load(open(a.recipes)), 0
        with open(a.out, "w") as fh:
            for i, (prompt, rec) in enumerate(recipes.items()):
                for v, pl in enumerate(variants(rec, a.variants, seed=a.seed * 100003 + i)):
                    fh.write(json.dumps(dict(name=f"{slug(prompt)}__{v}", prompt=prompt, **pl)) + "\n")
                    n += 1
        print(f"{n} plans from {len(recipes)} recipes -> {a.out}")


if __name__ == "__main__":
    main()
