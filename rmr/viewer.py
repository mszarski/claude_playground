"""Build the browser viewer's gallery (visualizer/examples/examples.json) from pipeline outputs.

  python -m rmr.viewer runs/demo="demo recipes" runs/e2e="zero-shot Kimi-K3" --out visualizer/examples/examples.json
  python -m http.server -d visualizer 8000      # then open http://localhost:8000

Each argument is a pipeline ``--out`` folder (``recipes.json`` + ``motions/*.json``) or a folder of move files,
optionally ``=label``. Moves are grouped by prompt (their ``description``), so a prompt's variants become tabs.
Numbers are rounded (1e-5 m / rad) to keep the file small; that is far below what the robot can resolve.
"""
import argparse
import glob
import json
import os
from collections import OrderedDict


def _round(x, nd=5):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, list):
        return [_round(v, nd) for v in x]
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    return x


def entries(folder, label=None):
    """``[{prompt, recipe, source, moves}]`` for one pipeline output folder or folder of moves."""
    mdir = os.path.join(folder, "motions") if os.path.isdir(os.path.join(folder, "motions")) else folder
    rec_path = os.path.join(folder, "recipes.json")
    recipes = json.load(open(rec_path)) if os.path.exists(rec_path) else {}
    groups = OrderedDict()
    for p in sorted(glob.glob(os.path.join(mdir, "*.json"))):
        m = json.load(open(p))
        if "set_target_data" not in m:
            continue
        prompt = m.get("description") or os.path.basename(p)[:-5]
        groups.setdefault(prompt, []).append(_round(m))
    source = label or os.path.basename(os.path.normpath(folder))
    return [dict(prompt=p, recipe=recipes.get(p, ""), source=source, moves=ms) for p, ms in groups.items()]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m rmr.viewer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", help="FOLDER or FOLDER=label")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "visualizer", "examples",
                                                  "examples.json"))
    a = ap.parse_args(argv)
    out = []
    for f in a.folders:
        folder, _, label = f.partition("=")
        out += entries(folder, label or None)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    n = sum(len(e["moves"]) for e in out)
    print(f"{len(out)} prompts, {n} moves -> {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
