#!/usr/bin/env python3
"""Re-score saved model outputs against the current labels (no GPU needed).

Every eval writes per-row outputs (eval_<split>.json from eval_model.py, or the --out JSON
from eval_baseline.py). When labels change, e.g. alternative answers are added, re-score
those outputs instead of re-running generation.

    python scripts/rescore.py outputs.json data/test_goal.jsonl
    python scripts/rescore.py hf://mszarski/design-edit-sft-v3/eval_test_goal.json data/test_goal.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from design_edit.evaluate import METRICS, format_report, summarize  # noqa: E402
from design_edit.rewards import score  # noqa: E402


def load_outputs(src: str) -> list[dict]:
    if src.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        repo, _, fname = src[len("hf://"):].rpartition("/")
        src = hf_hub_download(repo, fname)
    return json.load(open(src))["rows"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs", help="eval JSON (local path or hf://<repo>/<file>)")
    ap.add_argument("data", help="jsonl with the current labels, same row order")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    outs = load_outputs(args.outputs)
    rows = [json.loads(line) for line in open(args.data)]
    if len(outs) != len(rows):
        sys.exit(f"row count mismatch: {len(outs)} outputs vs {len(rows)} labels")
    per_row = []
    for r, o in zip(rows, outs):
        if o.get("kind") and o["kind"] != r["kind"]:
            sys.exit("row order mismatch between outputs and labels")
        s = score(r["before"], r["after"], o["output"], r["relax"], r.get("alts"))
        per_row.append({"kind": r["kind"], "output": o["output"],
                        **{m: float(getattr(s, m)) for m in METRICS}, "error": s.error})
    result = summarize(per_row)
    print(format_report(result))
    if args.out:
        Path(args.out).write_text(json.dumps(result | {"rows": per_row}, indent=1))


if __name__ == "__main__":
    main()
