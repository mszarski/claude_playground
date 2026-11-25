#!/usr/bin/env python3
"""Rejection sampling: turn sampled thinking traces (sample_traces.py) into reasoning-SFT rows.

For each request, keep the shortest trace that finished and whose patch is correct: an exact match to the reference
(rewards.score, incl. relax/alts), or, for rows with deterministic checks (compile_checks.py), a passing grade, so a
valid alternative to the reference also counts. Requests with no correct trace are dropped (RL can pick them up).

    python scripts/filter_traces.py --dataset user/ds --split train --traces traces.jsonl --out sft_think.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from design_edit.checks import grade_checks  # noqa: E402
from design_edit.schema import strip_fences  # noqa: E402

LEAK = re.compile(r"\bhint\b|private note|\bnote\b says|reference (answer|patch|solution)|given answer|"
                  r"provided answer|correct answer is|the answer (is )?(given|provided)", re.I)
KEEP = ("prompt", "before", "after", "kind", "instruction", "orig_instruction", "relax", "alts", "refs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Hub dataset or local .jsonl the traces were sampled from")
    ap.add_argument("--split", default="train")
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reject-hint-leak", action="store_true",
                    help="for --hint traces: drop any whose text mentions the hint (it is trained without one)")
    ap.add_argument("--max-tokens", type=int, default=2500, help="drop traces longer than this")
    args = ap.parse_args()
    if args.dataset.endswith(".jsonl"):
        rows = [json.loads(line) for line in open(args.dataset)]
    else:
        from datasets import load_dataset

        rows = list(load_dataset(args.dataset, split=args.split))
    out, n_prompts, n_samples, n_correct, lengths = [], 0, 0, 0, []
    for line in open(args.traces):
        t = json.loads(line)
        r = rows[t["index"]]
        assert r["instruction"] == t["instruction"], "traces and dataset rows are out of order"
        n_prompts += 1
        good = []
        for s in t["samples"]:
            n_samples += 1
            ok = s["exact"]
            if not ok and r.get("checks"):
                ok = grade_checks(r["before"], strip_fences(s["text"]), r["checks"], r["allowed"])["pass"]
            if ok:
                n_correct += 1
                if args.reject_hint_leak and LEAK.search(s["text"]):
                    continue
                if s["finished"] and "<channel|>" in s["text"] and s["tokens"] <= args.max_tokens:
                    good.append(s)
        if good:
            best = min(good, key=lambda s: s["tokens"])
            lengths.append(best["tokens"])
            out.append({**{k: r.get(k, "") for k in KEEP},
                        "completion": [{"role": "assistant", "content": best["text"].strip()}]})
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in out))
    lengths.sort()
    print(f"{n_prompts} prompts, sample accuracy {n_correct / max(n_samples, 1):.3f}, "
          f"{len(out)} prompts with a usable correct trace ({len(out) / max(n_prompts, 1):.1%}); "
          f"median kept trace {lengths[len(lengths) // 2] if lengths else 0} tokens -> {args.out}")


if __name__ == "__main__":
    main()
