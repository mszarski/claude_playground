# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl[vllm]==1.17.1",
#     "datasets>=3.0",
#     "huggingface_hub>=0.34",
# ]
# ///
"""Draw k samples per request with vLLM and measure pass@k plus per-row reward spread.

Answers two questions before spending on RL:
  * pass@k on held-out splits: does sampling *ever* find the right patch? If not, RL can't learn it from rollouts.
  * reward std on candidate training prompts: GRPO only learns from groups whose rewards differ,
    so rows where 0 < n_exact < k are the useful RL prompts.

    hf jobs uv run --flavor l40sx1 --timeout 1h --secrets HF_TOKEN scripts/sample_model.py \
        --dataset <user>/design-edits-v2 --model <user>/design-edit-sft-v2 \
        --splits test_novel,test_novel_paraphrased,grpo_pool --k 8

Writes sample_<split>.json to the model repo (per-row n_exact, n_valid, rewards, outputs).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def bootstrap(dataset_repo: str) -> None:
    try:
        import design_edit  # noqa: F401
        return
    except ImportError:
        pass
    local = Path(__file__).resolve().parents[1]
    if (local / "design_edit").is_dir():
        sys.path.insert(0, str(local))
        return
    from huggingface_hub import snapshot_download

    path = snapshot_download(dataset_repo, repo_type="dataset", allow_patterns=["code/**"])
    sys.path.insert(0, str(Path(path) / "code"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()
    bootstrap(args.dataset)

    from datasets import load_dataset
    from huggingface_hub import upload_file
    from vllm import LLM, SamplingParams

    from design_edit.rewards import score

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=3072, gpu_memory_utilization=0.85)
    params = SamplingParams(n=args.k, temperature=args.temperature, max_tokens=320, seed=0)

    summary = {}
    for split in args.splits.split(","):
        ds = load_dataset(args.dataset, split=split)
        if args.n:
            ds = ds.select(range(min(args.n, len(ds))))
        outs = llm.chat([r["prompt"] for r in ds], params, use_tqdm=True)
        rows, by_kind = [], defaultdict(list)
        for r, o in zip(ds, outs):
            texts = [c.text for c in o.outputs]
            scores = [score(r["before"], r["after"], t, r["relax"]) for t in texts]
            rewards = [s.reward for s in scores]
            row = {"kind": r["kind"], "instruction": r["instruction"],
                   "n_exact": sum(s.exact for s in scores), "n_valid": sum(s.valid for s in scores),
                   "reward_mean": statistics.fmean(rewards), "reward_std": statistics.pstdev(rewards),
                   "outputs": texts, "rewards": rewards}
            rows.append(row)
            by_kind[r["kind"]].append(row)

        def agg(items):
            n = len(items)
            return {"n": n,
                    "pass@k": round(sum(x["n_exact"] > 0 for x in items) / n, 3),
                    "mean_exact": round(sum(x["n_exact"] for x in items) / (n * args.k), 3),
                    "frac_learnable": round(sum(0 < x["n_exact"] < args.k for x in items) / n, 3),
                    "frac_nonzero_std": round(sum(x["reward_std"] > 1e-6 for x in items) / n, 3)}

        summary[split] = {"overall": agg(rows), "by_kind": {k: agg(v) for k, v in sorted(by_kind.items())}}
        print(f"\n=== {split} (n={len(rows)}, k={args.k}, T={args.temperature}) ===")
        print(f"{'kind':<26}{'n':>5}{'pass@k':>8}{'mean':>8}{'0<x<k':>8}{'std>0':>8}")
        for k, v in [("OVERALL", summary[split]["overall"]), *summary[split]["by_kind"].items()]:
            print(f"{k:<26}{v['n']:>5}{v['pass@k']:>8.3f}{v['mean_exact']:>8.3f}{v['frac_learnable']:>8.3f}"
                  f"{v['frac_nonzero_std']:>8.3f}", flush=True)
        path = f"sample_{split}.json"
        Path(path).write_text(json.dumps({"summary": summary[split], "rows": rows}, indent=1))
        if not args.no_push:
            upload_file(path_or_fileobj=path, path_in_repo=path, repo_id=args.model)
    print("\nSUMMARY " + json.dumps({k: v["overall"] for k, v in summary.items()}))


if __name__ == "__main__":
    main()
