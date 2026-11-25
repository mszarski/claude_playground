# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "vllm>=0.22",
#     "transformers>=5.0",
#     "datasets>=3.0",
#     "huggingface_hub>=0.34",
# ]
# ///
"""Rejection-sampling data for reasoning SFT: k thinking traces per training request, each scored.

The model reasons in its native thinking mode (Gemma 4: <|channel>thought ... <channel|> answer). Each sample's
patch is scored against the reference (rewards.score, incl. relax/alts); rows with design-system criteria
(eval-v2 style) are graded later (filter_traces.py) because a valid alternative to the reference can still pass.

    hf jobs uv run --flavor h200 --secrets HF_TOKEN scripts/sample_traces.py --dataset user/ds --split train \
        --model google/gemma-4-31B-it --k 4 --out-repo user/design-edit-traces --out-name gemma_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def bootstrap(dataset: str) -> None:
    here = Path(__file__).resolve().parents[1]
    if (here / "design_edit").is_dir():
        sys.path.insert(0, str(here))
        return
    from huggingface_hub import snapshot_download

    root = snapshot_download(dataset, repo_type="dataset", allow_patterns=["code/**"])
    sys.path.insert(0, os.path.join(root, "code"))


HINT = ("\n\n(Private note for this exercise: a correct answer is\n{ref}\nWork the request out yourself and arrive "
        "at an answer of this kind, but reason as if you had not seen this note: never mention it, a hint, a "
        "reference or a given answer.)")


def hinted(row: dict) -> list[dict]:
    ref = row["completion"][0]["content"] if isinstance(row["completion"], list) else row["completion"]
    msgs = [dict(m) for m in row["prompt"]]
    msgs[-1]["content"] += HINT.format(ref=ref)
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter(s) 'repo/sub' merged in order before sampling (comma-separated to stack)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--template-kwargs", default='{"enable_thinking": true}')
    ap.add_argument("--hint", action="store_true",
                    help="reference-guided rationalization (STaR): show the reference answer as a hint; the kept "
                         "trace is later trained on WITHOUT the hint (filter_traces.py --reject-hint-leak)")
    ap.add_argument("--indices", default=None,
                    help="'<dataset repo>:<path>' of a JSON list of row indices to sample (default: all rows)")
    ap.add_argument("--out-repo", required=True)
    ap.add_argument("--out-name", required=True)
    args = ap.parse_args()
    bootstrap(args.dataset)
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    from datasets import load_dataset
    from huggingface_hub import create_repo, upload_file
    from vllm import LLM, SamplingParams

    from design_edit.rewards import score

    ds = load_dataset(args.dataset, split=args.split)
    index = list(range(len(ds)))
    if args.indices:
        from huggingface_hub import hf_hub_download

        repo, path = args.indices.split(":", 1)
        index = json.load(open(hf_hub_download(repo, path, repo_type="dataset")))
    if args.n:
        index = index[:args.n]
    ds = ds.select(index)
    model = args.model
    if args.adapter:
        from design_edit.lora import merge_adapter

        model = merge_adapter(args.model, args.adapter, "/tmp/merged")
    llm = LLM(model=model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.92,
              limit_mm_per_prompt={"image": 0})
    params = SamplingParams(n=args.k, temperature=args.temperature, top_p=0.95, max_tokens=args.max_new_tokens,
                            skip_special_tokens=False, seed=0)
    tkw = json.loads(args.template_kwargs)
    out_path = Path(args.out_name)
    with out_path.open("w") as f:
        for start in range(0, len(ds), 512):
            batch = ds.select(range(start, min(start + 512, len(ds))))
            prompts = [hinted(r) if args.hint else r["prompt"] for r in batch]
            outs = llm.chat(prompts, params, chat_template_kwargs=tkw, use_tqdm=False)
            for j, (r, o) in enumerate(zip(batch, outs)):
                samples = []
                for c in o.outputs:
                    s = score(r["before"], r["after"], c.text, r.get("relax"), r.get("alts"))
                    samples.append({"text": c.text, "finished": c.finish_reason == "stop", "valid": s.valid,
                                    "exact": s.exact, "f1": round(s.f1, 4), "tokens": len(c.token_ids)})
                f.write(json.dumps({"index": index[start + j],
                                    "instruction": r["instruction"], "before": r["before"], "kind": r["kind"],
                                    "samples": samples}) + "\n")
            print(f"{min(start + 512, len(ds))}/{len(ds)} prompts", flush=True)
    create_repo(args.out_repo, repo_type="dataset", private=True, exist_ok=True)
    upload_file(path_or_fileobj=str(out_path), path_in_repo=args.out_name, repo_id=args.out_repo, repo_type="dataset")
    print("uploaded", args.out_repo, args.out_name)
    os._exit(0)


if __name__ == "__main__":
    main()
