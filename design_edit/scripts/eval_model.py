# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "transformers>=5.0",
#     "accelerate>=1.0",
#     "datasets>=3.0",
#     "huggingface_hub>=0.34",
#     "vllm>=0.22",
#     "trl==1.17.1",
#     "peft>=0.17",
#     "kernels",
#     "flash-linear-attention",
# ]
# ///
"""Score a Hub model (e.g. the SFT checkpoint) on one or more dataset splits, greedy decoding.

    hf jobs uv run --flavor l40sx1 --timeout 1h --secrets HF_TOKEN scripts/eval_model.py \
        --dataset <user>/design-edits --model <user>/design-edit-sft \
        --splits test,test_paraphrased,test_novel,test_novel_paraphrased

Writes eval_<split>.json (per-example outputs + summary) to the model repo unless --no-push.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    ap.add_argument("--splits", default="test")
    ap.add_argument("--n", type=int, default=None, help="max rows per split")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--vllm", action="store_true", help="generate with vLLM (much faster for large models)")
    ap.add_argument("--no-think", action="store_true", help="enable_thinking=False in the chat template")
    ap.add_argument("--template-kwargs", default=None, help="JSON chat-template kwargs")
    ap.add_argument("--results-repo", default=None,
                    help="upload eval JSONs here (prefixed by the model name) instead of to the model repo, e.g. for a third-party model")
    ap.add_argument("--multimodal", action="store_true",
                    help="transformers path for models that only load as AutoModelForMultimodalLM + AutoProcessor")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--max-model-len", type=int, default=4096, help="vLLM context (raise for reasoning runs)")
    ap.add_argument("--tag", default="", help="subfolder for the result files, e.g. 'think' (keeps runs apart)")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter(s) to merge into --model first, as <repo>/<subfolder>, comma-separated to stack (SFT,RL)")
    args = ap.parse_args()
    bootstrap(args.dataset)

    out_repo, out_prefix, model_path = args.model, (args.tag + "/" if args.tag else ""), args.model
    if args.adapter:
        # Merge on CPU and save locally, so the GPU is free for vLLM afterwards.
        from design_edit.lora import merge_adapter

        model_path = merge_adapter(args.model, args.adapter, "/tmp/merged_model")
        repo, sub = args.adapter.split(",")[-1].strip().rsplit("/", 1)  # results go next to the last adapter
        out_repo, out_prefix = repo, sub + "/" + out_prefix

    import torch
    from datasets import load_dataset
    from huggingface_hub import upload_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from design_edit.evaluate import evaluate, format_report

    tkw = {"enable_thinking": False} if args.no_think else {}
    if args.template_kwargs:
        tkw.update(json.loads(args.template_kwargs))

    if args.vllm:
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # no nvcc on the Jobs image
        from vllm import LLM, SamplingParams

        llm = LLM(model=model_path, dtype="bfloat16", max_model_len=args.max_model_len, gpu_memory_utilization=0.9,
                  limit_mm_per_prompt={"image": 0})
        params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

        def generate(prompts):
            outs = llm.chat(prompts, params, chat_template_kwargs=tkw, use_tqdm=False)
            texts = [o.outputs[0].text for o in outs]
            if not getattr(generate, "_shown", False):
                generate._shown = True
                for t in texts[:2]:
                    print("raw output sample:", repr(t[:200]), flush=True)
            return texts
    elif args.multimodal:
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        proc = AutoProcessor.from_pretrained(model_path)
        tok = getattr(proc, "tokenizer", proc)  # some repos ship only a tokenizer
        tok.padding_side = "left"
        model = AutoModelForMultimodalLM.from_pretrained(model_path, device_map="auto").eval()
        print("loaded", type(model).__name__, "dtype", next(model.parameters()).dtype, flush=True)

        @torch.no_grad()
        def generate(prompts):
            texts = [proc.apply_chat_template(p, tokenize=False, add_generation_prompt=True, **tkw) for p in prompts]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            raw = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=False)
            if not getattr(generate, "_shown", False):
                generate._shown = True
                print("prompt tail:", repr(texts[0][-300:]), flush=True)
                for t in raw[:3]:
                    print("raw output sample:", repr(t[:600]), flush=True)
            return raw  # strip_fences keeps only the answer segment
    else:
        tok = AutoTokenizer.from_pretrained(model_path)
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto").eval()

        @torch.no_grad()
        def generate(prompts):
            texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True, **tkw) for p in prompts]
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            out = model.generate(**enc, max_new_tokens=320, do_sample=False, pad_token_id=tok.pad_token_id)
            return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    overall = {}
    for split in args.splits.split(","):
        ds = load_dataset(args.dataset, split=split)
        if args.n:
            ds = ds.select(range(min(args.n, len(ds))))
        result = evaluate(ds, generate, batch_size=args.batch)
        print(f"\n=== {split} (n={len(ds)}) ===\n{format_report(result)}", flush=True)
        overall[split] = result["overall"]
        path = f"eval_{split}.json"
        Path(path).write_text(json.dumps(result, indent=1))
        if not args.no_push:
            if args.results_repo:
                upload_file(path_or_fileobj=path, repo_id=args.results_repo, repo_type="dataset",
                            path_in_repo=f"{args.model.replace('/', '__')}/{out_prefix}{path}")
            else:
                upload_file(path_or_fileobj=path, path_in_repo=out_prefix + path, repo_id=out_repo)
    print("\nSUMMARY " + json.dumps(overall))
    # vLLM engine processes can keep the job alive after all results are uploaded; exit hard.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
