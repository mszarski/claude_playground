# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=1.0",
#     "transformers>=5.0",
#     "peft>=0.17",
#     "accelerate>=1.0",
#     "datasets>=3.0",
#     "huggingface_hub>=0.34",
#     "trackio",
#     "kernels",
#     "flash-linear-attention",
# ]
# ///
"""Stage 1: supervised fine-tuning on synthetic (design, request) -> patch examples.

Trains a LoRA adapter on prompt/completion pairs (loss on the patch only), merges it,
pushes the merged model to the Hub, then scores greedy outputs on the test split.

Run on Hugging Face Jobs:
    hf jobs uv run --flavor a10g-large --timeout 3h --secrets HF_TOKEN \
        https://huggingface.co/datasets/<user>/design-edits/resolve/main/code/scripts/train_sft.py \
        --dataset <user>/design-edits --output <user>/design-edit-sft
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def bootstrap(dataset_repo: str) -> None:
    """Make ``design_edit`` importable: use the local checkout, or fetch it from the dataset repo's code/ dir."""
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
    ap.add_argument("--output", required=True, help="Hub model id for the merged model")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=2560)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--eval-batch", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--no-push", action="store_true", help="local smoke test: skip Hub uploads")
    ap.add_argument("--report-to", default="trackio")
    ap.add_argument("--no-think", action="store_true",
                    help="train/eval with enable_thinking=False (hybrid reasoning models, e.g. Qwen3.8)")
    args = ap.parse_args()
    bootstrap(args.dataset)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from design_edit.evaluate import evaluate, format_report

    if Path(args.dataset).is_dir():  # local jsonl dir from build_dataset.py
        ds = load_dataset("json", data_files={s: str(Path(args.dataset) / f"{s}.jsonl") for s in ("train", "test")})
    else:
        ds = load_dataset(args.dataset)
    train = ds["train"].shuffle(seed=0)
    if args.max_train:
        train = train.select(range(min(args.max_train, len(train))))
    train = train.select_columns(["prompt", "completion"])
    template_kwargs = {"enable_thinking": False} if args.no_think else {}
    if template_kwargs:
        train = train.add_column("chat_template_kwargs", [template_kwargs] * len(train))

    config = SFTConfig(
        output_dir="sft-out",
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        max_length=args.max_length,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to=args.report_to,
        bf16=torch.cuda.is_available(),
        use_cpu=not torch.cuda.is_available(),
        project="design-edit",
        run_name=f"sft-{args.model.split('/')[-1]}",
        model_init_kwargs={"dtype": torch.bfloat16},
    )
    trainer = SFTTrainer(
        model=args.model,
        train_dataset=train,
        args=config,
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                               target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    trainer.train()

    # Merge LoRA into the base so GRPO / serving can load a plain checkpoint.
    model = trainer.model.merge_and_unload()
    tok = AutoTokenizer.from_pretrained(args.model)
    if not args.no_push:
        model.push_to_hub(args.output, private=True)
        tok.push_to_hub(args.output, private=True)
        print(f"pushed merged model to https://huggingface.co/{args.output}")

    # Quick held-out eval with greedy decoding.
    model.eval()
    model.config.use_cache = True
    trainer.optimizer = None
    trainer.model_wrapped = None
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    tok.padding_side = "left"

    @torch.no_grad()
    def generate(prompts):
        texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True, **template_kwargs)
                 for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    test = ds["test"].select(range(min(args.n_eval, len(ds["test"]))))
    result = evaluate(test, generate, batch_size=args.eval_batch)
    print(format_report(result))
    summary = {k: v for k, v in result.items() if k != "rows"}
    with open("eval_sft.json", "w") as f:
        json.dump(result, f, indent=1)
    from huggingface_hub import upload_file

    if not args.no_push:
        upload_file(path_or_fileobj="eval_sft.json", path_in_repo="eval_sft.json", repo_id=args.output,
                    token=os.environ.get("HF_TOKEN"))
    print(json.dumps(summary["overall"]))


if __name__ == "__main__":
    main()
