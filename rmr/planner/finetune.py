"""LoRA SFT of a planner on one GPU (loss on the assistant answer only), a verified 16-bit merge, and the
generations the evaluation scores.

  python -m rmr.planner.finetune train    --data runs/sft --out runs/4b --model Qwen/Qwen3.5-4B
  python -m rmr.planner.finetune generate --model runs/4b/merged --out runs/4b/generations.json

``train`` keeps the adapter with the best val loss (evaluated every ``--eval-steps``) and writes the merged model to
``<out>/merged``. ``generate`` writes the planner's raw answers for the probes (``--samples`` each at temperature
0.7), the 12 held-out emotions and the 39 val prompts (greedy); ``rmr.planner.evaluate --generations`` scores them
on CPU. Needs the ``planner`` extra and a GPU; runs on Hugging Face Jobs through ``rmr.planner.hfjob``.

Reference: ``planner/distill/train.py`` and ``inference/engine.py`` in pham-tuan-binh/reachy-motion-generator
(Apache-2.0). The reference trains with Unsloth; this port uses plain TRL + PEFT with the same LoRA (rank 32 on
q/k/v/o and the MLPs, so Qwen3.5's Gated DeltaNet projections stay frozen), learning rate, schedule and
best-checkpoint selection, and TRL's prompt/completion format for the answer-only loss.
"""
import argparse
import json
import os
import time

from .prompt import student_messages

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def lm_class(path):
    """The transformers class a checkpoint was saved as. Multimodal checkpoints (...ForConditionalGeneration, e.g.
    Qwen3.5) must NOT be opened as a text-only CausalLM: their weights live under model.language_model.*, so every
    key would miss and be silently re-initialised (or every LoRA key silently fail to apply)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    arch = (AutoConfig.from_pretrained(path).architectures or [""])[0]
    return AutoModelForImageTextToText if "ConditionalGeneration" in arch else AutoModelForCausalLM


def _prompt_completion(row):
    """Chat row -> TRL prompt/completion, rendered without thinking as at inference (``Planner.text``): Qwen3.5's
    template otherwise ends the prompt inside an open ``<think>`` and the answer-only loss mask misaligns."""
    m = row["messages"]
    return {"prompt": m[:-1], "completion": m[-1:], "chat_template_kwargs": {"enable_thinking": False}}


def train(a):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = lm_class(a.model).from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    ds = load_dataset("json", data_files={"train": f"{a.data}/train.jsonl", "val": f"{a.data}/val.jsonl"})
    if a.max_rows:
        ds["train"] = ds["train"].select(range(min(a.max_rows, len(ds["train"]))))
    ds = ds.map(_prompt_completion, remove_columns=ds["train"].column_names)
    print(f"train {len(ds['train'])} rows | val {len(ds['val'])}", flush=True)

    trainer = SFTTrainer(
        model=model, processing_class=tok, train_dataset=ds["train"], eval_dataset=ds["val"],
        peft_config=LoraConfig(r=a.r, lora_alpha=a.r, lora_dropout=0.0, bias="none", target_modules=TARGETS,
                               task_type="CAUSAL_LM"),
        args=SFTConfig(
            output_dir=a.out, max_length=a.max_len, completion_only_loss=True,
            per_device_train_batch_size=a.bs, per_device_eval_batch_size=8, gradient_accumulation_steps=a.grad_accum,
            num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr, lr_scheduler_type="cosine",
            warmup_steps=0.05, weight_decay=0.01, logging_steps=10, eval_strategy="steps", eval_steps=a.eval_steps,
            save_strategy="steps", save_steps=a.eval_steps, save_total_limit=2, load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False, bf16=torch.cuda.is_available(),
            use_cpu=not torch.cuda.is_available(), optim=a.optim,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            seed=0, report_to="none", dataloader_num_workers=2))
    t0 = time.time()
    trainer.train()
    print(f"best checkpoint {trainer.state.best_model_checkpoint} (eval_loss {trainer.state.best_metric:.4f}) "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)
    trainer.model.save_pretrained(f"{a.out}/adapter")
    tok.save_pretrained(f"{a.out}/adapter")
    with open(f"{a.out}/log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f)


def merge(base, adapter, out):
    """Merge the LoRA into a 16-bit model with plain PEFT. Loads the base with the class it was trained as
    (``lm_class``) and refuses to save unless LoRA layers were injected and the weights actually changed."""
    import shutil

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    shutil.rmtree(out, ignore_errors=True)
    m = lm_class(base).from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
    pm = PeftModel.from_pretrained(m, adapter)
    injected = sum(1 for n, _ in pm.named_modules() if n.endswith("lora_A"))
    if injected == 0:
        raise SystemExit(f"merge: no LoRA layers matched {base}; adapter keys do not fit the model")
    name, w = next((n, p) for n, p in pm.named_parameters() if "base_layer.weight" in n)
    before = w.detach().float().clone()
    m = pm.merge_and_unload()
    after = dict(m.named_parameters())[name.replace("base_model.model.", "").replace(".base_layer", "")].detach().float()
    delta = (after - before).abs().max().item()
    if delta == 0:
        raise SystemExit("merge: weights unchanged after merging; refusing to save the base model as the planner")
    m.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
    print(f"merged {injected} LoRA layers (max weight change {delta:.2e}) -> {out}", flush=True)


class Planner:
    """A fine-tuned planner on plain transformers: batched decoding of ``{"idea", "recipe"}`` answers."""

    def __init__(self, path, max_tokens=600):
        import torch
        from transformers import AutoTokenizer

        self.torch, self.max_tokens = torch, max_tokens
        self.tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm = lm_class(path).from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()

    def text(self, prompt):
        return self.tok.apply_chat_template(student_messages(prompt), tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)

    def complete(self, prompts, temperature=0.0, seed=0, batch=32):
        outs = []
        for i in range(0, len(prompts), batch):
            self.torch.manual_seed(seed + i)
            enc = self.tok([self.text(p) for p in prompts[i:i + batch]], return_tensors="pt", padding=True)
            enc = enc.to(self.llm.device)
            kw = dict(do_sample=True, temperature=temperature, top_p=0.95) if temperature else dict(do_sample=False)
            with self.torch.no_grad():
                g = self.llm.generate(**enc, max_new_tokens=self.max_tokens, pad_token_id=self.tok.pad_token_id, **kw)
            outs += self.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return outs


def generate(a):
    from .evaluate import held_out_prompts
    from .probes import PROBES
    from .sft import DATA

    P = Planner(a.model)
    t0 = time.time()
    probe_prompts = [p for p, _, _ in PROBES]
    flat = [p for p in probe_prompts for _ in range(a.samples)]
    raw = P.complete(flat, temperature=0.7, seed=1)
    probes = {p: raw[k * a.samples:(k + 1) * a.samples] for k, p in enumerate(probe_prompts)}
    held = held_out_prompts()
    held_raw = dict(zip(held, P.complete(list(held.values()))))
    val = [json.loads(l)["prompt"] for l in open(os.path.join(DATA, "val.jsonl"))]
    val_raw = dict(zip(val, P.complete(val)))
    n = len(flat) + len(held) + len(val)
    out = dict(model=a.model, samples=a.samples, probes=probes, held=held_raw, val=val_raw,
               seconds_per_answer=(time.time() - t0) / n)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{n} answers in {time.time() - t0:.0f} s -> {a.out}", flush=True)


def main():
    ap = argparse.ArgumentParser(prog="python -m rmr.planner.finetune")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--model", default="Qwen/Qwen3.5-4B")
    t.add_argument("--epochs", type=float, default=2)
    t.add_argument("--max-steps", type=int, default=-1, help="cap the optimiser steps (smoke tests)")
    t.add_argument("--max-rows", type=int, default=0, help="use only the first N training rows (smoke tests)")
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--r", type=int, default=32)
    t.add_argument("--bs", type=int, default=16)
    t.add_argument("--grad-accum", type=int, default=2)
    t.add_argument("--max-len", type=int, default=1536)
    t.add_argument("--eval-steps", type=int, default=40)
    t.add_argument("--optim", default="adamw_8bit", help="adamw_torch without a GPU (bitsandbytes)")
    t.add_argument("--no-merge", action="store_true")
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--samples", type=int, default=12)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
        if not a.no_merge:
            merge(a.model, f"{a.out}/adapter", f"{a.out}/merged")
    else:
        generate(a)


if __name__ == "__main__":
    main()
