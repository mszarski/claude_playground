"""
LoRA fine-tune Voxtral Mini 3B for marine VHF transcription.

Same recipe as pphilip/voxtral-3B-atc-transcribe (Voxtral-Mini-3B-2507 on a
mix of noisy-radio and clean data), adapted to marine traffic. Data comes from
sources.py: marine corpora (VHF-8K, MARTTS, our own synthetic manifests) plus
ATC corpora for real VHF channel noise and accented radio English.

Usage:
    python -m marine_vhf_asr.train \\
        --train vhf8k:train#mix vhf8k:discovery#mix martts:train+clean atc:train@4000 atcosim:train@2000 \\
        --eval vhf8k:confirmation#mix@300 \\
        --wer-eval vhf8k:test#none@400 vhf8k:test#supported@400 vhf8k:test#decoys@400 martts:test atc:test@300 \\
        --hub-model-id mszarski/voxtral-3B-marine-vhf
"""

import argparse
import json
from pathlib import Path

import torch

from .evaluate import BASE_MODEL, evaluate_dataset
from .prompting import build_inputs
from .sources import load_audio, load_source, load_sources

LM_ONLY = r"^(?!.*audio_tower).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
LM_AND_AUDIO = r".*\.(q_proj|k_proj|v_proj|o_proj|out_proj|gate_proj|up_proj|down_proj|fc1|fc2)$"


class VoxtralCollator:
    """Builds [audio + optional context + transcribe prompt] + [transcript + EOS]; loss only on the transcript."""

    def __init__(self, processor, model_id: str, max_text_tokens: int = 512):
        self.processor = processor
        self.model_id = model_id
        self.tok = processor.tokenizer
        self.max_text_tokens = max_text_tokens

    def __call__(self, features):
        audios = [load_audio(f["audio"]) for f in features]
        prompt = build_inputs(self.processor, self.model_id, audios, [f.get("context", "") for f in features])
        extra = {k: v for k, v in prompt.items() if k not in ("input_ids", "attention_mask")}

        eos = self.tok.eos_token_id
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else eos
        text_ids = self.tok(
            [f["text"] for f in features], add_special_tokens=False,
            truncation=True, max_length=self.max_text_tokens,
        )["input_ids"]

        # Prompts come back left-padded, so appending the transcript is safe
        ids, attn, labels = [], [], []
        for p_ids, p_att, t_ids in zip(prompt["input_ids"].tolist(),
                                       prompt["attention_mask"].tolist(), text_ids):
            t_ids = t_ids + [eos]
            ids.append(p_ids + t_ids)
            attn.append(p_att + [1] * len(t_ids))
            labels.append([-100] * len(p_ids) + t_ids)

        L = max(len(x) for x in ids)
        pad_to = lambda seq, v: seq + [v] * (L - len(seq))  # noqa: E731
        batch = {
            "input_ids": torch.tensor([pad_to(x, pad) for x in ids]),
            "attention_mask": torch.tensor([pad_to(x, 0) for x in attn]),
            "labels": torch.tensor([pad_to(x, -100) for x in labels]),
        }
        batch.update(extra)
        return batch


def model_card(args, metrics, n_train) -> str:
    pct = lambda m, k: f"{m[k]:.1%}" if k in m else "–"  # noqa: E731
    rows = "\n".join(
        f"| `{spec}` | {m['n']} | {pct(m, 'wer')} | {pct(m, 'slot_accuracy')} | {pct(m, 'decoy_copy_rate')} |"
        for spec, m in metrics.items()
    )
    sources = "\n".join(f"- `{s}`" for s in args.train)
    return f"""---
base_model: {args.model}
library_name: peft
license: apache-2.0
language: [en]
pipeline_tag: automatic-speech-recognition
tags: [maritime, vhf, smcp, voxtral, lora]
---

# Voxtral Mini 3B: marine VHF transcription (LoRA)

LoRA adapter on `{args.model}` for marine VHF radio (VTS, port control,
pilots, coastguard, distress traffic). It is modeled on
[pphilip/voxtral-3B-atc-transcribe](https://huggingface.co/pphilip/voxtral-3B-atc-transcribe).

**Trained mostly on synthetic marine speech.** Validate on real radio traffic
before relying on it, and never use it as the only record of distress traffic.

## Training data ({n_train} clips)
{sources}

## Results (held out)
| Data | n | WER | Vessel names correct | Unspoken candidate copied |
|---|---|---|---|---|
{rows}

`#supported`: the prompt lists the spoken vessel name(s) plus phonetically similar
decoys. `#decoys`: decoys only (the spoken name is absent). `#none`: no list.

Output format: uppercase, no punctuation, numbers spelled out as spoken
(`CHANNEL ONE SIX`).

Optionally pass the vessels known to be nearby (e.g. from AIS) as context;
compare the `#none` and `#supported` rows for the effect.

## Usage
With the `marine_vhf_asr` package (handles the context prompt and output-length cap):
```python
from marine_vhf_asr.evaluate import load_model, transcribe
from marine_vhf_asr.sources import load_audio, prompt_text
model, processor = load_model(adapter="{args.hub_model_id}")
audio = load_audio({{"path": "call.wav", "bytes": None}})
context = prompt_text(["EVER GIVEN", "MAERSK ESSEN", "STENA BRITANNICA"])  # or "" for none
print(transcribe(model, processor, "{args.model}", [audio], [context])[0])
```

Hyperparameters: LoRA r={args.lora_r}, alpha={args.lora_alpha}, audio encoder
adapted={args.lora_audio}, lr={args.lr}, epochs={args.epochs},
effective batch={args.batch_size * args.grad_accum}.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", nargs="+", required=True, help="Training source specs (see sources.py)")
    parser.add_argument("--eval", nargs="*", default=[], help="Source specs for eval loss during training")
    parser.add_argument("--wer-eval", nargs="*", default=[], help="Source specs for WER after training")
    parser.add_argument("--baseline", action="store_true", help="Also score the untrained model on --wer-eval")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--output", type=Path, default=Path("runs/voxtral-marine"))
    parser.add_argument("--hub-model-id", help="Push adapter, results and model card here")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-audio", action="store_true",
                        help="Also adapt the audio encoder (helps with radio channel noise; more memory)")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-steps", type=int, default=-1, help="Override epochs (for smoke tests)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--trackio-project", help="Log to Trackio under this project")
    parser.add_argument("--trackio-space", help="Trackio Space to sync to, e.g. <user>/trackio")
    args = parser.parse_args(argv)

    from peft import LoraConfig, get_peft_model
    from transformers import (AutoProcessor, Trainer, TrainingArguments,
                              VoxtralForConditionalGeneration)

    train_ds = load_sources(args.train).shuffle(seed=0)
    eval_ds = load_sources(args.eval) if args.eval else None
    print(f"Train: {len(train_ds)} clips; eval: {len(eval_ds) if eval_ds else 0}")

    processor = AutoProcessor.from_pretrained(args.model)
    model = VoxtralForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")

    metrics = {}
    if args.baseline:
        for spec in args.wer_eval:
            print(f"Baseline WER on {spec}")
            m, _ = evaluate_dataset(model.eval(), processor, args.model, load_source(spec))
            metrics.update({f"baseline {spec}": v for v in m.values()})
        print(json.dumps(metrics, indent=2))

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
        target_modules=LM_AND_AUDIO if args.lora_audio else LM_ONLY,
    ))
    model.print_trainable_parameters()

    tracking = dict(report_to=[])
    if args.trackio_project:
        tracking = dict(report_to=["trackio"], project=args.trackio_project,
                        trackio_space_id=args.trackio_space, run_name=args.output.name)

    push = dict(push_to_hub=True, hub_model_id=args.hub_model_id, hub_private_repo=True,
                hub_strategy="every_save") if args.hub_model_id else {}
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=0.03,  # <1 means a fraction of total steps
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            bf16=torch.cuda.is_available(),
            logging_steps=10,
            save_steps=args.save_steps,
            eval_steps=args.save_steps if eval_ds is not None else None,
            eval_strategy="steps" if eval_ds is not None else "no",
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_num_workers=4,
            **tracking,
            **push,
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=VoxtralCollator(processor, args.model),
    )
    trainer.train()

    adapter_dir = args.output / "adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    model.eval()
    all_preds = []
    for spec in args.wer_eval:
        print(f"Fine-tuned WER on {spec}")
        m, preds = evaluate_dataset(model, processor, args.model, load_source(spec))
        metrics.update({spec: v for v in m.values()})
        all_preds += preds
    print(json.dumps(metrics, indent=2))

    (adapter_dir / "eval_results.json").write_text(json.dumps(metrics, indent=2))
    with open(adapter_dir / "eval_predictions.jsonl", "w") as f:
        for p in all_preds:
            f.write(json.dumps(p) + "\n")
    finetuned = {k: v for k, v in metrics.items() if not k.startswith("baseline")}
    (adapter_dir / "README.md").write_text(
        model_card(args, metrics if args.baseline else finetuned, len(train_ds)))

    if args.hub_model_id:
        from huggingface_hub import upload_folder
        upload_folder(repo_id=args.hub_model_id, folder_path=str(adapter_dir),
                      commit_message="Final adapter, eval results and model card")
        print(f"Pushed to https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
