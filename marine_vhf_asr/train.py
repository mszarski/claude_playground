"""
LoRA fine-tune Voxtral Mini 3B for marine VHF transcription.

Same recipe as pphilip/voxtral-3B-atc-transcribe (Voxtral-Mini-3B-2507 on a
mix of noisy-radio and clean data), adapted to marine traffic:
- Marine manifests (synthetic from marine_vhf_asr.synthesize and/or real
  recordings in the same JSONL format)
- Optionally mix in jacktol/ATC-ASR-Dataset for real VHF channel noise and
  accented radio English; its transcripts already use our conventions

Fits on a single 24 GB GPU in bf16 with batch size 2-4 and gradient
accumulation.

Usage:
    python -m marine_vhf_asr.train \\
        --train data/synth/manifest.jsonl data/real/train.jsonl \\
        --eval data/real/dev.jsonl \\
        --atc-fraction 0.3 --output runs/voxtral-marine
"""

import argparse
from pathlib import Path
from typing import List

import torch

from .normalize import normalize_transcript

BASE_MODEL = "mistralai/Voxtral-Mini-3B-2507"
SR = 16000

LM_ONLY = r"^(?!.*audio_tower).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
LM_AND_AUDIO = r".*\.(q_proj|k_proj|v_proj|o_proj|out_proj|gate_proj|up_proj|down_proj|fc1|fc2)$"


def load_manifests(paths: List[Path]):
    """Load one or more JSONL manifests; audio paths are relative to each manifest."""
    from datasets import Audio, concatenate_datasets, load_dataset

    parts = []
    for p in paths:
        ds = load_dataset("json", data_files=str(p), split="train")
        root = str(p.parent)
        ds = ds.map(lambda r: {"audio": f"{root}/{r['audio']}"})
        parts.append(ds.select_columns(["audio", "text"]))
    ds = concatenate_datasets(parts)
    return ds.cast_column("audio", Audio(sampling_rate=SR))


def load_atc(n: int, split: str = "train"):
    """n examples from the public ATC radio dataset, re-normalized to our conventions."""
    from datasets import Audio, load_dataset

    ds = load_dataset("jacktol/ATC-ASR-Dataset", split=split).shuffle(seed=0)
    ds = ds.select(range(min(n, len(ds)))).select_columns(["audio", "text"])
    ds = ds.map(lambda r: {"text": normalize_transcript(r["text"])})
    return ds.cast_column("audio", Audio(sampling_rate=SR))


class VoxtralCollator:
    """Builds [audio + transcribe prompt] + [transcript + EOS]; loss only on the transcript."""

    def __init__(self, processor, model_id: str, max_text_tokens: int = 256):
        self.processor = processor
        self.model_id = model_id
        self.tok = processor.tokenizer
        self.max_text_tokens = max_text_tokens

    def __call__(self, features):
        audios = [f["audio"]["array"] for f in features]
        prompt = self.processor.apply_transcription_request(
            language="en",
            audio=audios,
            model_id=self.model_id,
            sampling_rate=SR,
            format=["WAV"] * len(audios),
            return_tensors="pt",
        )
        extra = {k: v for k, v in prompt.items() if k not in ("input_ids", "attention_mask")}

        eos = self.tok.eos_token_id
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else eos
        text_ids = self.tok(
            [f["text"] for f in features], add_special_tokens=False,
            truncation=True, max_length=self.max_text_tokens,
        )["input_ids"]

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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", nargs="+", type=Path, required=True, help="Training JSONL manifests")
    parser.add_argument("--eval", nargs="*", type=Path, default=[], help="Eval JSONL manifests")
    parser.add_argument("--atc-fraction", type=float, default=0.0,
                        help="Add jacktol/ATC-ASR-Dataset examples equal to this fraction of the marine set")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--output", type=Path, default=Path("runs/voxtral-marine"))
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-audio", action="store_true",
                        help="Also adapt the audio encoder (helps with radio channel noise; more memory)")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--merge", action="store_true", help="Also save merged full weights for plain inference")
    args = parser.parse_args(argv)

    from datasets import concatenate_datasets
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoProcessor, Trainer, TrainingArguments,
                              VoxtralForConditionalGeneration)

    train_ds = load_manifests(args.train)
    if args.atc_fraction > 0:
        train_ds = concatenate_datasets([train_ds, load_atc(int(len(train_ds) * args.atc_fraction))])
    train_ds = train_ds.shuffle(seed=0)
    eval_ds = load_manifests(args.eval) if args.eval else None

    processor = AutoProcessor.from_pretrained(args.model)
    model = VoxtralForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
        target_modules=LM_AND_AUDIO if args.lora_audio else LM_ONLY,
    ))
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            num_train_epochs=args.epochs,
            bf16=True,
            logging_steps=20,
            save_steps=args.save_steps,
            eval_steps=args.save_steps if eval_ds is not None else None,
            eval_strategy="steps" if eval_ds is not None else "no",
            save_total_limit=3,
            remove_unused_columns=False,
            dataloader_num_workers=2,
            report_to=[],
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=VoxtralCollator(processor, args.model),
    )
    trainer.train()

    adapter_dir = args.output / "adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    if args.merge:
        merged_dir = args.output / "merged"
        model.merge_and_unload().save_pretrained(merged_dir)
        processor.save_pretrained(merged_dir)
    print(f"Saved adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
