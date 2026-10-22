"""
Evaluate a Voxtral checkpoint (base, merged, or LoRA adapter) on a manifest.

Reports corpus WER, plus slot accuracy (vessel names, channels, positions,
MMSI, ...) when the manifest carries slots, and WER per scenario.

Usage:
    python -m marine_vhf_asr.evaluate --manifest data/test/manifest.jsonl \\
        --model mistralai/Voxtral-Mini-3B-2507 --adapter runs/voxtral-marine/adapter
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .metrics import corpus_wer, slot_accuracy
from .train import BASE_MODEL, SR


def transcribe(model, processor, model_id, audios, max_new_tokens=256):
    import torch

    inputs = processor.apply_transcription_request(
        language="en", audio=audios, model_id=model_id, sampling_rate=SR,
        format=["WAV"] * len(audios), return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--adapter", type=Path, help="LoRA adapter directory from train.py")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--predictions", type=Path, help="Write per-example predictions JSONL here")
    args = parser.parse_args(argv)

    import librosa
    import torch
    from transformers import AutoProcessor, VoxtralForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model)
    model = VoxtralForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = [json.loads(line) for line in open(args.manifest)]
    rows = rows[:args.limit] if args.limit else rows
    root = args.manifest.parent

    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        audios = [librosa.load(root / r["audio"], sr=SR)[0] for r in batch]
        for r, hyp in zip(batch, transcribe(model, processor, args.model, audios)):
            r["hyp"] = hyp
        print(f"{min(i + args.batch_size, len(rows))}/{len(rows)}", end="\r")

    print(f"\nWER: {corpus_wer((r['text'], r['hyp']) for r in rows):.2%}  (n={len(rows)})")

    by_scenario = defaultdict(list)
    for r in rows:
        by_scenario[r.get("scenario", "unknown")].append((r["text"], r["hyp"]))
    for name, pairs in sorted(by_scenario.items()):
        print(f"  {name:<20} WER {corpus_wer(pairs):.2%}  (n={len(pairs)})")

    slotted = [(r["slots"], r["hyp"]) for r in rows if r.get("slots")]
    if slotted:
        print("Slot accuracy:")
        for name, acc in sorted(slot_accuracy(slotted).items()):
            print(f"  {name:<20} {acc:.2%}")

    if args.predictions:
        with open(args.predictions, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
