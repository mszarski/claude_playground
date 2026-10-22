"""
Evaluate a Voxtral checkpoint (base, merged, or LoRA adapter).

Reports corpus WER per source, slot accuracy (vessel names, channels,
positions, MMSI, ...) where the data carries slots, and - when the prompt
carries candidate vessel names - how often an unspoken candidate is copied.

Usage:
    python -m marine_vhf_asr.evaluate --data vhf8k:test#supported@400 vhf8k:test#decoys@400 \\
        martts:test atc:test@300 --adapter mszarski/voxtral-3B-marine-vhf
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .metrics import corpus_wer, decoy_copy_rate, slot_accuracy
from .prompting import build_inputs, max_new_tokens_for, trim_runaway
from .sources import SR, load_audio, load_source

BASE_MODEL = "mistralai/Voxtral-Mini-3B-2507"


def transcribe(model, processor, model_id, audios, contexts=None):
    """
    Transcribe a batch of 16 kHz clips, optionally with per-clip context
    (e.g. sources.prompt_text(nearby_vessel_names)).

    Output length is capped from each clip's duration so a runaway generation
    cannot flood the transcript.
    """
    import torch

    inputs = build_inputs(processor, model_id, audios, contexts).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens_for(audios), do_sample=False)
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    texts = processor.batch_decode(new_tokens, skip_special_tokens=True)
    return [trim_runaway(t, len(a) / SR) for t, a in zip(texts, audios)]


def evaluate_dataset(model, processor, model_id, ds, batch_size=8):
    """
    Transcribe a sources.py dataset and score it.

    Returns:
        (metrics dict, list of per-example predictions)
    """
    preds = []
    for i in range(0, len(ds), batch_size):
        batch = ds[i:i + batch_size]
        audios = [load_audio(a) for a in batch["audio"]]
        hyps = transcribe(model, processor, model_id, audios, batch["context"])
        for text, slots, context, decoys, source, hyp in zip(
                batch["text"], batch["slots"], batch["context"], batch["decoys"], batch["source"], hyps):
            preds.append({"source": source, "text": text, "hyp": hyp, "slots": json.loads(slots),
                          "context": context, "decoys": json.loads(decoys)})
        print(f"  {min(i + batch_size, len(ds))}/{len(ds)}", end="\r", flush=True)
    print()

    by_source = defaultdict(list)
    for p in preds:
        by_source[p["source"]].append(p)
    metrics = {}
    for source, ps in by_source.items():
        m = {"n": len(ps), "wer": corpus_wer((p["text"], p["hyp"]) for p in ps)}
        slotted = [(p["slots"], p["hyp"]) for p in ps if p["slots"]]
        if slotted:
            m["slot_accuracy"] = slot_accuracy(slotted)["all"]
        if any(p["decoys"] for p in ps):
            m["decoy_copy_rate"] = decoy_copy_rate((p["decoys"], p["text"], p["hyp"]) for p in ps)
        metrics[source] = m
    return metrics, preds


def load_model(model_id=BASE_MODEL, adapter=None):
    import torch
    from transformers import AutoProcessor, VoxtralForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval(), processor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", required=True, help="Source specs (see sources.py)")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--adapter", help="LoRA adapter directory or Hub repo from train.py")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--predictions", type=Path, help="Write per-example predictions JSONL here")
    args = parser.parse_args(argv)

    model, processor = load_model(args.model, args.adapter)
    all_metrics, all_preds = {}, []
    for spec in args.data:
        print(f"Evaluating {spec}")
        metrics, preds = evaluate_dataset(model, processor, args.model, load_source(spec), args.batch_size)
        all_metrics.update({f"{spec}": m for m in metrics.values()})
        all_preds += preds
    print(json.dumps(all_metrics, indent=2))

    if args.predictions:
        with open(args.predictions, "w") as f:
            for p in all_preds:
                f.write(json.dumps(p) + "\n")


if __name__ == "__main__":
    main()
