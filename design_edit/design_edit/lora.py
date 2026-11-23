"""Merge a LoRA adapter stored on the Hub into its base model, locally (CPU), for vLLM or further training."""

from __future__ import annotations

import os
import shutil


def merge_adapter(base: str, adapter: str, out_dir: str = "/tmp/merged_model") -> str:
    """`adapter` is <repo>/<subfolder> (e.g. user/run/adapter), or several separated by commas, merged in that order
    (an RL adapter trained on top of an SFT adapter). Returns the directory of the merged checkpoint."""
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from trl.trainer.utils import create_model_from_path

    merged = create_model_from_path(base, dtype=torch.bfloat16, device_map="cpu")  # "auto" would spread it over the GPUs
    for spec in adapter.split(","):
        repo, sub = spec.strip().rsplit("/", 1)
        ckpt = os.path.join(snapshot_download(repo, allow_patterns=[f"{sub}/adapter_*"]), sub)
        merged = PeftModel.from_pretrained(merged, ckpt).merge_and_unload()
    merged.save_pretrained(out_dir)
    aux = snapshot_download(base, ignore_patterns=["*.safetensors", "eval_*", "sample_*"])
    for f in os.listdir(aux):  # tokenizer, chat template, processor configs
        if os.path.isfile(os.path.join(aux, f)) and not os.path.exists(os.path.join(out_dir, f)):
            shutil.copy(os.path.join(aux, f), out_dir)
    del merged
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # leave the GPUs free for vLLM / training
    print(f"merged {adapter} into {base} -> {out_dir}", flush=True)
    return out_dir
