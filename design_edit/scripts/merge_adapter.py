# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=5.0", "trl==1.17.1", "peft>=0.17", "accelerate>=1.0", "huggingface_hub>=0.34",
#                 "kernels", "flash-linear-attention"]
# ///
"""Merge a saved LoRA checkpoint into its base model and push the result as a full model.

    hf jobs uv run --flavor cpu-performance --timeout 90m --secrets HF_TOKEN scripts/merge_adapter.py \
        --model <user>/design-edit-sft27b --adapter <user>/run-ckpt/checkpoint-150 --output <user>/design-edit-best
"""

import argparse
import os

import torch
from huggingface_hub import HfApi, snapshot_download
from peft import PeftModel
from transformers import AutoTokenizer
from trl.trainer.utils import create_model_from_path

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapter", required=True, help="<repo>/<subfolder>")
ap.add_argument("--output", required=True)
args = ap.parse_args()

repo, sub = args.adapter.rsplit("/", 1)
ckpt = os.path.join(snapshot_download(repo, allow_patterns=[f"{sub}/adapter_*"]), sub)
model = PeftModel.from_pretrained(create_model_from_path(args.model, dtype=torch.bfloat16), ckpt).merge_and_unload()
model.push_to_hub(args.output, private=True)
AutoTokenizer.from_pretrained(args.model).push_to_hub(args.output, private=True)
api = HfApi()
aux = snapshot_download(args.model, allow_patterns=["*processor_config.json", "vocab.json", "merges.txt"])
for f in os.listdir(aux):
    if os.path.isfile(os.path.join(aux, f)):
        api.upload_file(path_or_fileobj=os.path.join(aux, f), path_in_repo=f, repo_id=args.output)
print("pushed", args.output, sorted(f for f in api.list_repo_files(args.output) if not f.startswith("model-")))
