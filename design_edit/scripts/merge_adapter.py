# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=5.0", "trl==1.17.1", "peft>=0.17", "accelerate>=1.0", "huggingface_hub>=0.34",
#                 "kernels", "flash-linear-attention"]
# ///
"""Merge a saved LoRA checkpoint into its base model and push the result as a full model.

Run it on a machine that holds the whole model in memory (a GPU flavor such as h200). The script checks that the
merge changed the weights and that the pushed files differ from the base model's, and fails otherwise.

    hf jobs uv run --flavor h200 --timeout 60m --secrets HF_TOKEN scripts/merge_adapter.py \
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
base = create_model_from_path(args.model, dtype=torch.bfloat16)
offloaded = {str(v) for v in getattr(base, "hf_device_map", {}).values()} & {"disk"}
if offloaded:
    raise SystemExit("base model is partly offloaded to disk; the merge would be a no-op. Use a larger machine.")
peft_model = PeftModel.from_pretrained(base, ckpt)
# Probe the layer whose adapter moved furthest (layers that got no gradient, e.g. a vision tower, have lora_B = 0).
suffix = ".lora_B.default.weight"
probe = max(((p.detach().float().norm().item(), n[: -len(suffix)]) for n, p in peft_model.named_parameters()
             if n.endswith(suffix)))[1]
before = peft_model.get_submodule(probe).weight.detach().float().cpu().clone()
model = peft_model.merge_and_unload()
after = model.get_submodule(probe.removeprefix("base_model.model.")).weight.detach().float().cpu()
changed = (before != after).float().mean().item()
print(f"probe {probe}: {changed:.1%} of weights changed by the merge", flush=True)
if changed == 0:
    raise SystemExit("merge changed nothing; refusing to push a copy of the base model")
model.push_to_hub(args.output, private=True)
AutoTokenizer.from_pretrained(args.model).push_to_hub(args.output, private=True)
api = HfApi()
# Copy processor files only. The cached snapshot also holds the base weights (downloaded when the model was loaded):
# uploading the whole folder would overwrite the merged weights with the base ones.
AUX = ("preprocessor_config.json", "video_preprocessor_config.json", "processor_config.json", "vocab.json", "merges.txt")
aux = snapshot_download(args.model, allow_patterns=list(AUX))
for f in AUX:
    if os.path.isfile(os.path.join(aux, f)):
        api.upload_file(path_or_fileobj=os.path.join(aux, f), path_in_repo=f, repo_id=args.output)
# Check what is on the Hub, not what is in memory: the pushed weights must differ from the base model's.
sha = lambda r: {s.rfilename: s.lfs.sha256 for s in api.model_info(r, files_metadata=True).siblings
                 if s.rfilename.endswith(".safetensors")}
if sha(args.output) == sha(args.model):
    raise SystemExit("pushed weights are identical to the base model")
print("pushed", args.output, sorted(f for f in api.list_repo_files(args.output) if not f.startswith("model-")))
