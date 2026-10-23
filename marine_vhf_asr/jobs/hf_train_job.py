# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.14.1",
#     "transformers==5.20.2",
#     "peft==0.21.2",
#     "accelerate==1.15.0",
#     "datasets==5.1.0",
#     "huggingface_hub==2.2.0",
#     "mistral-common[audio]==1.12.0",
#     "soundfile==0.14.0",
#     "scipy==1.18.1",
#     "numpy",
#     "trackio==0.43.0",
# ]
# ///
"""
Hugging Face Jobs entry point for marine_vhf_asr.train.

The job container has no checkout of this repository, so the package is
uploaded to a private Hub repo first (see README) and fetched here.

    hf upload <user>/marine-vhf-asr-code marine_vhf_asr marine_vhf_asr --private \\
        --exclude "tests/*" --exclude "__pycache__/*"
    hf jobs uv run --flavor a100-large --timeout 5h --secrets HF_TOKEN \\
        --env CODE_REPO=<user>/marine-vhf-asr-code \\
        marine_vhf_asr/jobs/hf_train_job.py --train ... --hub-model-id <user>/voxtral-3B-marine-vhf

With --env DATA_REPO=<user>/<dataset>, "{DATA}" in arguments is replaced by a
local snapshot of that dataset.
"""

import os
import sys

from huggingface_hub import snapshot_download

code_dir = snapshot_download(os.environ["CODE_REPO"], revision=os.environ.get("CODE_REVISION"))
sys.path.insert(0, code_dir)

from marine_vhf_asr.train import main  # noqa: E402

argv = sys.argv[1:]
if os.environ.get("DATA_REPO"):
    # Source specs may refer to a Hub dataset folder as {DATA}, e.g. {DATA}/train/manifest.jsonl#mix
    data = snapshot_download(os.environ["DATA_REPO"], repo_type="dataset",
                             allow_patterns=os.environ.get("DATA_PATTERNS", "*").split(","))
    argv = [a.replace("{DATA}", data) for a in argv]
main(argv)
