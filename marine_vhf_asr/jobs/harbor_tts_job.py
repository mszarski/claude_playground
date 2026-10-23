# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "kokoro>=0.9.4", "soundfile", "scipy", "numpy", "huggingface_hub", "pip",
#     "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
# ]
# ///
"""
Synthesize a text-only manifest (harbor.py output) with Kokoro + the VHF channel
simulator on HF Jobs, and upload wavs + manifest to a Hub dataset folder.

    hf jobs uv run --flavor a10g-small --timeout 2h --secrets HF_TOKEN \\
        --env CODE_REPO=<user>/marine-vhf-asr-code --env DATA_REPO=<user>/marine-vhf-nyc-synth \\
        --env SPLIT=test --env SEED=1 marine_vhf_asr/jobs/harbor_tts_job.py
"""
import os
import sys

from huggingface_hub import hf_hub_download, snapshot_download, upload_folder

sys.path.insert(0, snapshot_download(os.environ["CODE_REPO"]))
from marine_vhf_asr import synthesize  # noqa: E402

repo = os.environ["DATA_REPO"]
for i, split in enumerate(os.environ["SPLIT"].split(",")):  # comma-separated splits, one seed each
    text = hf_hub_download(repo, f"{split}/text.jsonl", repo_type="dataset")
    out = f"/tmp/{split}"
    seed = str(int(os.environ.get("SEED", "0")) + i)
    synthesize.main(["--out", out, "--from-text", text, "--tts", "kokoro", "--seed", seed])
    upload_folder(repo_id=repo, repo_type="dataset", folder_path=out, path_in_repo=split,
                  commit_message=f"Synthesized {split} audio")
    print("UPLOADED", split, flush=True)
