# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["chatterbox-tts", "datasets>=3,<4", "pyarrow>=15,<18", "setuptools<81", "soundfile", "scipy", "huggingface_hub"]
# ///
"""
Synthesize a text-only manifest (harbor.py output) with Kokoro + the VHF channel
simulator on HF Jobs, and upload wavs + manifest to a Hub dataset folder.

    hf jobs uv run --flavor a10g-small --timeout 2h --secrets HF_TOKEN \\
        --env CODE_REPO=<user>/marine-vhf-asr-code --env DATA_REPO=<user>/marine-vhf-nyc-synth \\
        --env SPLIT=test --env SEED=1 --env VOICE_SET=test --env OUT_SUFFIX=_kv \\
        marine_vhf_asr/jobs/harbor_tts_job.py

TTS=kokoro|chatterbox (Chatterbox needs jobs/harbor_tts_chatterbox_job.py, which
pins its dependencies); VOICE_SET=all|train|test (Kokoro voice subsets).
"""
import os
import sys

from huggingface_hub import hf_hub_download, snapshot_download, upload_folder

sys.path.insert(0, snapshot_download(os.environ["CODE_REPO"]))
from marine_vhf_asr import synthesize  # noqa: E402

repo = os.environ["DATA_REPO"]
tts = os.environ.get("TTS", "chatterbox")
voice_set = os.environ.get("VOICE_SET", "all")
suffix = os.environ.get("OUT_SUFFIX", "")  # e.g. "_kv" -> audio for test_ny goes to test_ny_kv/
for i, split in enumerate(os.environ["SPLIT"].split(",")):  # comma-separated splits, one seed each
    text = hf_hub_download(repo, f"{split}/text.jsonl", repo_type="dataset")
    out = f"/tmp/{split}{suffix}"
    seed = str(int(os.environ.get("SEED", "0")) + i)
    extra = ["--select", os.environ["SELECT"]] if os.environ.get("SELECT") else []
    synthesize.main(["--out", out, "--from-text", text, "--tts", tts, "--voice-set", voice_set, "--seed", seed] + extra)
    upload_folder(repo_id=repo, repo_type="dataset", folder_path=out, path_in_repo=f"{split}{suffix}",
                  commit_message=f"Synthesized {split}{suffix} audio ({tts}, voices: {voice_set})")
    print("UPLOADED", split + suffix, flush=True)
