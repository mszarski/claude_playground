#!/bin/bash
# Score a GGUF build of the voice student on a GPU job: llama.cpp's CUDA server + scripts/student_respond.py.
#   hf jobs run --flavor l4x1 --secrets HF_TOKEN -e GGUF=gguf/v8-dpo/reachy-voice-4b-Q4_K_M.gguf -e OUT=gguf/v8-dpo/eval \
#       ghcr.io/ggml-org/llama.cpp:server-cuda bash -c "$(cat scripts/gguf_eval_job.sh)"
# Inputs: the clean evaluation sets (eval/lines.jsonl, eval/tone.jsonl in mszarski/reachy-voice); the code from the
# job-code dataset (deploy/hf_job.py uploads it). Writes eval_{lines,tone}.jsonl and timing to mszarski/reachy-voice:$OUT.
set -euo pipefail
apt-get update -qq && apt-get install -y -qq python3 python3-venv >/dev/null && python3 -m venv /v && . /v/bin/activate
pip install -q numpy scipy "huggingface_hub>=1.0"
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
hf download mszarski/reachy-planner-sft --repo-type dataset --include "code/*" --local-dir /w
hf download mszarski/reachy-voice "$GGUF" eval/lines.jsonl eval/tone.jsonl --local-dir /w
cd /w/code && pip install -q --no-deps -e .
/app/llama-server -m "/w/$GGUF" --host 127.0.0.1 --port 8080 -ngl 99 -c 4096 -np 4 --no-webui > /w/llama.log 2>&1 &
until python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')" 2>/dev/null; do sleep 1; done
mkdir -p /w/out
for s in lines tone; do
  python scripts/student_respond.py --model http://127.0.0.1:8080/v1 --heard /w/eval/$s.jsonl --out /w/out/eval_$s.jsonl --workers 4
done
grep -E "eval time|prompt eval" /w/llama.log | tail -4 > /w/out/timing.txt || true
python - <<'PY'
import os
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path="/w/out", repo_id="mszarski/reachy-voice", path_in_repo=os.environ["OUT"])
print("uploaded", os.environ["OUT"])
PY
