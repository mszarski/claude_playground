#!/bin/bash
# Serve the audio responder with vLLM (FP8) on a GPU job and benchmark it like the live server would use it.
#   python - (see docs/results/voice.md): run_job(image="vllm/vllm-openai:v0.30.0", command=["bash","-c",<this>], flavor="l4x1")
# Writes bench_tone.jsonl / bench_lines.jsonl (answers in eval_clean's shape) and the latency summary to
# mszarski/reachy-voice:audio/vllm_bench.
set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
pip install -q soundfile pyarrow scipy "reachy-mini-rust-kinematics>=1.0.3"
hf download mszarski/reachy-planner-sft --repo-type dataset --include "code/*" --local-dir /w
cd /w/code && pip install -q --no-deps -e .
hf download mszarski/reachy-voice --include "audio/student/v1/merged/*" --local-dir /w
hf download mszarski/reachy-voice eval/tone.jsonl eval/lines.jsonl --local-dir /w
hf download mszarski/reachy-voice --include "audio/voiced/eval_lines/*.wav" --local-dir /w
mkdir -p /w/out
vllm serve /w/audio/student/v1/merged --served-model-name reachy-audio --quantization ${QUANT:-fp8} \
    --max-model-len 4096 --gpu-memory-utilization ${GPU_UTIL:-0.6} --limit-mm-per-prompt '{"audio": 1}' \
    --port 8000 > /w/out/vllm.log 2>&1 &
pid=$!
until curl -sf http://127.0.0.1:8000/health >/dev/null; do kill -0 $pid || { tail -50 /w/out/vllm.log; exit 1; }; sleep 2; done
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv | tee /w/out/gpu.txt
python3 scripts/audio_llm.py bench --model "http://127.0.0.1:8000/v1#reachy-audio" --heard /w/eval/tone.jsonl \
    --limit ${LIMIT:-100} --out /w/out/bench_tone.jsonl 2>&1 | tee /w/out/bench.log
python3 scripts/audio_llm.py bench --model "http://127.0.0.1:8000/v1#reachy-audio" --heard /w/eval/lines.jsonl \
    --voiced /w/audio/voiced/eval_lines --limit ${LIMIT:-100} --out /w/out/bench_lines.jsonl 2>&1 | tee -a /w/out/bench.log
python3 - <<'PY'
from huggingface_hub import HfApi
import os
HfApi().upload_folder(folder_path="/w/out", repo_id="mszarski/reachy-voice", path_in_repo=os.environ.get("OUT", "audio/vllm_bench"))
PY
