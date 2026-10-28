#!/usr/bin/env bash
# Evaluate NVIDIA Nemotron-3.5-Transcribe-30B-A4B on marine_vhf_asr test sets (HF Jobs, A100/H100 80 GB).
# Follows the model card: vLLM 0.32 + NeMo Speech main as a vLLM plugin; a separate client env scores.
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=0
cd /tmp
git clone --depth 1 https://github.com/NVIDIA-NeMo/Speech.git nemo-speech
cd nemo-speech
uv venv --python 3.13 .venv
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv sync --extra asr --extra vllm --locked --no-install-project
uv pip install --python .venv/bin/python --no-deps -e .
uv pip install --python .venv/bin/python --no-deps av==19.0.2 opencv-python-headless==5.0.0.93
source .venv/bin/activate
# FlashInfer JIT-compiles kernels with nvcc; use the pip CUDA toolkit, with nvcc pinned to
# the CUDA version of the installed headers/runtime (the lockfile pulls a newer nvcc)
CU=$(python -c "import importlib.metadata as m; print(m.version('nvidia-cuda-runtime').rsplit('.', 1)[0])")
uv pip install --python .venv/bin/python --no-deps "nvidia-cuda-nvcc==$CU.*" "nvidia-cuda-crt==$CU.*" || true
NVCC=$(find "$PWD/.venv" -path "*/bin/nvcc" -type f | head -1)
export CUDA_HOME=$(dirname "$(dirname "$NVCC")")
export PATH="$CUDA_HOME/bin:$PATH"
echo "CUDA_HOME=$CUDA_HOME"; ls "$CUDA_HOME" || true; ls "$CUDA_HOME/include" | head -5 || true
nvcc --version | tail -2 || true
hf download nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B --local-dir /tmp/nemotron
export VLLM_PLUGINS=nemo_speechlm VLLM_MAX_AUDIO_DECODE_DURATION_S=3600 VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180
COMMON=(--served-model-name nemotron --trust-remote-code --dtype bfloat16 --gpu-memory-utilization 0.92
  --max-model-len 32768 --max-num-batched-tokens 4096 --max-num-seqs 64
  --enable-chunked-prefill --no-enable-prefix-caching --mamba-cache-mode none
  --mm-processor-cache-gb 0 --no-async-scheduling
  --limit-mm-per-prompt '{"audio":{"count":1,"length":120000000}}' --reasoning-parser nemotron_v3)
start() {  # start vLLM with extra args; succeed once /health answers
  vllm serve /tmp/nemotron "${COMMON[@]}" "$@" > /tmp/vllm.log 2>&1 &
  PID=$!
  for i in $(seq 1 90); do
    curl -sf http://127.0.0.1:8000/health >/dev/null && return 0
    kill -0 $PID 2>/dev/null || break
    sleep 10
  done
  echo "vLLM config failed: $*"; grep -E "Error|error" /tmp/vllm.log | tail -5
  kill $PID 2>/dev/null || true; sleep 5; return 1
}
# Model card configuration first; fall back to vLLM's default kernels (no JIT) if it fails
start --use-replayssm --mamba-backend flashinfer --speculative-config '{"method":"mtp","num_speculative_tokens":8}' \
  || start || { echo "vLLM failed to start"; tail -60 /tmp/vllm.log; exit 1; }
deactivate
echo "SERVER READY"
curl -sSL -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/$CODE_REPO/resolve/main/marine_vhf_asr/jobs/nemotron/client.py" -o /tmp/client.py
uv run --python 3.12 --with "openai,datasets>=3,soundfile,scipy,numpy,huggingface_hub" python /tmp/client.py
