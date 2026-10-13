#!/bin/sh
# Space entrypoint: start llama.cpp on the voice student's GGUF (if LLAMA_GGUF = repo:path is set), then the server.
set -e
if [ -n "$LLAMA_GGUF" ]; then
  gguf=$(python -c "import os; from huggingface_hub import hf_hub_download as d; r, p = os.environ['LLAMA_GGUF'].split(':', 1); print(d(r, p))")
  LD_LIBRARY_PATH=/app:${LD_LIBRARY_PATH:-} /app/llama-server -m "$gguf" --host 127.0.0.1 --port 8080 -ngl 99 -c 4096 --no-webui &
  pid=$!
  until python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')" 2>/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "llama-server exited"; exit 1; }; sleep 1
  done
  echo "llama-server ready: $LLAMA_GGUF"
  export VOICE_MODEL="http://127.0.0.1:8080/v1"
fi
if [ -n "$AUDIO_VLLM" ]; then
  model=$(python3 -c "import os; from rmr.planner.finetune import resolve; print(resolve(os.environ['AUDIO_VLLM']))")
  vllm serve "$model" --served-model-name reachy-audio --quantization fp8 --max-model-len 4096 \
      --gpu-memory-utilization ${VLLM_GPU_UTIL:-0.6} --limit-mm-per-prompt '{"audio": 1}' --host 127.0.0.1 --port 8000 &
  pid=$!
  until python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" 2>/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "vllm exited"; exit 1; }; sleep 2
  done
  echo "vllm ready: $AUDIO_VLLM"
  export AUDIO_MODEL="http://127.0.0.1:8000/v1#reachy-audio"
fi
exec python3 -m rmr.server --port 7860
