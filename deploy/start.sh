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
exec python -m rmr.server --port 7860
