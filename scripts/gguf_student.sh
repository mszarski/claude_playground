#!/bin/bash
# Convert the distilled voice responder to GGUF (bf16 -> Q4_K_M and Q8_0) for llama.cpp / Ollama. Runs in a job:
#   python deploy/hf_job.py --flavor cpu-upgrade --max-usd 0.5 --out-repo mszarski/reachy-voice --out-path gguf/v1 \
#       -- bash scripts/gguf_student.sh mszarski/reachy-voice student/v1/merged
set -euo pipefail
REPO=$1; SUB=$2; BASE=${3:-Qwen/Qwen3-4B-Instruct-2507}   # the base model the student was tuned from
apt-get update -qq && apt-get install -y -qq git cmake build-essential >/dev/null
git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /work/llama.cpp
uv pip install --system -q -r /work/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
hf download "$REPO" --include "$SUB/*" --local-dir /work/model
# The merge was saved by transformers 5, whose tokenizer_config the converter's pinned transformers 4 can't read.
# The LoRA never touched the tokenizer, so take the base model's files.
hf download "$BASE" --include "tokenizer*" "vocab.json" "merges.txt" --local-dir "/work/model/$SUB"
python /work/llama.cpp/convert_hf_to_gguf.py "/work/model/$SUB" --outtype bf16 --outfile /work/bf16.gguf
cmake -S /work/llama.cpp -B /work/llama.cpp/build -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF >/dev/null
cmake --build /work/llama.cpp/build --target llama-quantize -j 8 >/dev/null
for q in Q4_K_M Q8_0; do
  /work/llama.cpp/build/bin/llama-quantize /work/bf16.gguf "/work/out/reachy-voice-4b-$q.gguf" $q | tail -2
done
ls -la /work/out
