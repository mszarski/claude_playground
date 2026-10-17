#!/bin/bash
# Benchmark a fine-tuned LoRA against its base model inside one Hugging Face Job.
#
# vLLM serves the base model and the adapter (as model "circuit") on localhost,
# and the normal `circuit_gen eval` runs next to it, so nothing is exposed.
# The circuit_gen code comes from code/circuit_gen.tar.gz in $DATASET; results
# are uploaded back to $DATASET under eval/<run>/.
#
#   hf jobs run --flavor l40sx1 --timeout 3h --secrets HF_TOKEN \
#       -e DATASET=user/circuit-gen-sft -e ADAPTER=user/qwen3-8b-circuit-gen \
#       ubuntu:24.04 bash -c "$(cat circuit_gen/train/eval_job.sh)"
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
SAMPLES="${SAMPLES:-2}"
JOBS="${JOBS:-12}"  # parallel samples per model; vLLM batches them on one GPU
# Specs come from the model that wrote the training specs, via Inference Providers,
# so every arm is judged against specs of the quality the adapter trained on.
SPEC_MODEL="${SPEC_MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}"
SPEC_PROVIDER="${SPEC_PROVIDER:-deepinfra}"
RUN="${RUN:-$(date +%Y%m%d-%H%M)}"
export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq --no-install-recommends kicad kicad-symbols kicad-footprints ngspice \
    default-jre-headless python3-venv python3-dev gcc g++ curl ca-certificates > /dev/null
# A venv that still sees the system packages (KiCad's pcbnew module).
python3 -m venv --system-site-packages /venv
export PATH=/venv/bin:$PATH
mkdir -p /opt/freerouting
curl -sSfL -o /opt/freerouting/freerouting.jar \
    https://github.com/freerouting/freerouting/releases/download/v2.1.0/freerouting-2.1.0.jar

pip install -q vllm "huggingface_hub>=1.0"
mkdir -p /work && cd /work
hf download "$DATASET" code/circuit_gen.tar.gz --repo-type dataset --local-dir . > /dev/null
tar xzf code/circuit_gen.tar.gz
hf download "$ADAPTER" --local-dir adapter --exclude "last-checkpoint/*" > /dev/null
pip install -q -r circuit_gen/requirements.txt
pip install -q --no-deps skidl==2.3.0 InSpice==1.7.0.7

# No CUDA toolkit (nvcc) in this image, so skip FlashInfer's JIT-compiled sampler.
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve "$BASE_MODEL" --port 8000 --max-model-len 40960 \
    --enable-lora --max-lora-rank 32 --lora-modules circuit=/work/adapter \
    --enable-auto-tool-choice --tool-call-parser hermes > vllm.log 2>&1 &
until curl -sf localhost:8000/v1/models > /dev/null; do
    kill -0 $! 2> /dev/null || { grep -iE "error|exception|failed" vllm.log | tail -40; exit 1; }
    sleep 10
done
echo "vLLM up: $(curl -s localhost:8000/v1/models)"

export CIRCUIT_GEN_HF_BASE_URL=http://localhost:8000/v1 CIRCUIT_GEN_HF_MAX_TOKENS=8192
upload() {
    hf upload "$DATASET" runs "eval/$RUN" --repo-type dataset --commit-message "eval $RUN" \
        --exclude "*.kicad_pcb" --exclude "*.dsn" --exclude "*.ses" --exclude "gerbers/*" > /dev/null || true
}
# The job's --timeout kills everything at once, so stop the evals in time to
# upload, and upload finished samples (result.json) as they accumulate.
EVAL_MINUTES="${EVAL_MINUTES:-140}"
# Every arm gets its spec from SPEC_MODEL, so the arms differ only in the designer.
# Arms (ARMS, space-separated): base, base_nothink, tuned (adapter, thinking on),
# tuned_nothink. Training sessions render with no think block, which matches the
# thinking-on prompt; thinking off adds an empty <think></think> the adapter never saw.
ARMS="${ARMS:-base tuned}"
eval_one() {  # name model [extra eval args]
    local name=$1 model=$2
    shift 2
    timeout "${EVAL_MINUTES}m" python3 -m circuit_gen eval circuit_gen/benchmarks/boards.json \
        -o "runs/$name" --samples "$SAMPLES" --jobs "$JOBS" --max-turns 25 --spec-first \
        --model "$model" --spec-model "$SPEC_MODEL" --spec-provider "$SPEC_PROVIDER" "$@" \
        > "runs/$name.log" 2>&1 || echo "$name eval stopped (exit $?)"
}
mkdir -p runs
(while sleep 600; do upload; done) &
uploader=$!
pids=()
for arm in $ARMS; do
    case $arm in
        base) eval_one base "$BASE_MODEL" & ;;
        base_nothink) eval_one base_nothink "$BASE_MODEL" --no-thinking & ;;
        tuned) eval_one tuned circuit & ;;
        tuned_nothink) eval_one tuned_nothink circuit --no-thinking & ;;
        *) echo "unknown arm $arm"; exit 1 ;;
    esac
    pids+=($!)
done
wait "${pids[@]}"
kill $uploader
python3 -m circuit_gen compare $(printf "runs/%s " $ARMS) | tee runs/compare.md || true
for arm in $ARMS; do tail -5 "runs/$arm.log"; done
upload
echo "results: https://huggingface.co/datasets/$DATASET/tree/main/eval/$RUN"
