# Live text-to-motion Space: python -m rmr.server. Speech goes to the voice student: the 4-bit GGUF build behind
# llama.cpp's CUDA server (LLAMA_GGUF, ~3x faster than bf16 transformers on the same GPU), or a transformers model
# (VOICE_MODEL) when LLAMA_GGUF is unset. Whisper and the speech-emotion models run on the GPU too; typed prompts use
# the planner on HF Inference Providers.
FROM ghcr.io/ggml-org/llama.cpp:server-cuda
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* && python3 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
RUN useradd -m -u 1000 user || true
WORKDIR /code
COPY --chown=1000 . .
RUN pip install --no-cache-dir -e ".[ik,hf,serve,voice]"
USER 1000
ENV HF_HOME=/tmp/hf PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=2
EXPOSE 7860
ENTRYPOINT []
CMD ["sh", "start.sh"]
