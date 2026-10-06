# Live text-to-motion Space: python -m rmr.server on CPU (planner and responder through HF Inference Providers;
# Whisper and the speech-emotion models run in the container).
FROM python:3.11-slim
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN useradd -m -u 1000 user
WORKDIR /app
COPY --chown=user . .
RUN pip install --no-cache-dir -e ".[ik,hf,serve,voice]"
USER user
ENV HF_HOME=/tmp/hf PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=2
EXPOSE 7860
CMD ["python", "-m", "rmr.server", "--port", "7860"]
