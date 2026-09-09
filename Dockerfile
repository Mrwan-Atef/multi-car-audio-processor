FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# FFmpeg is used to decode AAC/M4A/MP3/WAV/OGG and other common formats
# into a single stable input format before STT.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY app ./app
COPY scripts ./scripts

RUN mkdir -p /app/models /tmp/ai-service/uploads

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=5 CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

# The STT model is downloaded at container startup into the persistent
# /app/models volume, then the API starts.
CMD ["sh", "-c", "python scripts/download_models.py && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
