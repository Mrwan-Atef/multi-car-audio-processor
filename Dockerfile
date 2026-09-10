FROM pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime

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
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Create an isolated environment while keeping access to the
# PyTorch packages already installed in the base image.
RUN python -m venv --system-site-packages /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

ENV PATH="/opt/venv/bin:$PATH"

COPY app ./app
COPY scripts ./scripts

RUN mkdir -p /app/models /tmp/ai-service/uploads

EXPOSE 8000

HEALTHCHECK \
    --interval=15s \
    --timeout=5s \
    --start-period=10m \
    --retries=5 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

# The STT model is downloaded at container startup into the persistent
# /app/models volume, then the API starts.
CMD ["sh", "-c", "python scripts/download_models.py && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]