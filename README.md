# Multi-Car Audio Processor — Batch AI Service
The project works with arbic audios to extract some vehicles infos , espicially 3 mean sections :
1- plate num ( 4 nums and 3 arbic latters)
2- dedicated pre-difined attrinutes : (كوستر , ونبد , شاص ...) ---> i saw the use  of LowRA will be more efficient but i faced a technical debit (computing power)
3- some notes related to the car's state 
*look at the architecture file for more info*
This repository contains the isolated AI service used by the .NET backend.

The AI service does **not** connect to SQL Server, Excel, or the .NET business database. It receives an audio recording through HTTP and returns the extracted vehicle JSON.

## Architecture

```text
.NET Backend
     |
     | HTTP multipart/form-data
     v
+---------------------------+
|       ai-service          |
|                           |
|  FastAPI                  |
|    |                      |
|    +--> FFmpeg            |
|    |       |              |
|    |       v              |
|    |   16 kHz mono WAV    |
|    |       |              |
|    |       v              |
|    |   Cohere Arabic STT  |
|    |       |              |
|    |       v              |
|    | deterministic plates |
|    |       |              |
|    |       v              |
|    | bounded vehicle ctx  |
|    |       |              |
|    |       v              |
|    +--> Ollama / Qwen 7B  |
|                           |
+---------------------------+
```

Ollama is a separate internal Compose service. The AI container talks to it through the private Compose network.

## Models

STT:

```text
CohereLabs/cohere-transcribe-arabic-07-2026
```

LLM:

```text
qwen2.5:7b
```

The Cohere model is loaded locally through Transformers. Its official model card supports long-form transcription and automatically chunks long audio before reassembling the transcript.

Do **not** commit model weights to GitHub. They are downloaded into Docker volumes on startup.

## Prerequisites

For the CPU-compatible default Compose file:

- Docker Desktop / Docker Engine
- Docker Compose
- Internet access during the first startup
- Hugging Face access to the Cohere model if your account requires a token

For RTX GPU execution, also install the NVIDIA Container Toolkit on the deployment host. Docker Compose supports GPU device reservations through `deploy.resources.reservations.devices`. citeturn768206search3

## Hugging Face token

The Cohere model is available through Hugging Face. If your account requires authentication/access approval:

1. Accept the model access terms on Hugging Face.
2. Create a Hugging Face read token.
3. Create `.env` from `.env.example`.
4. Set:

```text
HF_TOKEN=your_read_token
```

Never commit `.env`.

## Start the service — CPU/default

The default Compose configuration is intentionally CPU-compatible so the repository can be cloned and started on machines without NVIDIA GPUs. The Docker image itself contains CUDA-enabled PyTorch libraries; the default Compose file simply does not reserve a host GPU.

```bash
docker compose up --build
```

The startup sequence is:

```text
ollama starts
    -> ollama-init pulls qwen2.5:7b
    -> ai-service starts
    -> Cohere model is downloaded into the stt_models volume
    -> FastAPI starts on port 8000
```

The Cohere download uses Hugging Face `snapshot_download(local_dir=...)`; the local directory maintains download metadata so later starts can reuse the existing model instead of blindly downloading it again. citeturn768206search2turn768206search6

## Start on an NVIDIA GPU

On an RTX deployment host:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The GPU override enables one NVIDIA GPU for both Ollama and the AI service.

## API

### Health

```text
GET /api/v1/health
```

Example:

```bash
curl http://localhost:8000/api/v1/health
```

Expected shape:

```json
{
  "status": "ok",
  "stt_loaded": true,
  "ollama_reachable": true,
  "llm_model": "qwen2.5:7b"
}
```

### Process one recording

```text
POST /api/v1/process
Content-Type: multipart/form-data
field: file
```

Linux/macOS:

```bash
curl -X POST http://localhost:8000/api/v1/process \
  -F "file=@tests/sample2.aac"
```

Windows PowerShell:

```powershell
curl.exe -X POST http://localhost:8000/api/v1/process `
  -F "file=@tests/sample2.aac"
```

The endpoint accepts the uploaded file, decodes it with FFmpeg to mono 16 kHz PCM WAV, runs the STT/extraction pipeline, and returns the final JSON.

## Supported audio inputs

FFmpeg performs the input decoding, so the API is not tied to AAC. Common inputs such as AAC, M4A, MP3, WAV, and OGG can be uploaded as long as the installed FFmpeg build can decode them.

The internal STT input is always:

```text
mono
16 kHz
PCM WAV
```

## Important data boundaries

The AI service owns only AI processing.

```text
.NET / SQL Server / Excel
        |
        | HTTP
        v
AI SERVICE
```

The AI service does not write to the business database and does not read Excel files.

The plate number is produced by the deterministic parser. Qwen is used only for semantic vehicle information and safe note cleanup; it is not trusted to invent or modify plates.

## Docker volumes

```text
ollama_models
    -> Qwen model data

stt_models
    -> Cohere STT model

ai_uploads
    -> temporary uploaded audio
```

To completely remove downloaded models and temporary persisted data:

```bash
docker compose down -v
```

Be aware that this deletes the model volumes, so the next startup will download them again.

## GitHub workflow

After pushing this repository:

```bash
git clone <repo-url>
cd multi-car-audio-processor
cp .env.example .env
# set HF_TOKEN if required

docker compose up --build
```

For an RTX host:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

No model files need to be committed to GitHub.

## Project structure

```text
ai-service/
├── app/
│   ├── api/
│   │   └── routes.py
│   ├── core/
│   │   └── config.py
│   ├── models/
│   │   ├── candidates.py
│   │   ├── response.py
│   │   └── transcript.py
│   ├── services/
│   │   ├── audio.py
│   │   ├── extraction.py
│   │   ├── llm.py
│   │   ├── normalizer.py
│   │   ├── plate_candidate.py
│   │   ├── plate_normalizer.py
│   │   └── stt.py
│   └── main.py
├── models/
│   └── .gitkeep
├── scripts/
│   ├── download_models.py
│   └── ollama_pull.sh
├── tests/
│   └── test_api_smoke.py
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.gpu.yml
├── docker-compose.yml
├── Dockerfile
├── README.md
└── requirements.txt
```
