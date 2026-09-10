from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import settings
from app.services.extraction import VehicleExtractionService
from app.services.llm import LLMService
from app.services.stt import STTService

router = APIRouter(prefix='/api/v1', tags=['batch'])

stt_service = STTService()
extraction_service = VehicleExtractionService()
llm_service = LLMService()
INFERENCE_LOCK = asyncio.Lock()


@router.get('/health')
def health() -> dict:
    stt_loaded = stt_service.model is not None
    vad_loaded = (
        not settings.vad_enabled
        or stt_service.vad.model is not None
    )
    ollama_reachable = llm_service.health()

    return {
        'status': (
            'ok'
            if stt_loaded and vad_loaded and ollama_reachable
            else 'degraded'
        ),
        'stt_loaded': stt_loaded,
        'vad_loaded': vad_loaded,
        'ollama_reachable': ollama_reachable,
        'llm_model': settings.llm_model,
        'timing': {
            'enabled': settings.vad_enabled,
            'reference': 'recording_start',
            'unit': 'milliseconds',
            'native_asr_timestamps': False,
        },
    }


@router.post('/process')
async def process_audio(
    file: UploadFile = File(...),
):
    suffix = Path(file.filename or '').suffix.lower() or '.bin'
    upload_root = Path(settings.upload_dir)
    upload_root.mkdir(parents=True, exist_ok=True)

    request_id = uuid.uuid4().hex
    input_path = upload_root / f'{request_id}{suffix}'

    max_bytes = settings.max_upload_mb * 1024 * 1024
    total = 0

    try:
        with input_path.open('wb') as destination:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f'Audio file is larger than '
                            f'{settings.max_upload_mb} MB.'
                        ),
                    )

                destination.write(chunk)

        # Model inference is synchronous/heavy. Running it in a thread keeps
        # the FastAPI event loop responsive for health checks.
        async with INFERENCE_LOCK:
            transcript = await asyncio.to_thread(
                stt_service.transcribe,
                input_path,
            )

            result = await asyncio.to_thread(
                extraction_service.extract,
                transcript.segments,
                str(input_path),
                transcript.duration_ms,
            )

        return result.model_dump()

    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
    finally:
        await file.close()
        try:
            input_path.unlink(missing_ok=True)
            input_path.with_suffix('.16k.wav').unlink(missing_ok=True)
        except OSError:
            pass
