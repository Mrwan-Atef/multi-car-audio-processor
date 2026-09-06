import os
import tempfile

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from app.core.config import settings
from app.services.extraction import (
    VehicleExtractionService,
)
from app.services.stt import (
    SpeechToTextService,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["AI"],
)


# ONE STT INSTANCE
stt_service = SpeechToTextService()

# ONE extraction service
extraction_service = (
    VehicleExtractionService()
)


@router.get("/health")
def health():

    return {
        "status": "ok",
        "service": (
            "multi-car-audio-processor"
        ),
    }


@router.post("/batch/process")
async def process_batch(
    file: UploadFile = File(...),
):

    temp_path: str | None = None

    try:

        suffix = ".wav"

        if file.filename:

            file_suffix = (
                os.path.splitext(
                    file.filename
                )[1]
            )

            if file_suffix:
                suffix = file_suffix

        max_bytes = (
            settings.MAX_AUDIO_SIZE_MB
            * 1024
            * 1024
        )

        total_bytes = 0

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp:

            temp_path = temp.name

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > max_bytes:

                    raise ValueError(
                        "Audio file is too large."
                    )

                temp.write(chunk)

        segments = (
            stt_service.transcribe(
                temp_path
            )
        )

        result = (
            extraction_service.extract(
                segments=segments,
                audio_path=temp_path,
            )
        )

        return result.model_dump()

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        print(
            f"AI processing error: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="AI processing failed.",
        ) from exc

    finally:

        await file.close()

        if (
            temp_path
            and os.path.exists(
                temp_path
            )
        ):

            try:
                os.remove(
                    temp_path
                )
            except OSError:
                pass