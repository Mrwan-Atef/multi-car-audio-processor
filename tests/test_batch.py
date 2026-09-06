import json
from pathlib import Path

from app.core.config import settings
from app.services.stt import (
    SpeechToTextService,
)
from app.services.extraction import (
    VehicleExtractionService,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

AUDIO_PATH = (
    PROJECT_ROOT
    / "tests"
    / "sample2.aac"
)


def main():

    print("=" * 60)
    print("INITIALIZING SERVICES")
    print("=" * 60)

    print(
        f"Model path: "
        f"{settings.STT_MODEL_PATH}"
    )

    print(
        f"Device: "
        f"{settings.STT_DEVICE}"
    )

    print(
        f"Compute type: "
        f"{settings.STT_COMPUTE_TYPE}"
    )

    print()

    stt = SpeechToTextService()

    extractor = (
        VehicleExtractionService()
    )

    print("=" * 60)
    print("TRANSCRIPT")
    print("=" * 60)

    segments = stt.transcribe(
        str(AUDIO_PATH)
    )

    for segment in segments:

        print(
            f"[{segment.start:.2f}s -> "
            f"{segment.end:.2f}s] "
            f"{segment.text}"
        )

    print()

    print("=" * 60)
    print("FINAL JSON")
    print("=" * 60)

    result = extractor.extract(
        segments=segments,
        audio_path=str(AUDIO_PATH),
    )

    print(
        json.dumps(
            result.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()