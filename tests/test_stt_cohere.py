from pathlib import Path

from app.core.config import settings
from app.services.stt import (
    STTService,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

AUDIO_PATH = (
    PROJECT_ROOT
    / "tests"
    / "sample4.m4a"
)


def main():

    print("=" * 60)
    print("INITIALIZING COHERE STT")
    print("=" * 60)

    print(
        f"Model path: "
        f"{settings.STT_MODEL_PATH}"
    )

    print(
        f"Device: "
        f"{settings.STT_DEVICE}"
    )

    print()

    stt = STTService()

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


if __name__ == "__main__":
    main()