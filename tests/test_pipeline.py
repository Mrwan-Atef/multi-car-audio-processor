from pathlib import Path

from app.services.extraction import CarExtractionService
from app.services.plate_candidate import extract_plate_candidates
from app.services.stt import SpeechToTextService


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIO_PATH = (
    PROJECT_ROOT
    / "tests"
    / "sample.aac"
)


def main() -> None:

    print("=" * 60)
    print("1. SPEECH TO TEXT")
    print("=" * 60)

    stt = SpeechToTextService()

    segments = stt.transcribe(
        str(AUDIO_PATH)
    )

    for index, segment in enumerate(segments):

        print(
            f"[{index}] "
            f"[{segment.start:.2f}s -> "
            f"{segment.end:.2f}s] "
            f"{segment.text}"
        )

    print("\n" + "=" * 60)
    print("2. PLATE CANDIDATES")
    print("=" * 60)

    candidates = extract_plate_candidates(
        segments
    )

    for candidate in candidates:

        print(
            f"[segment={candidate.segment_index}] "
            f"{candidate.text}"
        )

    print("\n" + "=" * 60)
    print("3. VEHICLE EXTRACTION")
    print("=" * 60)

    extractor = CarExtractionService()

    result = extractor.extract(
        segments,
        candidates,
    )

    print("\nFINAL JSON:\n")

    print(
        result.model_dump_json(
            indent=2
        )
    )


if __name__ == "__main__":
    main()