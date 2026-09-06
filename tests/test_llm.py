from app.models.transcript import TranscriptSegment
from app.services.extraction import CarExtractionService


def main() -> None:
    segments = [
        TranscriptSegment(
            start=0.56,
            end=9.10,
            text="راء قافعين 7665 هافال H6 GT اسود",
        ),
        TranscriptSegment(
            start=9.10,
            end=19.46,
            text="به سين ميم 6469 تويوتا باكاب غمارتين ابيض",
        ),
    ]

    extractor = CarExtractionService()

    result = extractor.extract(segments)

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()