from app.models.response import Vehicle, VehicleExtractionResponse
from app.models.transcript import TranscriptSegment
from app.services.extraction import VehicleExtractionService
from app.services.plate_candidate import tokenize_segments
from app.services.vad import SpeechRegion, VADService


def test_tokenize_fallback_preserves_absolute_segment_time():
    segment = TranscriptSegment(
        text='ر م ك خمسة سبعة أربعة خمسة',
        start=10.0,
        end=12.0,
        words=[],
    )

    tokens = tokenize_segments([segment])

    assert tokens
    assert tokens[0].start >= 10.0
    assert tokens[-1].end <= 12.0
    assert all(
        left.start <= right.start
        for left, right in zip(tokens, tokens[1:])
    )


def test_timestamp_fields_are_recording_relative(monkeypatch):
    service = VehicleExtractionService()

    monkeypatch.setattr(
        service,
        '_extract_vehicle_type_for_car',
        lambda _plate, _context: {
            'vehicle_type': None,
            'source': None,
        },
    )

    segment = TranscriptSegment(
        text='ب ط ص 5 7 4 5',
        start=10.0,
        end=12.0,
        words=[],
    )

    result = service.extract(
        [segment],
        recording_duration_ms=20000,
    )

    assert len(result.cars) == 1
    car = result.cars[0]
    assert car.plate_number == 'بطص5745'
    assert car.timestamp_start_ms >= 10000
    assert car.timestamp_end_ms <= 12000
    assert car.timestamp_start_ms <= car.timestamp_ms <= car.timestamp_end_ms
    assert result.recording_duration_ms == 20000
    assert result.timestamp_reference == 'recording_start'


def test_vad_merges_only_short_gaps():
    regions = [
        SpeechRegion(0, 16000),
        SpeechRegion(16000 + 1000, 32000),
        SpeechRegion(52000, 64000),
    ]

    merged = VADService._merge_regions(
        regions,
        merge_gap_ms=900,
        total_samples=64000,
        sample_rate=16000,
    )

    assert [(r.start_sample, r.end_sample) for r in merged] == [
        (0, 32000),
        (52000, 64000),
    ]


def test_response_schema_accepts_timestamped_vehicle():
    response = VehicleExtractionResponse(
        recording_duration_ms=30000,
        cars=[
            Vehicle(
                plate_number='بطص5745',
                timestamp_ms=1250,
                timestamp_start_ms=900,
                timestamp_end_ms=1600,
                vehicle_type='نقل',
                notes=None,
            )
        ],
    )

    assert response.cars[0].timestamp_ms == 1250
