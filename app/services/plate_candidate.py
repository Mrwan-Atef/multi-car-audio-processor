from dataclasses import dataclass

from app.models.candidates import PlateCandidate, SuspiciousCandidate
from app.models.transcript import TranscriptSegment
from app.services.plate_normalizer import (
    clean_transcript_text,
    decode_letter_phrase_scored,
    is_digit_token,
    normalize_digits_from_tokens,
    normalize_number_expression,
    normalize_plate_from_text,
)


@dataclass
class TranscriptToken:
    text: str
    segment_index: int
    start: float
    end: float


@dataclass(frozen=True)
class _CandidateMeta:
    plate: str
    cost: int
    repeated: bool
    strong: bool


def tokenize_segments(segments: list[TranscriptSegment]) -> list[TranscriptToken]:
    tokens: list[TranscriptToken] = []

    for segment_index, segment in enumerate(segments):
        text = clean_transcript_text(segment.text)
        words = text.split()
        if not words:
            continue

        if segment.words:
            for word in segment.words:
                word_text = clean_transcript_text(word.text)
                if word_text:
                    tokens.append(
                        TranscriptToken(
                            text=word_text,
                            segment_index=segment_index,
                            start=word.start,
                            end=word.end,
                        )
                    )
            continue

        duration = max(segment.end - segment.start, 0.001)
        step = duration / len(words)
        for index, word in enumerate(words):
            tokens.append(
                TranscriptToken(
                    text=word,
                    segment_index=segment_index,
                    start=segment.start + index * step,
                    end=segment.start + (index + 1) * step,
                )
            )

    return tokens


def _build_candidate(tokens: list[TranscriptToken], start: int, end: int) -> PlateCandidate:
    selected = tokens[start:end]
    return PlateCandidate(
        text=" ".join(x.text for x in selected),
        start=selected[0].start,
        end=selected[-1].end,
        segment_index=selected[0].segment_index,
        token_start=start,
        token_end=end - 1,
    )


def _digit_span(tokens: list[TranscriptToken], start: int) -> tuple[int, int]:
    raw = [x.text for x in tokens]
    count = 0
    current = start

    while current < len(tokens):
        value, next_index = normalize_number_expression(raw, current)
        if value is None:
            break
        count += len(value)
        if count > 4:
            break
        current = next_index

    return count, current


def _find_best_prefix(
    tokens: list[TranscriptToken],
    digit_start: int,
    boundary_start: int,
) -> tuple[int, object] | None:
    # A normal plate uses 3 letter units. A duplicated spoken plate uses
    # 3+3 units. Eight tokens leaves room for merged ASR words without letting
    # a vehicle description swallow the plate.
    min_start = max(boundary_start, digit_start - 8)
    best = None

    for start in range(digit_start - 1, min_start - 1, -1):
        phrase = tokens[start:digit_start]
        raw = [x.text for x in phrase]

        if any(is_digit_token(x) for x in raw):
            continue

        decoded = decode_letter_phrase_scored(raw)
        if decoded is None:
            continue

        # A non-repeated candidate may use only light repair. Repeated 3+3
        # candidates can use one additional generic ASR repair.
        if not decoded.repeated and decoded.cost > 2:
            continue
        if decoded.repeated and decoded.cost > 4:
            continue

        # A single weak Arabic word is not enough evidence for a plate.
        if len(raw) == 1 and not decoded.strong:
            continue

        # Prefer a complete repeated 3+3 spelling over a shorter ordinary
        # 3-letter suffix. This is essential for merged ASR such as:
        #   بهي حا بهي حا 1935
        # where the last three tokens alone can also look like a valid plate.
        score = (
            0 if decoded.repeated else 1,
            decoded.cost - (1 if decoded.repeated else 0),
            -int(decoded.strong),
            -len(raw),
            start,
        )

        if best is None or score < best[0]:
            best = (score, start, decoded)

    if best is None:
        return None
    return best[1], best[2]


def _candidate_meta(candidate: PlateCandidate) -> _CandidateMeta | None:
    tokens = candidate.text.split()
    for digit_start in range(1, len(tokens)):
        digits = normalize_digits_from_tokens(tokens[digit_start:])
        if digits is None:
            continue
        decoded = decode_letter_phrase_scored(tokens[:digit_start])
        if decoded is None:
            continue
        return _CandidateMeta(
            plate="".join(decoded.letters) + digits,
            cost=decoded.cost,
            repeated=decoded.repeated,
            strong=decoded.strong,
        )
    return None


def extract_plate_candidates(segments: list[TranscriptSegment]) -> list[PlateCandidate]:
    tokens = tokenize_segments(segments)
    found: list[tuple[PlateCandidate, _CandidateMeta]] = []

    boundary_start = 0
    index = 0

    while index < len(tokens):
        digit_count, digit_end = _digit_span(tokens, index)
        if digit_count != 4:
            index += 1
            continue

        prefix = _find_best_prefix(tokens, index, boundary_start)
        if prefix is None:
            index += 1
            continue

        letter_start, _decoded = prefix
        candidate = _build_candidate(tokens, letter_start, digit_end)
        meta = _candidate_meta(candidate)
        if meta is None or len(meta.plate) != 7 or not meta.plate[-4:].isdigit():
            index += 1
            continue

        found.append((candidate, meta))
        boundary_start = candidate.token_end + 1
        index = digit_end

    return _deduplicate(found)


def _quality(meta: _CandidateMeta) -> tuple:
    return (
        meta.cost - (1 if meta.repeated else 0),
        0 if meta.strong else 1,
        0 if meta.repeated else 1,
    )


def _deduplicate(items: list[tuple[PlateCandidate, _CandidateMeta]]) -> list[PlateCandidate]:
    result: list[tuple[PlateCandidate, _CandidateMeta]] = []

    for candidate, meta in items:
        if not result:
            result.append((candidate, meta))
            continue

        previous, previous_meta = result[-1]
        token_gap = candidate.token_start - previous.token_end - 1

        # Same complete plate spoken twice very close together -> one logical
        # vehicle record.
        if meta.plate == previous_meta.plate and token_gap <= 12:
            continue

        # Generic ASR alternative: same four digits, close together, but one
        # candidate has much stronger letter evidence. Keep the stronger one.
        if meta.plate[-4:] == previous_meta.plate[-4:] and token_gap <= 20:
            if _quality(meta) < _quality(previous_meta):
                result[-1] = (candidate, meta)
            continue

        result.append((candidate, meta))

    return [candidate for candidate, _ in result]


def extract_suspicious_candidates(segments: list[TranscriptSegment]) -> list[SuspiciousCandidate]:
    return []
