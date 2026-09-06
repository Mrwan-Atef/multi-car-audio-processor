from dataclasses import dataclass

from app.models.candidates import (
    PlateCandidate,
    SuspiciousCandidate,
)

from app.models.transcript import (
    TranscriptSegment,
)

from app.services.plate_normalizer import (
    is_digit_token,
    normalize_letter_token_exact,
    normalize_number_expression,
)


@dataclass
class TranscriptToken:
    text: str
    segment_index: int
    start: float
    end: float


def tokenize_segments(
    segments: list[TranscriptSegment],
) -> list[TranscriptToken]:

    tokens = []

    for segment_index, segment in enumerate(
        segments
    ):

        if segment.words:

            for word in segment.words:

                text = word.text.strip()

                if not text:
                    continue

                tokens.append(
                    TranscriptToken(
                        text=text,
                        segment_index=segment_index,
                        start=word.start,
                        end=word.end,
                    )
                )

        else:

            words = segment.text.split()

            if not words:
                continue

            duration = max(
                segment.end - segment.start,
                0.001,
            )

            word_duration = (
                duration
                / len(words)
            )

            for i, word in enumerate(
                words
            ):

                tokens.append(
                    TranscriptToken(
                        text=word,
                        segment_index=segment_index,
                        start=(
                            segment.start
                            + i * word_duration
                        ),
                        end=(
                            segment.start
                            + (
                                i + 1
                            ) * word_duration
                        ),
                    )
                )

    return tokens


def _build_candidate(
    tokens: list[TranscriptToken],
    start: int,
    end: int,
) -> PlateCandidate:

    selected = tokens[
        start:end
    ]

    return PlateCandidate(
        text=" ".join(
            item.text
            for item in selected
        ),
        start=selected[0].start,
        end=selected[-1].end,
        segment_index=(
            selected[0].segment_index
        ),
        token_start=start,
        token_end=end - 1,
    )


def _digit_span(
    tokens: list[TranscriptToken],
    start: int,
) -> tuple[int, int]:

    raw = [
        token.text
        for token in tokens
    ]

    count = 0
    index = start

    while index < len(tokens):

        value, next_index = (
            normalize_number_expression(
                raw,
                index,
            )
        )

        if value is None:
            break

        count += len(value)

        if count > 4:
            break

        index = next_index

    return count, index


def _exact_letters(
    tokens: list[TranscriptToken],
) -> int:

    count = 0

    for token in tokens:

        count += len(
            normalize_letter_token_exact(
                token.text
            )
        )

    return count


def _find_letter_prefix(
    tokens: list[TranscriptToken],
    digit_start: int,
    max_back_tokens: int = 8,
) -> tuple[int, int] | None:

    lower = max(
        0,
        digit_start - max_back_tokens,
    )

    best = None

    # Search from closest token backwards.
    for start in range(
        digit_start - 1,
        lower - 1,
        -1,
    ):

        phrase = tokens[
            start:digit_start
        ]

        # Any number terminates the letter region.
        if any(
            is_digit_token(
                token.text
            )
            for token in phrase
        ):
            continue

        letter_count = _exact_letters(
            phrase
        )

        # We only accept explicitly recognized letter
        # tokens here.
        if letter_count != 3:
            continue

        score = (
            letter_count * 100
            - len(phrase)
        )

        candidate = (
            score,
            start,
        )

        if (
            best is None
            or candidate[0] > best[0]
        ):
            best = candidate

    if best is None:
        return None

    return (
        best[1],
        digit_start,
    )


def extract_plate_candidates(
    segments: list[TranscriptSegment],
) -> list[PlateCandidate]:

    tokens = tokenize_segments(
        segments
    )

    candidates = []

    index = 0

    while index < len(tokens):

        digit_count, digit_end = (
            _digit_span(
                tokens,
                index,
            )
        )

        if digit_count != 4:
            index += 1
            continue

        result = _find_letter_prefix(
            tokens,
            index,
        )

        if result is None:
            index += 1
            continue

        letter_start, _ = result

        candidates.append(
            _build_candidate(
                tokens,
                letter_start,
                digit_end,
            )
        )

        index = digit_end

    return _deduplicate(
        candidates
    )


def _deduplicate(
    candidates: list[PlateCandidate],
) -> list[PlateCandidate]:

    result = []

    for candidate in candidates:

        duplicate = False

        for previous in reversed(
            result[-3:]
        ):

            if (
                candidate.text
                == previous.text
                and candidate.start
                - previous.end
                <= 2.0
            ):

                duplicate = True
                break

        if not duplicate:
            result.append(
                candidate
            )

    return result


def extract_suspicious_candidates(
    segments: list[TranscriptSegment],
) -> list[SuspiciousCandidate]:

    # Recovery remains a future stage.
    # We intentionally do not manufacture incomplete plates.
    return []