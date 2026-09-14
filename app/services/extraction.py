from __future__ import annotations

from typing import Any

from app.schemas.response import (
    Vehicle,
    VehicleExtractionResponse,
)

from app.schemas.transcript import (
    TranscriptSegment,
)

from app.services.llm import LLMService

from app.services.plate_candidate import (
    TranscriptToken,
    extract_plate_candidates,
    tokenize_segments,
)

from app.services.plate_normalizer import (
    decode_token,
    normalize_number_expression,
    normalize_plate_from_text,
)


class VehicleExtractionService:
    """
    Deterministic plate extraction + semantic vehicle enrichment.

    Supported correction commands:

    1. Latest vehicle:
       تعديل اخر لوحه الي NEW_PLATE

    2. Specific vehicle:
       تعديل اللوحة OLD_PLATE إلى NEW_PLATE

    Important:
    - Correction never creates a new vehicle.
    - Correction only changes plate_number.
    - Existing timestamp, vehicle_type, and notes are preserved.
    - Valid plate = exactly 3 Arabic letters + exactly 4 digits.
    - No zero padding.
    """

    def __init__(self) -> None:
        self.llm = LLMService()

    # ========================================================
    # GENERIC HELPERS
    # ========================================================

    @staticmethod
    def _clean(value: Any) -> str | None:
        if not isinstance(value, str):
            return None

        value = " ".join(
            value.strip().split()
        )

        return value or None

    @staticmethod
    def _compact(value: str | None) -> str:
        if not isinstance(value, str):
            return ""

        return "".join(
            value.split()
        )

    @staticmethod
    def _is_modify_command(
        text: str,
    ) -> bool:
        return text.strip() == "تعديل"

    @staticmethod
    def _is_to_word(
        text: str,
    ) -> bool:
        return text.strip() in {
            "إلى",
            "الى",
        }
    @staticmethod
    def _normalize_command_word(text: str) -> str:
        text = text.strip()

        return {
            "اخر": "اخر",
            "آخر": "اخر",

            "لوحه": "لوحه",
            "لوحة": "لوحه",

            "الي": "الى",
            "إلى": "الى",
            "الى": "الى",
        }.get(text, text)

    # ========================================================
    # STRICT PLATE VALIDATION
    # ========================================================

    @staticmethod
    def _is_valid_plate(
        plate: str | None,
    ) -> bool:
        """
        Valid Arabic vehicle plate:

            3 Arabic letters + 4 digits
        """

        if not plate:
            return False

        if len(plate) != 7:
            return False

        if not plate[-4:].isdigit():
            return False

        return True

    @classmethod
    def _normalize_plate_text(
        cls,
        text: str,
    ) -> str | None:

        if not text:
            return None

        plate = normalize_plate_from_text(
            text
        )

        if not cls._is_valid_plate(
            plate
        ):
            return None

        return plate

    # ========================================================
    # LETTER NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_letter(
        token: str,
    ) -> str | None:
        """
        Normalize ASR letter expressions.

        Used as a deterministic decoder for plate-like
        candidate spans.

        Example:
            طه -> ط
        """

        token = token.strip()

        mapping = {
            "ا": "ا",
            "أ": "ا",
            "إ": "ا",
            "آ": "ا",
            "الف": "ا",
            "ألف": "ا",

            "ب": "ب",
            "به": "ب",
            "بي": "ب",
            "باء": "ب",

            "ت": "ت",
            "تي": "ت",
            "تاء": "ت",

            "ث": "ث",
            "ثاء": "ث",

            "ج": "ج",
            "جيم": "ج",

            "ح": "ح",
            "حا": "ح",
            "حاء": "ح",

            "خ": "خ",
            "خا": "خ",
            "خاء": "خ",

            "د": "د",
            "دا": "د",
            "ده": "د",
            "دال": "د",

            "ذ": "ذ",
            "ذال": "ذ",

            "ر": "ر",
            "را": "ر",
            "راء": "ر",

            "ز": "ز",
            "زاي": "ز",

            "س": "س",
            "سي": "س",
            "سين": "س",

            "ش": "ش",
            "شي": "ش",
            "شين": "ش",

            "ص": "ص",
            "صا": "ص",
            "صاد": "ص",

            "ض": "ض",
            "ضا": "ض",
            "ضاد": "ض",

            "ط": "ط",
            "طا": "ط",
            "طاء": "ط",
            "طي": "ط",
            "طه": "ط",

            "ظ": "ظ",
            "ظا": "ظ",
            "ظاء": "ظ",

            "ع": "ع",
            "عا": "ع",
            "عين": "ع",

            "غ": "غ",
            "غا": "غ",
            "غين": "غ",

            "ف": "ف",
            "فا": "ف",
            "فاء": "ف",

            "ق": "ق",
            "قا": "ق",
            "قاف": "ق",

            "ك": "ك",
            "كا": "ك",
            "كاف": "ك",

            "ل": "ل",
            "لا": "ل",
            "لام": "ل",

            "م": "م",
            "مي": "م",
            "ميم": "م",

            "ن": "ن",
            "نو": "ن",
            "نون": "ن",

            "ه": "ه",
            "ها": "ه",
            "هاء": "ه",

            "و": "و",
            "وا": "و",
            "واو": "و",

            "ي": "ي",
            "يا": "ي",
            "ياء": "ي",
        }

        return mapping.get(token)

    # ========================================================
    # DIRECT PLATE DECODING
    # ========================================================

    @classmethod
    def _decode_plate_tokens(
        cls,
        tokens: list[str],
    ) -> str | None:
        """
        Decode a complete plate-like token sequence.

        Expected:

            3 letter units + exactly 4 digits

        This is deliberately deterministic.

        Examples:

            ب طه صاد سبعين ستين
            -> بطص7060

            ح عين ح سبعين حداشر
            -> حعح7011
        """

        if not tokens:
            return None

        letters: list[str] = []
        index = 0

        # ----------------------------------------------------
        # Exactly 3 letters.
        # ----------------------------------------------------

        while (
            index < len(tokens)
            and len(letters) < 3
        ):

            letter = cls._normalize_letter(
                tokens[index]
            )

            if letter is None:
                break

            letters.append(
                letter
            )

            index += 1

        if len(letters) != 3:
            return None

        # ----------------------------------------------------
        # Everything after the letters must form exactly
        # 4 digits.
        # ----------------------------------------------------

        digits: list[str] = []

        while index < len(tokens):

            value, next_index = (
                normalize_number_expression(
                    tokens,
                    index,
                )
            )

            if value is None:
                return None

            digits.extend(
                list(value)
            )

            if len(digits) > 4:
                return None

            index = next_index

        if len(digits) != 4:
            return None

        plate = (
            "".join(letters)
            + "".join(digits)
        )

        if not cls._is_valid_plate(
            plate
        ):
            return None

        return plate

    # ========================================================
    # CANDIDATE REVALIDATION
    # ========================================================

    @classmethod
    def _normalize_candidate(
        cls,
        candidate: Any,
    ) -> str | None:
        """
        Revalidate a candidate using its original token text.

        Direct deterministic decoding gets priority over the
        general fuzzy normalizer.

        This is important for cases such as:

            ح عين ح سبعين حداشر

        where the numeric part becomes valid after supporting
        compound numbers like حداشر.
        """

        if candidate is None:
            return None

        text = getattr(
            candidate,
            "text",
            None,
        )

        if not isinstance(
            text,
            str,
        ):
            return None

        raw_tokens = text.split()

        direct = cls._decode_plate_tokens(
            raw_tokens
        )

        if direct:
            return direct

        return cls._normalize_plate_text(
            text
        )

    # ========================================================
    # CORRECTION PLATE SEARCH
    # ========================================================

    @classmethod
    def _find_correction_plate(
        cls,
        tokens: list[TranscriptToken],
        start: int,
        end: int,
    ) -> tuple[
        str | None,
        int | None,
        int | None,
    ]:
        """
        Search a bounded correction span for one complete plate.

        The search is deterministic:
            3 letter units
            followed by exactly 4 digits
        """

        start = max(
            start,
            0,
        )

        end = min(
            end,
            len(tokens),
        )

        if start >= end:
            return (
                None,
                None,
                None,
            )

        raw = [
            token.text.strip()
            for token in tokens
        ]

        # ----------------------------------------------------
        # Try every possible beginning.
        # ----------------------------------------------------

        for plate_start in range(
            start,
            end,
        ):

            letters: list[str] = []

            index = plate_start

            while (
                index < end
                and len(letters) < 3
            ):

                letter = (
                    cls._normalize_letter(
                        raw[index]
                    )
                )

                if letter is None:
                    break

                letters.append(
                    letter
                )

                index += 1

            if len(letters) != 3:
                continue

            # ------------------------------------------------
            # Parse exactly four digits.
            # ------------------------------------------------

            digits: list[str] = []

            digit_end = index

            while digit_end < end:

                value, next_index = (
                    normalize_number_expression(
                        raw,
                        digit_end,
                    )
                )

                if value is None:
                    break

                digits.extend(
                    list(value)
                )

                if len(digits) > 4:
                    break

                digit_end = next_index

                if len(digits) == 4:
                    break

            if len(digits) != 4:
                continue

            plate = (
                "".join(letters)
                + "".join(digits)
            )

            if cls._is_valid_plate(
                plate
            ):
                return (
                    plate,
                    plate_start,
                    digit_end - 1,
                )

        # ----------------------------------------------------
        # Fallback to the normal normalizer.
        # ----------------------------------------------------

        maximum_window = min(
            10,
            end - start,
        )

        for window_size in range(
            1,
            maximum_window + 1,
        ):

            for window_start in range(
                start,
                end - window_size + 1,
            ):

                window_end = (
                    window_start
                    + window_size
                )

                text = " ".join(
                    token.text
                    for token in tokens[
                        window_start:window_end
                    ]
                )

                plate = (
                    cls._normalize_plate_text(
                        text
                    )
                )

                if plate:
                    return (
                        plate,
                        window_start,
                        window_end - 1,
                    )

        return (
            None,
            None,
            None,
        )

    # ========================================================
    # CORRECTION DETECTION
    # ========================================================

    @classmethod
    def _detect_corrections(
        cls,
        tokens: list[TranscriptToken],
    ) -> list[dict[str, Any]]:
        """
        Supported correction formats:

        1. تعديل اخر لوحه الي NEW

        2. تعديل اللوحة OLD إلى NEW

        No other latest-correction form is supported.
        """

        corrections: list[
            dict[str, Any]
        ] = []

        if not tokens:
            return corrections

        i = 0

        while i < len(tokens):

            if not cls._is_modify_command(
                tokens[i].text
            ):
                i += 1
                continue

            command_index = i

            # =================================================
            # LATEST:
            #
            # تعديل اخر لوحه الي NEW
            # =================================================

            if (
                command_index + 3
                < len(tokens)
            ):

                word1 = word1 = cls._normalize_command_word(
                    tokens[command_index + 1].text
                )

                word2 = cls._normalize_command_word(
                    tokens[command_index + 2].text
                )

                word3 = cls._normalize_command_word(
                    tokens[command_index + 3].text
                )

                if (
                    word1 == "اخر"
                    and word2 == "لوحه"
                    and word3 == "الى"
                ):

                    new_start = (
                        command_index + 4
                    )

                    # Search only a local span.
                    new_end_limit = min(
                        len(tokens),
                        new_start + 12,
                    )

                    # Stop before another command.
                    new_end = new_end_limit

                    k = new_start

                    while k < new_end_limit:

                        if (
                            k > new_start
                            and cls._is_modify_command(
                                tokens[k].text
                            )
                        ):
                            new_end = k
                            break

                        k += 1

                    (
                        new_plate,
                        plate_start,
                        plate_end,
                    ) = cls._find_correction_plate(
                        tokens,
                        new_start,
                        new_end,
                    )

                    if new_plate:

                        corrections.append(
                            {
                                "type": "latest",
                                "command_index":
                                    command_index,
                                "new_plate":
                                    new_plate,
                                "new_start":
                                    plate_start,
                                "new_end":
                                    plate_end,
                                "span_end":
                                    plate_end,
                            }
                        )

                        i = (
                            plate_end + 1
                            if plate_end is not None
                            else command_index + 4
                        )

                        continue

            # =================================================
            # SPECIFIC:
            #
            # تعديل اللوحة OLD إلى NEW
            # =================================================

            to_index: int | None = None

            search_end = min(
                len(tokens),
                command_index + 25,
            )

            j = command_index + 1

            while j < search_end:

                text = (
                    tokens[j]
                    .text
                    .strip()
                )

                # Stop at another command.
                if (
                    j > command_index + 1
                    and cls._is_modify_command(
                        text
                    )
                ):
                    break

                if cls._is_to_word(
                    text
                ):
                    to_index = j
                    break

                j += 1

            if to_index is not None:

                # ------------------------------------------------
                # OLD
                # ------------------------------------------------

                (
                    old_plate,
                    old_start,
                    old_end,
                ) = cls._find_correction_plate(
                    tokens,
                    command_index + 1,
                    to_index,
                )

                if old_plate is None:
                    i += 1
                    continue

                # ------------------------------------------------
                # NEW
                # ------------------------------------------------

                new_search_end = min(
                    len(tokens),
                    to_index + 15,
                )

                k = to_index + 1

                while k < new_search_end:

                    if (
                        k > to_index + 1
                        and cls._is_modify_command(
                            tokens[k].text
                        )
                    ):
                        new_search_end = k
                        break

                    k += 1

                (
                    new_plate,
                    new_start,
                    new_end,
                ) = cls._find_correction_plate(
                    tokens,
                    to_index + 1,
                    new_search_end,
                )

                if new_plate is None:
                    i += 1
                    continue

                corrections.append(
                    {
                        "type": "specific",
                        "command_index":
                            command_index,
                        "old_plate":
                            old_plate,
                        "new_plate":
                            new_plate,
                        "old_start":
                            old_start,
                        "old_end":
                            old_end,
                        "new_start":
                            new_start,
                        "new_end":
                            new_end,
                        "span_end":
                            new_end,
                    }
                )

                i = (
                    new_end + 1
                    if new_end is not None
                    else command_index + 1
                )

                continue

            i += 1

        return corrections

    # ========================================================
    # CORRECTION TOKEN SPANS
    # ========================================================

    @staticmethod
    def _correction_token_indexes(
        corrections: list[dict[str, Any]],
    ) -> set[int]:

        indexes: set[int] = set()

        for correction in corrections:

            command_index = correction.get(
                "command_index"
            )

            span_end = correction.get(
                "span_end"
            )

            if (
                isinstance(
                    command_index,
                    int,
                )
                and isinstance(
                    span_end,
                    int,
                )
                and span_end >= command_index
            ):

                indexes.update(
                    range(
                        command_index,
                        span_end + 1,
                    )
                )

        return indexes

    @staticmethod
    def _candidate_overlaps_correction(
        candidate: Any,
        correction_indexes: set[int],
    ) -> bool:

        if not correction_indexes:
            return False

        start = getattr(
            candidate,
            "token_start",
            None,
        )

        end = getattr(
            candidate,
            "token_end",
            None,
        )

        if (
            not isinstance(start, int)
            or not isinstance(end, int)
        ):
            return False

        for index in range(
            start,
            end + 1,
        ):

            if index in correction_indexes:
                return True

        return False

    # ========================================================
    # NUMBER / PLATE-LIKE FRAGMENTS
    # ========================================================

    @staticmethod
    def _number_run_end(
        tokens: list[TranscriptToken],
        start: int,
    ) -> tuple[int, int]:

        raw = [
            token.text
            for token in tokens
        ]

        index = start
        digit_count = 0

        while index < len(tokens):

            value, next_index = (
                normalize_number_expression(
                    raw,
                    index,
                )
            )

            if value is None:
                break

            digit_count += len(value)

            if digit_count > 4:
                break

            index = next_index

        return (
            digit_count,
            index,
        )

    @staticmethod
    def _token_has_letter_evidence(
        token: str,
    ) -> bool:
        return bool(
            decode_token(
                token,
                max_letters=3,
            )
        )

    @classmethod
    def _remove_plate_like_fragments(
        cls,
        tokens: list[TranscriptToken],
    ) -> list[TranscriptToken]:

        if not tokens:
            return []

        remove: set[int] = set()

        index = 0

        while index < len(tokens):

            (
                digit_count,
                digit_end,
            ) = cls._number_run_end(
                tokens,
                index,
            )

            if digit_count != 4:
                index += 1
                continue

            start = index - 1
            letter_count = 0

            while (
                start >= 0
                and index - start <= 6
            ):

                if not cls._token_has_letter_evidence(
                    tokens[start].text
                ):
                    break

                letter_count += 1
                start -= 1

            actual_start = (
                start + 1
            )

            if (
                letter_count >= 2
                and actual_start < index
            ):

                for j in range(
                    actual_start,
                    digit_end,
                ):
                    remove.add(j)

                index = digit_end
                continue

            index += 1

        return [
            token
            for i, token in enumerate(tokens)
            if i not in remove
        ]

    # ========================================================
    # CONTEXT
    # ========================================================

    @classmethod
    def _build_vehicle_contexts(
        cls,
        tokens: list[TranscriptToken],
        normalized: list[dict[str, Any]],
        corrections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        contexts: list[
            dict[str, Any]
        ] = []

        correction_indexes = (
            cls._correction_token_indexes(
                corrections
            )
        )

        command_indexes = sorted(
            {
                correction[
                    "command_index"
                ]
                for correction in corrections
                if isinstance(
                    correction.get(
                        "command_index"
                    ),
                    int,
                )
            }
        )

        for i, item in enumerate(
            normalized
        ):

            candidate = item[
                "candidate"
            ]

            # ------------------------------------------------
            # Start after current plate.
            # ------------------------------------------------

            start = (
                candidate.token_end
                + 1
            )

            # ------------------------------------------------
            # End before next plate.
            # ------------------------------------------------

            if i + 1 < len(
                normalized
            ):

                end = normalized[
                    i + 1
                ][
                    "candidate"
                ].token_start

            else:

                end = len(
                    tokens
                )

            # ------------------------------------------------
            # Correction command terminates vehicle context.
            # ------------------------------------------------

            for command_index in (
                command_indexes
            ):

                if (
                    command_index >= start
                    and command_index < end
                ):

                    end = command_index
                    break

            if end < start:
                end = start

            filtered_tokens = []

            for token_index, token in enumerate(
                tokens[start:end],
                start=start,
            ):

                if (
                    token_index
                    in correction_indexes
                ):
                    continue

                filtered_tokens.append(
                    token
                )

            # ------------------------------------------------
            # Remove plate-like fragments.
            # ------------------------------------------------

            cleaned_tokens = (
                cls._remove_plate_like_fragments(
                    filtered_tokens
                )
            )

            context = " ".join(
                token.text
                for token in cleaned_tokens
            ).strip()

            contexts.append(
                {
                    "plate_number":
                        item[
                            "plate_number"
                        ],
                    "context":
                        context,
                    "_candidate":
                        candidate,
                }
            )

        return contexts

    # ========================================================
    # SEMANTIC PROMPT
    # ========================================================

    @staticmethod
    def _semantic_prompt() -> str:
        return r"""
أنت طبقة الفهم الدلالي لنظام بيانات مركبات.

حلل سيارة واحدة فقط.

النص الموجود في CONTEXT هو المصدر الوحيد للمعلومات.

أعد أربعة حقول فقط:
- vehicle_type
- vehicle_type_evidence
- notes
- notes_evidence

RULE 1 — VEHICLE TYPE

vehicle_type هو نوع أو فئة المركبة نفسها فقط.

أي كلمة أو عبارة واضحة تصف نوع أو فئة المركبة يمكن أن تكون
vehicle_type.

لا تستخدم whitelist.

إذا كان نوع المركبة يتكون من أكثر من كلمة، يجب إرجاع
العبارة الكاملة وليس جزءًا منها.

مثال:

CONTEXT = نقل عام

الصحيح:
vehicle_type = نقل عام
vehicle_type_evidence = نقل عام

وليس:
vehicle_type = نقل

مثال:

CONTEXT = ونيت

الصحيح:
vehicle_type = ونيت
vehicle_type_evidence = ونيت

مثال:

CONTEXT = فان بضاعة

إذا كانت العبارة كلها تشير إلى نوع المركبة،
فأعد العبارة الكاملة.

لا تختصر vehicle_type إلى جزء من عبارة أطول.

RULE 2 — NOT VEHICLE TYPE

الشركة المصنعة، الموديل، اللون، الحالة، الاتجاه، المكان،
والوصف الإضافي ليست vehicle_type.

يمكن أن تكون notes.

RULE 3 — NOTES

notes هي المعلومات المفيدة الأخرى الخاصة بالمركبة
بعد استثناء vehicle_type.

RULE 4 — OWNERSHIP

كل جزء مفيد من CONTEXT يجب أن يذهب إلى vehicle_type
أو notes أو يتم تجاهله.

لا تستخدم نفس evidence في الحقلين.

RULE 5 — EVIDENCE

vehicle_type_evidence يجب أن تكون العبارة الأصلية حرفياً من CONTEXT.

notes_evidence يجب أن تكون العبارة الأصلية حرفياً من CONTEXT.

لا تخترع evidence.

RULE 6 — ASR NORMALIZATION

يمكن تصحيح خطأ Speech-to-Text واضح جداً داخل value
عندما يكون المقصود واضحاً جداً.

لكن evidence يجب أن تبقى النص الأصلي الموجود حرفياً
في CONTEXT.

لا تقم بالتصحيح عند الشك.

RULE 7 — NO HALLUCINATION

لا تستخرج plate_number.

لا تغير plate_number.

لا تضيف كلمات غير مدعومة.

لا تخترع نوع مركبة أو ملاحظة.

عند الشك:
أعد null.
"""

    @staticmethod
    def _semantic_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "vehicle_type": {
                    "type": [
                        "string",
                        "null",
                    ],
                },
                "vehicle_type_evidence": {
                    "type": [
                        "string",
                        "null",
                    ],
                },
                "notes": {
                    "type": [
                        "string",
                        "null",
                    ],
                },
                "notes_evidence": {
                    "type": [
                        "string",
                        "null",
                    ],
                },
            },
            "required": [
                "vehicle_type",
                "vehicle_type_evidence",
                "notes",
                "notes_evidence",
            ],
        }

    # ========================================================
    # SEMANTIC VALIDATION
    # ========================================================

    @classmethod
    def _grounded(
        cls,
        evidence: Any,
        context: str,
    ) -> str | None:

        evidence = cls._clean(
            evidence
        )

        if not evidence:
            return None

        if (
            cls._compact(
                evidence
            )
            not in cls._compact(
                context
            )
        ):
            return None

        return evidence

    @classmethod
    def _safe_value(
        cls,
        value: Any,
        evidence: str,
    ) -> str | None:

        value = cls._clean(
            value
        )

        if not value:
            return None

        source_tokens = (
            evidence.split()
        )

        value_tokens = (
            value.split()
        )

        if abs(
            len(value_tokens)
            - len(source_tokens)
        ) > 2:
            return evidence

        if len(value) > max(
            len(evidence) + 32,
            int(
                len(evidence)
                * 2.0
            ) + 8,
        ):
            return evidence

        return value

    @classmethod
    def _validate_semantic(
        cls,
        raw: Any,
        context: str,
    ) -> dict[str, Any]:

        result = {
            "vehicle_type": None,
            "vehicle_type_evidence": None,
            "notes": None,
        }

        if (
            not isinstance(
                raw,
                dict,
            )
            or not context
        ):
            return result

        type_evidence = (
            cls._grounded(
                raw.get(
                    "vehicle_type_evidence"
                ),
                context,
            )
        )

        note_evidence = (
            cls._grounded(
                raw.get(
                    "notes_evidence"
                ),
                context,
            )
        )

        vehicle_type = (
            cls._safe_value(
                raw.get(
                    "vehicle_type"
                ),
                type_evidence,
            )
            if type_evidence
            else None
        )

        notes = (
            cls._safe_value(
                raw.get(
                    "notes"
                ),
                note_evidence,
            )
            if note_evidence
            else None
        )

        # One evidence span cannot belong to both fields.
        if (
            type_evidence
            and note_evidence
            and cls._compact(
                type_evidence
            )
            == cls._compact(
                note_evidence
            )
        ):
            notes = None
            note_evidence = None

        if not type_evidence:
            vehicle_type = None

        if not note_evidence:
            notes = None

        result[
            "vehicle_type"
        ] = vehicle_type

        result[
            "vehicle_type_evidence"
        ] = type_evidence

        result[
            "notes"
        ] = notes

        return result

    # ========================================================
    # SINGLE CONTEXT SEMANTIC EXTRACTION
    # ========================================================

    def _semantic_for_context(
        self,
        plate_number: str,
        context: str,
    ) -> dict[str, Any]:

        if not context:
            return {
                "vehicle_type": None,
                "vehicle_type_evidence": None,
                "notes": None,
            }

        user_prompt = (
            "PLATE (reference only):\n"
            f"{plate_number}\n\n"
            "CONTEXT:\n"
            f"{context}\n\n"
            "Classify the context. "
            "Return only the JSON object "
            "required by the schema."
        )

        try:

            raw = (
                self.llm.generate_structured(
                    system_prompt=(
                        self._semantic_prompt()
                    ),
                    user_prompt=user_prompt,
                    response_schema=(
                        self._semantic_schema()
                    ),
                    retries=1,
                )
            )

        except Exception:

            raw = {}

        validated = (
            self._validate_semantic(
                raw,
                context,
            )
        )
        # ====================================================
        # SPLIT VEHICLE TYPE FROM TRAILING DESCRIPTION
        # ====================================================

        if (
            validated["vehicle_type"]
            and validated["notes"] is None
        ):
            vehicle_type = validated["vehicle_type"]
            type_evidence = validated["vehicle_type_evidence"]

            if type_evidence:
                type_tokens = type_evidence.split()

                if len(type_tokens) >= 2:
                    split_words = {
                        "حادث",
                        "مصدوم",
                        "مصدومه",
                        "مصدومة",
                        "مركونه",
                        "مركونة",
                        "متربه",
                        "متربة",
                        "متحرك",
                        "مطربه",
                        "مطربة",
                    }

                    split_at = None

                    for index in range(1, len(type_tokens)):
                        if type_tokens[index] in split_words:
                            split_at = index
                            break

                    if split_at is not None:
                        clean_type_evidence = " ".join(
                            type_tokens[:split_at]
                        )

                        note_evidence = " ".join(
                            type_tokens[split_at:]
                        )

                        validated["vehicle_type"] = clean_type_evidence
                        validated["vehicle_type_evidence"] = (
                            clean_type_evidence
                        )

                        validated["notes"] = note_evidence
        # ====================================================
        # SECOND OPINION
        # ====================================================

        if (
            not validated[
                "vehicle_type"
            ]
            or not validated[
                "vehicle_type_evidence"
            ]
        ):

            review_schema = {
                "type": "object",
                "properties": {
                    "vehicle_type": {
                        "type": [
                            "string",
                            "null",
                        ],
                    },
                    "evidence": {
                        "type": [
                            "string",
                            "null",
                        ],
                    },
                },
                "required": [
                    "vehicle_type",
                    "evidence",
                ],
            }

            try:

                review = (
                    self.llm.generate_structured(
                        system_prompt=(
                            "Find the clearest vehicle "
                            "type in CONTEXT.\n\n"
                            "A vehicle type may be one "
                            "word or multiple words.\n\n"
                            "If a multi-word phrase expresses "
                            "the vehicle type, return the "
                            "COMPLETE phrase, not a shorter "
                            "substring.\n\n"
                            "For example, if CONTEXT contains "
                            "'نقل عام', return 'نقل عام', "
                            "not only 'نقل'.\n\n"
                            "If CONTEXT contains a clear "
                            "vehicle type such as 'ونيت', "
                            "return it as vehicle_type.\n\n"
                            "Do not use a whitelist.\n\n"
                            "Manufacturer, model, color, "
                            "condition, direction and location "
                            "are not vehicle_type.\n\n"
                            "Evidence must be copied exactly "
                            "from CONTEXT.\n\n"
                            "ASR normalization is allowed only "
                            "when the intended vehicle type is "
                            "very clear.\n\n"
                            "No guessing."
                        ),
                        user_prompt=(
                            f"CONTEXT:\n{context}"
                        ),
                        response_schema=(
                            review_schema
                        ),
                        retries=1,
                    )
                )

            except Exception:

                review = {}

            evidence = (
                self._grounded(
                    (
                        review.get(
                            "evidence"
                        )
                        if isinstance(
                            review,
                            dict,
                        )
                        else None
                    ),
                    context,
                )
            )

            value = (
                self._safe_value(
                    (
                        review.get(
                            "vehicle_type"
                        )
                        if isinstance(
                            review,
                            dict,
                        )
                        else None
                    ),
                    evidence,
                )
                if evidence
                else None
            )

            if value and evidence:

                validated[
                    "vehicle_type"
                ] = value

                validated[
                    "vehicle_type_evidence"
                ] = evidence

                if (
                    validated[
                        "notes"
                    ]
                    and self._compact(
                        validated[
                            "notes"
                        ]
                    )
                    == self._compact(
                        evidence
                    )
                ):
                    validated[
                        "notes"
                    ] = None

        return validated

    # ========================================================
    # APPLY CORRECTIONS
    # ========================================================

    @staticmethod
    def _apply_corrections(
        cars: list[Vehicle],
        corrections: list[dict[str, Any]],
    ) -> list[Vehicle]:

        result = cars.copy()

        for correction in corrections:

            correction_type = (
                correction.get(
                    "type"
                )
            )

            # =================================================
            # LATEST
            #
            # IMPORTANT:
            # This function is no longer used to blindly modify
            # result[-1].
            #
            # The latest correction is applied to normalized
            # candidates BEFORE semantic extraction.
            #
            # This method exists only for compatibility and
            # specific corrections.
            # =================================================

            if correction_type == "latest":
                continue

            # =================================================
            # SPECIFIC
            # =================================================

            if correction_type == "specific":

                old_plate = (
                    correction.get(
                        "old_plate"
                    )
                )

                new_plate = (
                    correction.get(
                        "new_plate"
                    )
                )

                if (
                    not old_plate
                    or not new_plate
                ):
                    continue

                for index, car in enumerate(
                    result
                ):

                    if (
                        car.plate_number
                        == old_plate
                    ):

                        result[index] = (
                            car.model_copy(
                                update={
                                    "plate_number":
                                        new_plate
                                }
                            )
                        )

                        break

        return result

    # ========================================================
    # APPLY CORRECTIONS TO NORMALIZED PLATES
    # ========================================================

    @staticmethod
    def _apply_corrections_to_normalized(
        normalized: list[dict[str, Any]],
        corrections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Apply corrections to the existing real vehicle occurrence.

        For latest correction:

            تعديل اخر لوحه الي NEW

        update the latest vehicle that appeared BEFORE the command.

        For specific correction:

            تعديل اللوحة OLD إلى NEW

        update the existing vehicle whose plate is OLD.

        No new vehicle is ever created.
        """

        result = [
            item.copy()
            for item in normalized
        ]

        for correction in corrections:

            correction_type = (
                correction.get(
                    "type"
                )
            )

            # =================================================
            # LATEST
            # =================================================

            if correction_type == "latest":

                command_index = (
                    correction.get(
                        "command_index"
                    )
                )

                new_plate = (
                    correction.get(
                        "new_plate"
                    )
                )

                if (
                    not isinstance(
                        command_index,
                        int,
                    )
                    or not new_plate
                ):
                    continue

                target_index = None

                # Search from the end because we need the most
                # recent REAL vehicle before this command.
                for index in range(
                    len(result) - 1,
                    -1,
                    -1,
                ):

                    candidate = result[
                        index
                    ][
                        "candidate"
                    ]

                    if (
                        candidate.token_end
                        < command_index
                    ):
                        target_index = index
                        break

                if target_index is not None:

                    result[
                        target_index
                    ] = {
                        **result[
                            target_index
                        ],
                        "plate_number":
                            new_plate,
                    }

                continue

            # =================================================
            # SPECIFIC
            # =================================================

            if correction_type == "specific":

                old_plate = (
                    correction.get(
                        "old_plate"
                    )
                )

                new_plate = (
                    correction.get(
                        "new_plate"
                    )
                )

                if (
                    not old_plate
                    or not new_plate
                ):
                    continue

                for index, item in enumerate(
                    result
                ):

                    if (
                        item[
                            "plate_number"
                        ]
                        == old_plate
                    ):

                        result[
                            index
                        ] = {
                            **result[
                                index
                            ],
                            "plate_number":
                                new_plate,
                        }

                        break

        return result

    # ========================================================
    # TIMESTAMPS
    # ========================================================

    @staticmethod
    def _timestamp_fields(candidate: Any) -> tuple[int, int, int]:
        start_seconds = max(float(candidate.start), 0.0)
        end_seconds = max(float(candidate.end), start_seconds)

        start_ms = int(round(start_seconds * 1000.0))
        end_ms = int(round(end_seconds * 1000.0))

        midpoint = int(round((start_ms + end_ms) / 2.0))

        return midpoint, start_ms, end_ms
    # ========================================================
    # MAIN
    # ========================================================

    def extract(
        self,
        segments: list[TranscriptSegment],
        audio_path: str | None = None,
        recording_duration_ms: int = 0,
    ) -> VehicleExtractionResponse:

        # ----------------------------------------------------
        # Recording duration
        # ----------------------------------------------------

        if recording_duration_ms <= 0:

            recording_duration_ms = int(
                round(
                    max(
                        (
                            float(
                                segment.end
                            )
                            for segment in segments
                        ),
                        default=0.0,
                    )
                    * 1000.0
                )
            )

        # ----------------------------------------------------
        # Tokenize transcript.
        # ----------------------------------------------------

        tokens = tokenize_segments(
            segments
        )

        if not tokens:

            return VehicleExtractionResponse(
                recording_duration_ms=max(
                    int(
                        recording_duration_ms
                    ),
                    0,
                ),
                cars=[],
            )

        # ====================================================
        # STEP 1
        #
        # Detect corrections BEFORE normal vehicle extraction.
        # ====================================================

        corrections = (
            self._detect_corrections(
                tokens
            )
        )

        correction_indexes = (
            self._correction_token_indexes(
                corrections
            )
        )

        # ====================================================
        # STEP 2
        #
        # Extract normal candidates.
        # ====================================================

        candidates = (
            extract_plate_candidates(
                segments
            )
        )

        if not candidates:

            return VehicleExtractionResponse(
                recording_duration_ms=max(
                    int(
                        recording_duration_ms
                    ),
                    0,
                ),
                cars=[],
            )

        normalized: list[
            dict[str, Any]
        ] = []

        for candidate_index, candidate in enumerate(
            candidates
        ):

            # ------------------------------------------------
            # Ignore candidates inside correction commands.
            # ------------------------------------------------

            if (
                self._candidate_overlaps_correction(
                    candidate,
                    correction_indexes,
                )
            ):
                continue

            # ------------------------------------------------
            # Revalidate candidate directly from its original
            # token text.
            # ------------------------------------------------

            plate = (
                self._normalize_candidate(
                    candidate
                )
            )

            if not self._is_valid_plate(
                plate
            ):
                continue

            normalized.append(
                {
                    "plate_number":
                        plate,
                    "candidate":
                        candidate,
                    "candidate_index":
                        candidate_index,
                }
            )

        # ====================================================
        # STEP 3
        #
        # Consecutive duplicate removal.
        # ====================================================

        unique: list[
            dict[str, Any]
        ] = []

        for item in normalized:

            if not unique:

                unique.append(
                    item
                )

                continue

            previous = unique[-1]

            same_plate = (
                item[
                    "plate_number"
                ]
                ==
                previous[
                    "plate_number"
                ]
            )

            close_in_time = (
                item[
                    "candidate"
                ].start
                -
                previous[
                    "candidate"
                ].end
                <= 2.0
            )

            if (
                same_plate
                and close_in_time
            ):
                continue

            unique.append(
                item
            )

        normalized = unique

        if not normalized:

            return VehicleExtractionResponse(
                recording_duration_ms=max(
                    int(
                        recording_duration_ms
                    ),
                    0,
                ),
                cars=[],
            )

        # ====================================================
        # STEP 4
        #
        # APPLY CORRECTIONS TO THE EXISTING VEHICLE OCCURRENCES
        # ====================================================

        normalized = (
            self._apply_corrections_to_normalized(
                normalized,
                corrections,
            )
        )

        # ====================================================
        # STEP 5
        #
        # Build semantic contexts AFTER corrections.
        # ====================================================

        contexts = (
            self._build_vehicle_contexts(
                tokens,
                normalized,
                corrections,
            )
        )

        # ====================================================
        # STEP 6
        #
        # Semantic extraction.
        # ====================================================

        final_cars: list[
            Vehicle
        ] = []

        for context_item in contexts:

            plate_number = (
                context_item[
                    "plate_number"
                ]
            )

            context = (
                context_item[
                    "context"
                ]
            )

            semantic = (
                self._semantic_for_context(
                    plate_number,
                    context,
                )
            )

            (
                timestamp_ms,
                timestamp_start_ms,
                timestamp_end_ms,
            ) = self._timestamp_fields(
                context_item[
                    "_candidate"
                ]
            )

            final_cars.append(
                Vehicle(
                    plate_number=(
                        plate_number
                    ),
                    timestamp_ms=(
                        timestamp_ms
                    ),
                    timestamp_start_ms=(
                        timestamp_start_ms
                    ),
                    timestamp_end_ms=(
                        timestamp_end_ms
                    ),
                    vehicle_type=(
                        self._clean(
                            semantic.get(
                                "vehicle_type"
                            )
                        )
                    ),
                    notes=(
                        self._clean(
                            semantic.get(
                                "notes"
                            )
                        )
                    ),
                )
            )

        # ====================================================
        # STEP 7
        #
        # Specific corrections are already applied to
        # normalized before semantic extraction.
        #
        # This call is intentionally kept for compatibility,
        # but latest corrections are skipped here.
        # ====================================================

        final_cars = (
            self._apply_corrections(
                final_cars,
                corrections,
            )
        )

        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        return VehicleExtractionResponse(
            recording_duration_ms=max(
                int(
                    recording_duration_ms
                ),
                0,
            ),
            cars=final_cars,
        )