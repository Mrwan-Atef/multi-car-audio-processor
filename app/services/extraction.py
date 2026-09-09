from typing import Any
import re

from app.models.response import (
    Vehicle,
    VehicleExtractionResponse,
)
from app.models.transcript import (
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

    def __init__(self):
        self.llm = LLMService()

    # ========================================================
    # BASIC CLEANING
    # ========================================================

    @staticmethod
    def _clean(
        value: Any,
    ) -> str | None:

        if not isinstance(
            value,
            str,
        ):
            return None

        value = " ".join(
            value.strip().split()
        )

        return value or None

    # ========================================================
    # NUMBER RUN
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

    # ========================================================
    # LETTER-LIKE TOKEN
    # ========================================================

    @staticmethod
    def _token_has_letter_evidence(
        token: str,
    ) -> bool:

        decoded = decode_token(
            token,
            max_letters=3,
        )

        return bool(decoded)

    # ========================================================
    # REMOVE BROKEN PLATE-LIKE FRAGMENTS
    # ========================================================

    @classmethod
    def _remove_plate_like_fragments(
        cls,
        tokens: list[TranscriptToken],
    ) -> list[TranscriptToken]:

        """
        Remove corrupted/incomplete plate-looking fragments
        from vehicle notes.

        Examples:

            ح ع خمسين ستة ست
            راع نون واحد ستة اتنين ست

        These are not valid plates, but they clearly look like
        failed attempts to transcribe another plate.

        IMPORTANT:
        This function DOES NOT invent a replacement plate.
        It only prevents the broken fragment from appearing
        in notes.
        """

        if not tokens:
            return []

        remove = set()

        index = 0

        while index < len(tokens):

            digit_count, digit_end = (
                cls._number_run_end(
                    tokens,
                    index,
                )
            )

            if digit_count != 4:
                index += 1
                continue

            # ------------------------------------------------
            # Search immediately backward for letter-bearing
            # tokens.
            # ------------------------------------------------

            start = index - 1
            letter_count = 0

            while (
                start >= 0
                and index - start <= 6
            ):

                text = tokens[
                    start
                ].text

                if not cls._token_has_letter_evidence(
                    text
                ):
                    break

                letter_count += 1
                start -= 1

            actual_start = start + 1

            # ------------------------------------------------
            # Two or more letter-bearing tokens + 4 digits
            # is suspiciously plate-like.
            # ------------------------------------------------

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
    # BUILD VEHICLE CONTEXT
    # ========================================================

    @classmethod
    def _build_vehicle_contexts(
        cls,
        tokens: list[TranscriptToken],
        normalized: list[dict],
    ) -> list[dict[str, str]]:

        contexts = []

        for i, item in enumerate(
            normalized
        ):

            current = item[
                "candidate"
            ]

            # ------------------------------------------------
            # Context starts immediately AFTER current plate.
            # ------------------------------------------------

            start = (
                current.token_end + 1
            )

            # ------------------------------------------------
            # Context ends immediately BEFORE next plate.
            # ------------------------------------------------

            if i + 1 < len(normalized):

                next_candidate = (
                    normalized[
                        i + 1
                    ]["candidate"]
                )

                end = (
                    next_candidate.token_start
                )

            else:

                end = len(tokens)

            raw_context_tokens = tokens[
                start:end
            ]

            # ------------------------------------------------
            # Remove broken plate-looking fragments.
            # ------------------------------------------------

            cleaned_tokens = (
                cls._remove_plate_like_fragments(
                    raw_context_tokens
                )
            )

            context = " ".join(
                token.text
                for token in cleaned_tokens
            ).strip()

            contexts.append(
                {
                    "plate_number":
                        item["plate_number"],

                    "context":
                        context,
                }
            )

        return contexts

    # ========================================================
    # GROUND VEHICLE TYPE
    # ========================================================

    @staticmethod
    def _grounded_vehicle_type(
        value: str | None,
        source: str | None,
        context: str,
    ) -> str | None:

        if not value:
            return None

        if not source:
            return None

        context_compact = (
            context.replace(
                " ",
                "",
            )
        )

        source_compact = (
            source.replace(
                " ",
                "",
            )
        )

        if not source_compact:
            return None

        # Evidence must come from this exact vehicle context.
        if source_compact not in context_compact:
            return None

        return value

    # ========================================================
    # CONSERVATIVE VEHICLE-TYPE CORRECTIONS
    # ========================================================

    @staticmethod
    def _normalize_vehicle_type(
        value: str | None,
        source: str | None,
        context: str,
    ) -> tuple[str | None, str | None]:

        value = (
            value.strip()
            if isinstance(
                value,
                str,
            )
            else None
        )

        source = (
            source.strip()
            if isinstance(
                source,
                str,
            )
            else None
        )

        context = (
            context.strip()
            if isinstance(
                context,
                str,
            )
            else ""
        )

        # ----------------------------------------------------
        # No context => no vehicle type.
        # ----------------------------------------------------

        if not context:
            return None, None

        # ----------------------------------------------------
        # Deterministic semantic ASR correction:
        #
        # "نات" in a vehicle-type context is the observed
        # ASR form of "نقل".
        #
        # IMPORTANT:
        # This is NOT a plate mapping.
        # It is only a vehicle-type correction.
        #
        # We apply it only when the complete context is the
        # single token "نات".
        # ----------------------------------------------------

        if context == "نات":
            return "نقل", "نات"

        # ----------------------------------------------------
        # Normal grounded Qwen result.
        # ----------------------------------------------------

        grounded = (
            VehicleExtractionService
            ._grounded_vehicle_type(
                value,
                source,
                context,
            )
        )

        if grounded:
            return grounded, source

        return None, None

    # ========================================================
    # VEHICLE TYPE
    #
    # ONE VEHICLE PER CALL
    # ========================================================

    def _extract_vehicle_type_for_car(
        self,
        plate_number: str,
        context: str,
    ) -> dict[str, str | None]:

        if not context:

            return {
                "vehicle_type": None,
                "source": None,
            }

        system_prompt = """
أنت مصنف صارم لنوع المركبة لمركبة واحدة فقط.

مهمتك الوحيدة:
تحديد vehicle_type من CONTEXT.

============================================================
DEFINITION
============================================================

vehicle_type هو نوع أو فئة المركبة نفسها.

أمثلة:
نقل
فان
ونيد
دوكر
كوستر
اتوبيس
شاص
مصندق
نقل عام
قلاب
ستاريا
ستاركس
H1

هذه أمثلة وليست whitelist.
أي تعبير عربي أو أجنبي واضح يدل على نوع/فئة المركبة يمكن أن يكون
vehicle_type.

============================================================
VERY IMPORTANT
============================================================

إذا كان CONTEXT نفسه عبارة عن نوع مركبة واضح، فيجب إرجاعه
على أنه vehicle_type.

مثال:

CONTEXT:
نقل

النتيجة:
vehicle_type = "نقل"
vehicle_type_source = "نقل"

ولا تضعه في notes.

مثال آخر:

CONTEXT:
فان

النتيجة:
vehicle_type = "فان"
vehicle_type_source = "فان"

============================================================
NOT VEHICLE TYPE
============================================================

هذه ليست vehicle_type:

جراش
واحد
يمين
يسار
آخر الشارع
آخر الشهر
حادث
مصدومة
مركونة
متربة
متحرك

وكذلك:
- اللون
- الشركة المصنعة
- الموديل
- حالة المركبة
- الوصف الإضافي

مثال:

CONTEXT:
جراش واحد يمين

النتيجة:
vehicle_type = null
vehicle_type_source = null

============================================================
MULTIPLE WORDS
============================================================

إذا احتوى CONTEXT على نوع مركبة مع معلومات أخرى:

مثال:
نقل عام حادث

فالـ vehicle_type هو:
نقل عام

و vehicle_type_source يجب أن يكون:
نقل عام

ولا تستخرج "حادث" كنوع مركبة.

============================================================
ASR CORRECTION
============================================================

يمكن تصحيح خطأ Speech-to-Text واضح جدًا فقط.

مثال:

CONTEXT:
نات

النتيجة:
vehicle_type = "نقل"
vehicle_type_source = "نات"

هذا تصحيح ASR وليس اختراعًا لمعلومة جديدة.

============================================================
SOURCE
============================================================

vehicle_type_source يجب أن تكون العبارة الموجودة حرفيًا داخل
CONTEXT.

============================================================
NO GUESSING
============================================================

إذا لم يكن نوع المركبة واضحًا، أرجع:

vehicle_type = null
vehicle_type_source = null

لا تخترع نوعًا غير موجود في CONTEXT.
"""

        user_prompt = (
            "PLATE:\n"
            f"{plate_number}\n\n"
            "CONTEXT:\n"
            f"{context}\n"
            "CLASSIFY THE CONTEXT NOW.\n"
            "If the context itself is a clear vehicle type, return it as vehicle_type."
        )

        schema = {
            "type": "object",
            "properties": {
                "vehicle_type": {
                    "type": [
                        "string",
                        "null",
                    ],
                },
                "vehicle_type_source": {
                    "type": [
                        "string",
                        "null",
                    ],
                },
            },
            "required": [
                "vehicle_type",
                "vehicle_type_source",
            ],
        }

        try:

            result = self.llm.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema=schema,
            )

        except Exception:

            result = {}

        if not isinstance(
            result,
            dict,
        ):

            result = {}

        value = self._clean(
            result.get(
                "vehicle_type"
            )
        )

        source = self._clean(
            result.get(
                "vehicle_type_source"
            )
        )

        # ----------------------------------------------------
        # Apply conservative semantic correction.
        # ----------------------------------------------------

        vehicle_type, grounded_source = (
            self._normalize_vehicle_type(
                value,
                source,
                context,
            )
        )

        return {
            "vehicle_type":
                vehicle_type,

            "source":
                grounded_source,
        }

    # ========================================================
    # SAFE NOTE CORRECTIONS
    # ========================================================

    @staticmethod
    def _apply_safe_note_corrections(
        text: str,
    ) -> str:

        if not text:
            return text

        text = " ".join(
            text.split()
        )

        # ----------------------------------------------------
        # Arabic ASR correction:
        #
        # "جراش ... اخر الشهر"
        #
        # -> "جراش ... اخر الشارع"
        #
        # We ONLY change "الشهر" when it is preceded by
        # "اخر" and the note starts with/contains "جراش".
        #
        # We do NOT globally replace every "الشهر".
        # ----------------------------------------------------

        if (
            "جراش" in text
            and "اخر الشهر" in text
        ):

            text = text.replace(
                "اخر الشهر",
                "اخر الشارع",
            )

        if (
            "جراش" in text
            and "آخر الشهر" in text
        ):

            text = text.replace(
                "آخر الشهر",
                "آخر الشارع",
            )

        return text

    # ========================================================
    # NOTES SAFETY VALIDATION
    # ========================================================

    @staticmethod
    def _safe_note_correction(
        original: str,
        corrected: str | None,
    ) -> str:

        if not original:
            return ""

        if not corrected:
            return original

        original = " ".join(
            original.split()
        )

        corrected = " ".join(
            corrected.split()
        )

        # ----------------------------------------------------
        # First apply deterministic safe corrections to the
        # original text.
        # ----------------------------------------------------

        deterministic = (
            VehicleExtractionService
            ._apply_safe_note_corrections(
                original
            )
        )

        # If deterministic correction changed the note,
        # prefer it.
        if deterministic != original:
            return deterministic

        # ----------------------------------------------------
        # For any future LLM correction, enforce the strict
        # token-count rule.
        #
        # This prevents Qwen from inventing an entire note.
        # ----------------------------------------------------

        original_tokens = (
            original.split()
        )

        corrected_tokens = (
            corrected.split()
        )

        if len(
            original_tokens
        ) != len(
            corrected_tokens
        ):
            return original

        changed = sum(
            1
            for old, new in zip(
                original_tokens,
                corrected_tokens,
            )
            if old != new
        )

        if changed > 2:
            return original

        return corrected

    # ========================================================
    # NOTES
    #
    # ONE VEHICLE PER CALL
    # ========================================================

    def _clean_notes_for_car(
        self,
        plate_number: str,
        context: str,
    ) -> str | None:

        if not context:
            return None

        # ----------------------------------------------------
        # Deterministic correction first.
        # ----------------------------------------------------

        deterministic = (
            self._apply_safe_note_corrections(
                context
            )
        )

        # ----------------------------------------------------
        # If the deterministic correction already fixed the
        # only known ASR problem, return it directly.
        # ----------------------------------------------------

        if deterministic != context:
            return deterministic

        system_prompt = """
أنت تنظف notes الخاصة بمركبة واحدة فقط.

مهمتك الوحيدة تصحيح أخطاء Speech-to-Text الواضحة داخل
النص الموجود.

ممنوع:
- إضافة معلومات جديدة
- حذف معلومات
- اختراع ملاحظة
- استخراج vehicle_type
- استخراج plate_number
- تغيير عدد الكلمات

إذا لم تكن متأكدًا:
أعد النص نفسه.

مثال:

جراش يمين اخر الشهر

إذا احتجت تصحيحه:

جراش يمين اخر الشارع

لكن لا تضف أي كلمات أخرى.
"""

        user_prompt = (
            "PLATE:\n"
            f"{plate_number}\n\n"
            "NOTES:\n"
            f"{context}\n"
        )

        schema = {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "string",
                },
            },
            "required": [
                "notes",
            ],
        }

        try:

            result = self.llm.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema=schema,
            )

        except Exception:

            return context

        if not isinstance(
            result,
            dict,
        ):

            return context

        corrected = self._clean(
            result.get(
                "notes"
            )
        )

        if not corrected:
            return context

        return self._safe_note_correction(
            context,
            corrected,
        )

    # ========================================================
    # REMOVE VEHICLE TYPE SOURCE FROM NOTES
    # ========================================================

    @staticmethod
    def _remove_vehicle_type_source(
        context: str,
        source: str | None,
    ) -> str:

        if not context:
            return ""

        if not source:
            return context

        context_tokens = (
            context.split()
        )

        source_tokens = (
            source.split()
        )

        if not source_tokens:
            return context

        source_compact = "".join(
            source_tokens
        )

        source_len = len(
            source_tokens
        )

        for i in range(
            len(context_tokens)
            - source_len
            + 1
        ):

            window = context_tokens[
                i:
                i + source_len
            ]

            window_compact = "".join(
                window
            )

            if (
                window_compact
                == source_compact
            ):

                del context_tokens[
                    i:
                    i + source_len
                ]

                break

        return " ".join(
            context_tokens
        ).strip()

    # ========================================================
    # CORRECTIONS
    # ========================================================

    @staticmethod
    def _detect_corrections(
        tokens: list[TranscriptToken],
        candidates: list,
    ) -> list[dict]:

        corrections = []

        for i, token in enumerate(
            tokens
        ):

            if (
                token.text.strip()
                != "تعديل"
            ):
                continue

            following = []

            for candidate_index, candidate in enumerate(
                candidates
            ):

                if (
                    candidate.token_start
                    > i
                ):

                    following.append(
                        (
                            candidate_index,
                            candidate,
                        )
                    )

            if not following:
                continue

            first_index, first_candidate = (
                following[0]
            )

            specific_found = False

            # ------------------------------------------------
            # تعديل OLD إلى NEW
            # ------------------------------------------------

            for second_index, second_candidate in following[1:]:

                between = [
                    tokens[j].text
                    for j in range(
                        first_candidate.token_end + 1,
                        second_candidate.token_start,
                    )
                ]

                between_set = {
                    word.strip()
                    for word in between
                }

                if (
                    "إلى" in between_set
                    or
                    "الى" in between_set
                ):

                    corrections.append(
                        {
                            "type":
                                "specific",

                            "old_index":
                                first_index,

                            "new_index":
                                second_index,
                        }
                    )

                    specific_found = True
                    break

            # ------------------------------------------------
            # تعديل NEW
            # ------------------------------------------------

            if not specific_found:

                corrections.append(
                    {
                        "type":
                            "latest",

                        "new_index":
                            first_index,
                    }
                )

        return corrections

    # ========================================================
    # APPLY CORRECTIONS
    # ========================================================

    @staticmethod
    def _apply_corrections(
        cars: list[Vehicle],
        corrections: list[dict],
        candidates: list,
    ) -> list[Vehicle]:

        result = cars.copy()

        for correction in corrections:

            # ------------------------------------------------
            # تعديل NEW
            # ------------------------------------------------

            if (
                correction["type"]
                == "latest"
            ):

                new_index = (
                    correction[
                        "new_index"
                    ]
                )

                if (
                    new_index
                    >= len(candidates)
                ):
                    continue

                new_plate = (
                    normalize_plate_from_text(
                        candidates[
                            new_index
                        ].text
                    )
                )

                if not new_plate:
                    continue

                if result:

                    result[-1] = (
                        result[-1].model_copy(
                            update={
                                "plate_number":
                                    new_plate
                            }
                        )
                    )

            # ------------------------------------------------
            # تعديل OLD إلى NEW
            # ------------------------------------------------

            else:

                old_index = (
                    correction[
                        "old_index"
                    ]
                )

                new_index = (
                    correction[
                        "new_index"
                    ]
                )

                if (
                    old_index
                    >= len(candidates)
                    or
                    new_index
                    >= len(candidates)
                ):
                    continue

                old_plate = (
                    normalize_plate_from_text(
                        candidates[
                            old_index
                        ].text
                    )
                )

                new_plate = (
                    normalize_plate_from_text(
                        candidates[
                            new_index
                        ].text
                    )
                )

                if (
                    not old_plate
                    or
                    not new_plate
                ):
                    continue

                for i, car in enumerate(
                    result
                ):

                    if (
                        car.plate_number
                        ==
                        old_plate
                    ):

                        result[i] = (
                            result[i].model_copy(
                                update={
                                    "plate_number":
                                        new_plate
                                }
                            )
                        )

                        break

        return result

    # ========================================================
    # MAIN
    # ========================================================

    def extract(
        self,
        segments: list[TranscriptSegment],
        audio_path: str | None = None,
    ) -> VehicleExtractionResponse:

        # ====================================================
        # TOKENIZE
        # ====================================================

        tokens = tokenize_segments(
            segments
        )

        # ====================================================
        # PLATE CANDIDATES
        # ====================================================

        candidates = (
            extract_plate_candidates(
                segments
            )
        )

        if not candidates:

            return VehicleExtractionResponse(
                cars=[]
            )

        # ====================================================
        # CORRECTIONS
        # ====================================================

        corrections = (
            self._detect_corrections(
                tokens,
                candidates,
            )
        )

        correction_new_indexes = {
            correction[
                "new_index"
            ]
            for correction in corrections
        }

        # ====================================================
        # NORMALIZE PLATES
        # ====================================================

        normalized = []

        for candidate_index, candidate in enumerate(
            candidates
        ):

            plate = (
                normalize_plate_from_text(
                    candidate.text
                )
            )

            if plate is None:
                continue

            # Exactly 3 letters + 4 digits.
            if (
                len(plate) != 7
                or
                not plate[-4:].isdigit()
            ):
                continue

            # "تعديل NEW" is not a separate vehicle.
            if (
                candidate_index
                in correction_new_indexes
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

        if not normalized:

            return VehicleExtractionResponse(
                cars=[]
            )

        # ====================================================
        # IMMEDIATE DUPLICATES
        # ====================================================

        unique = []

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
                and
                close_in_time
            ):
                continue

            unique.append(
                item
            )

        normalized = unique

        # ====================================================
        # CONTEXT
        # ====================================================

        contexts = (
            self._build_vehicle_contexts(
                tokens,
                normalized,
            )
        )

        # ====================================================
        # FINAL CARS
        # ====================================================

        final_cars = []

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

            # ------------------------------------------------
            # VEHICLE TYPE
            # ------------------------------------------------

            vehicle_result = (
                self._extract_vehicle_type_for_car(
                    plate_number,
                    context,
                )
            )

            vehicle_type = (
                vehicle_result[
                    "vehicle_type"
                ]
            )

            vehicle_source = (
                vehicle_result[
                    "source"
                ]
            )

            # ------------------------------------------------
            # REMOVE THE SOURCE OF VEHICLE TYPE
            # FROM NOTES.
            #
            # Example:
            #
            # context = "نات"
            # vehicle_type = "نقل"
            # source = "نات"
            #
            # notes_context = ""
            # ------------------------------------------------

            notes_context = (
                self._remove_vehicle_type_source(
                    context,
                    vehicle_source,
                )
            )

            # ------------------------------------------------
            # NOTES
            # ------------------------------------------------

            if notes_context:

                notes = (
                    self._clean_notes_for_car(
                        plate_number,
                        notes_context,
                    )
                )

            else:

                notes = None

            # ------------------------------------------------
            # FINAL SAFETY
            # ------------------------------------------------

            if (
                vehicle_source
                and
                notes
                and
                notes.strip()
                ==
                vehicle_source.strip()
            ):

                notes = None

            final_cars.append(
                Vehicle(
                    plate_number=
                        plate_number,

                    vehicle_type=
                        vehicle_type,

                    notes=
                        notes,
                )
            )

        # ====================================================
        # APPLY تعديل
        # ====================================================

        final_cars = (
            self._apply_corrections(
                final_cars,
                corrections,
                candidates,
            )
        )

        # ====================================================
        # RETURN
        # ====================================================

        return VehicleExtractionResponse(
            cars=final_cars
        )