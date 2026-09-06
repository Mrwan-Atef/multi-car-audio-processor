from typing import Any

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
    normalize_plate_from_text,
)


class VehicleExtractionService:

    def __init__(self):

        self.llm = LLMService()

    # ========================================================
    # CONTEXT
    # ========================================================

    @staticmethod
    def _build_context(
        tokens: list[TranscriptToken],
        candidate_start: int,
        candidate_end: int,
        next_candidate_start: int | None,
    ) -> str:

        # We mainly want information immediately AFTER
        # the current plate.
        #
        # If the next plate starts, stop immediately.

        start = candidate_end + 1

        if next_candidate_start is None:

            end = min(
                len(tokens),
                start + 8,
            )

        else:

            end = min(
                next_candidate_start,
                start + 8,
            )

        return " ".join(
            token.text
            for token in tokens[
                start:end
            ]
        ).strip()

    # ========================================================
    # CLEAN
    # ========================================================

    @staticmethod
    def _string_or_none(
        value: Any,
    ) -> str | None:

        if not isinstance(
            value,
            str,
        ):
            return None

        value = value.strip()

        return value or None

    # ========================================================
    # EVIDENCE
    # ========================================================

    @staticmethod
    def _has_grounded_type(
        vehicle_type: str | None,
        evidence: str | None,
        context: str,
    ) -> bool:

        if not vehicle_type:
            return False

        if not evidence:
            return False

        context_compact = (
            context.replace(" ", "")
        )

        evidence_compact = (
            evidence.replace(" ", "")
        )

        type_compact = (
            vehicle_type.replace(" ", "")
        )

        return (
            evidence_compact
            in context_compact
            and type_compact
            in evidence_compact
        )

    # ========================================================
    # QWEN
    # ========================================================

    def _extract_semantics(
        self,
        items: list[dict[str, str]],
    ) -> list[dict[str, Any]]:

        if not items:
            return []

        system_prompt = """
أنت تعمل داخل نظام استخراج بيانات مركبات.

مهمتك فقط استخراج:
- vehicle_type
- vehicle_type_evidence
- notes

لا تستخرج أو تعدل plate_number.

قواعد صارمة جدًا:

vehicle_type:
- يجب أن يكون نوع أو فئة المركبة.
- يجب أن يكون النوع مذكورًا بوضوح في CONTEXT.
- لا تخمن.
- لا تستنتج النوع من لون أو موديل أو شركة مصنعة.
- لا تستنتج النوع من كلمة لا تصف المركبة بوضوح.
- إذا لم يوجد دليل واضح: null.

vehicle_type_evidence:
- يجب أن تكون عبارة موجودة حرفيًا في CONTEXT.
- يجب أن تحتوي على vehicle_type.
- إذا لم يوجد دليل: null.

notes:
- معلومات إضافية مذكورة حرفيًا في CONTEXT فقط.
- لا تخترع.
- لا تأخذ معلومات من سيارة أخرى.
- لا تضع كلمات أو أرقام اللوحة في notes.
- إذا لم توجد ملاحظات: null.

مهم جدًا:
إذا كان السياق:
"جراش واحد يمين"
فلا تستنتج أن "جراش" vehicle_type.

إذا لم تكن متأكدًا، استخدم null.
"""

        user_prompt = (
            "استخرج البيانات التالية "
            "مع الحفاظ على ترتيب السيارات:\n"
        )

        for i, item in enumerate(
            items,
            start=1,
        ):

            user_prompt += (
                f"\nCAR {i}\n"
                f"PLATE: {item['plate_number']}\n"
                f"CONTEXT: "
                f"{item['context'] or 'لا توجد معلومات'}\n"
            )

        schema = {
            "type": "object",
            "properties": {
                "cars": {
                    "type": "array",
                    "items": {
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
                        },
                        "required": [
                            "vehicle_type",
                            "vehicle_type_evidence",
                            "notes",
                        ],
                    },
                },
            },
            "required": [
                "cars",
            ],
        }

        result = self.llm.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=schema,
        )

        cars = result.get(
            "cars",
            [],
        )

        return (
            cars
            if isinstance(cars, list)
            else []
        )

    # ========================================================
    # MAIN
    # ========================================================

    def extract(
        self,
        segments: list[TranscriptSegment],
        audio_path: str | None = None,
    ) -> VehicleExtractionResponse:

        tokens = tokenize_segments(
            segments
        )

        candidates = extract_plate_candidates(
            segments
        )

        normalized = []

        for candidate in candidates:

            plate = normalize_plate_from_text(
                candidate.text
            )

            if plate is None:
                continue

            # Exactly:
            # 3 letters + 4 digits.
            if len(plate) != 7:
                continue

            if not plate[-4:].isdigit():
                continue

            normalized.append(
                {
                    "plate_number": plate,
                    "candidate": candidate,
                }
            )

        if not normalized:

            return VehicleExtractionResponse(
                cars=[]
            )

        # ====================================================
        # BUILD SEMANTIC CONTEXTS
        # ====================================================

        semantic_items = []

        for i, item in enumerate(
            normalized
        ):

            candidate = item[
                "candidate"
            ]

            next_start = None

            if i + 1 < len(
                normalized
            ):

                next_start = normalized[
                    i + 1
                ][
                    "candidate"
                ].token_start

            context = self._build_context(
                tokens=tokens,
                candidate_start=candidate.token_start,
                candidate_end=candidate.token_end,
                next_candidate_start=next_start,
            )

            semantic_items.append(
                {
                    "plate_number": item[
                        "plate_number"
                    ],
                    "context": context,
                }
            )

        # ====================================================
        # ONE QWEN CALL
        # ====================================================

        llm_cars = self._extract_semantics(
            semantic_items
        )

        # ====================================================
        # FINAL
        # ====================================================

        final_cars = []

        for i, item in enumerate(
            normalized
        ):

            vehicle_type = None
            notes = None

            context = semantic_items[
                i
            ]["context"]

            if (
                i < len(llm_cars)
                and isinstance(
                    llm_cars[i],
                    dict,
                )
            ):

                candidate_type = (
                    self._string_or_none(
                        llm_cars[i].get(
                            "vehicle_type"
                        )
                    )
                )

                evidence = (
                    self._string_or_none(
                        llm_cars[i].get(
                            "vehicle_type_evidence"
                        )
                    )
                )

                if self._has_grounded_type(
                    vehicle_type=candidate_type,
                    evidence=evidence,
                    context=context,
                ):

                    vehicle_type = (
                        candidate_type
                    )

                notes = (
                    self._string_or_none(
                        llm_cars[i].get(
                            "notes"
                        )
                    )
                )

                if notes:

                    notes = notes.replace(
                        item["plate_number"],
                        "",
                    ).strip()

                    if not notes:
                        notes = None

            final_cars.append(
                Vehicle(
                    plate_number=item[
                        "plate_number"
                    ],
                    vehicle_type=vehicle_type,
                    notes=notes,
                )
            )

        return VehicleExtractionResponse(
            cars=final_cars
        )