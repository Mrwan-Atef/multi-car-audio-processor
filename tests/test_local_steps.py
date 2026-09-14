
"""
LOCAL PIPELINE TESTS
====================

Run these tests on Windows WITHOUT Docker.

Place this file here:
    <your-project>\tests\test_local_steps.py

Then run from the PROJECT ROOT:

    python tests/test_local_steps.py --step normalizer
    python tests/test_local_steps.py --step candidate
    python tests/test_local_steps.py --step semantic
    python tests/test_local_steps.py --step asr --audio "D:\path\record.wav"
    python tests/test_local_steps.py --step all --audio "D:\path\record.wav"

Important:
- normalizer/candidate tests DO NOT need Ollama or the ASR model.
- semantic tests need Ollama running locally and Qwen installed in Ollama.
- ASR test uses your existing app.services.stt.STTService.
- This file does NOT modify your project files.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def ok(name: str, expected, actual) -> None:
    if expected == actual:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name}")
        print(f"       expected: {expected!r}")
        print(f"       actual:   {actual!r}")


def make_segment(text: str, start: float = 0.0, end: float = 1.0):
    """
    Build TranscriptSegment without depending on its exact constructor
    signature. We first try the normal Pydantic/dataclass constructor, then
    fall back to SimpleNamespace for tokenize_segments().
    """
    from app.models.transcript import TranscriptSegment

    attempts = [
        {"text": text, "start": start, "end": end, "words": []},
        {"text": text, "start": start, "end": end},
    ]

    for kwargs in attempts:
        try:
            return TranscriptSegment(**kwargs)
        except Exception:
            pass

    obj = SimpleNamespace(
        text=text,
        start=start,
        end=end,
        words=[],
    )
    return obj


# ---------------------------------------------------------------------------
# STEP 1: PLATE NORMALIZER
# ---------------------------------------------------------------------------

def test_normalizer() -> int:
    banner("STEP 1 — PLATE NORMALIZER")

    from app.services.plate_normalizer import (
        clean_transcript_text,
        decode_letter_phrase,
        normalize_digits_from_tokens,
        normalize_plate_from_text,
    )

    cases = [
        ("repeated letters: روب", "ر و ب ر و ب 1729", "روب1729"),
        ("repeated ألف", "ا ا ا 1234", "ااا1234"),
        ("normal plate", "ح ع ا 5077", "حعا5077"),
        ("repeated group", "ب ط ص ب ط ص 5745", "بطص5745"),
        ("English digits mixed", "ا ن ل ١٠٢٤", "انل1024"),
        ("Arabic spoken digits", "ا ن ل واحد صفر اتنين اربعة", "انل1024"),
        ("invalid: only two letters", "ر و 1729", None),
        ("invalid: only three digits", "ر و ب 172", None),
        ("invalid: five digits", "ر و ب 17290", None),
    ]

    failures = 0

    for name, text, expected in cases:
        try:
            actual = normalize_plate_from_text(text)
        except Exception as exc:
            actual = f"EXCEPTION: {type(exc).__name__}: {exc}"

        ok(name, expected, actual)
        failures += int(actual != expected)

    # Extra low-level diagnostics
    print("\n--- low-level diagnostics ---")
    for phrase in [
        "ر و ب ر و ب",
        "ا ا ا",
        "ب ط ص ب ط ص",
        "ح ع ا",
    ]:
        try:
            print(f"decode_letter_phrase({phrase!r}) -> {decode_letter_phrase(phrase.split())}")
        except Exception as exc:
            print(f"decode_letter_phrase({phrase!r}) -> EXCEPTION: {exc}")

    for text in [
        "خمسة سبعة اربعة خمسة",
        "واحد صفر اتنين اربعة",
        "واحد و عشرين",
    ]:
        try:
            print(f"normalize_digits_from_tokens({text!r}) -> "
                  f"{normalize_digits_from_tokens(text.split())}")
        except Exception as exc:
            print(f"normalize_digits_from_tokens({text!r}) -> EXCEPTION: {exc}")

    print(f"\nNormalizer failures: {failures}")
    return failures


# ---------------------------------------------------------------------------
# STEP 2: PLATE CANDIDATE EXTRACTOR
# ---------------------------------------------------------------------------

def test_candidate() -> int:
    banner("STEP 2 — PLATE CANDIDATE EXTRACTION")

    from app.services.plate_candidate import (
        extract_plate_candidates,
        tokenize_segments,
    )

    cases = [
        (
            "single repeated plate",
            "ر و ب ر و ب 1729",
            ["روب1729"],
        ),
        (
            "repeated ألف",
            "ا ا ا 1234",
            ["ااا1234"],
        ),
        (
            "two vehicles",
            "ا ع ا 1234 دوكر حادث ا س ي 1254",
            ["اعا1234", "اسي1254"],
        ),
        (
            "plate + vehicle type",
            "ب ط ص ب ط ص 5745 نقل",
            ["بطص5745"],
        ),
        (
            "sample 4 first two",
            "ح ع ح 5077 ب ط ص 1234 نقل",
            ["حعح5077", "بطص1234"],
        ),
    ]

    failures = 0

    for name, text, expected in cases:
        segments = [make_segment(text)]
        try:
            candidates = extract_plate_candidates(segments)
            actual = [getattr(c, "text", str(c)) for c in candidates]

            # Show normalized candidate metadata too.
            print(f"\n{name}")
            for c in candidates:
                print(
                    f"  raw={getattr(c, 'text', None)!r}, "
                    f"start={getattr(c, 'start', None)}, "
                    f"end={getattr(c, 'end', None)}"
                )

            from app.services.plate_normalizer import normalize_plate_from_text
            normalized = [normalize_plate_from_text(x) for x in actual]

        except Exception as exc:
            actual = f"EXCEPTION: {type(exc).__name__}: {exc}"
            normalized = actual

        ok(name, expected, normalized)
        failures += int(normalized != expected)

    # Tokenizer diagnostic
    print("\n--- tokenizer diagnostic ---")
    text = "ا ا ا 1234"
    tokens = tokenize_segments([make_segment(text, 0.0, 2.0)])
    for i, token in enumerate(tokens):
        print(
            f"{i}: text={token.text!r}, "
            f"start={token.start:.3f}, end={token.end:.3f}"
        )

    print(f"\nCandidate failures: {failures}")
    return failures


# ---------------------------------------------------------------------------
# STEP 3: CONTEXT / SEGMENTATION
# ---------------------------------------------------------------------------

def test_contexts() -> int:
    banner("STEP 3 — VEHICLE CONTEXT / RECORD BOUNDARIES")

    from app.services.plate_candidate import extract_plate_candidates, tokenize_segments
    from app.services.extraction import VehicleExtractionService

    cases = [
        (
            "two-record boundary",
            "ا ع ا 1234 دوكر حادث ا س ي 1254",
            [
                ("اعا1234", "دوكر حادث"),
                ("اسي1254", ""),
            ],
        ),
        (
            "vehicle types should stay with their own plate",
            "ح ع ح 5077 ب ط ص 1234 نقل ع ا د شاص متربة ه ن ي اتوبيس",
            [
                ("حعح5077", ""),
                ("بطص1234", "نقل"),
                # The rest depends on candidate extraction.
            ],
        ),
    ]

    failures = 0
    service = VehicleExtractionService.__new__(VehicleExtractionService)

    for name, text, expected_prefix in cases:
        try:
            segments = [make_segment(text)]
            candidates = extract_plate_candidates(segments)
            tokens = tokenize_segments(segments)

            normalized = []
            for c in candidates:
                from app.services.plate_normalizer import normalize_plate_from_text
                plate = normalize_plate_from_text(c.text)
                normalized.append({"plate_number": plate, "candidate": c})

            contexts = service._build_vehicle_contexts(tokens, normalized)

            print(f"\n{name}")
            for item in contexts:
                print(
                    f"  {item['plate_number']}: "
                    f"context={item['context']!r}"
                )

            for plate, expected_context in expected_prefix:
                found = next(
                    (x for x in contexts if x["plate_number"] == plate),
                    None,
                )
                actual_context = found["context"] if found else None

                if not expected_context:
                    # For records where we only care that no text leaks from the
                    # next plate, allow the context to be empty only.
                    expected = ""
                else:
                    expected = expected_context

                ok(
                    f"{name}: {plate}",
                    expected,
                    actual_context,
                )
                failures += int(actual_context != expected)

        except Exception as exc:
            print(f"[FAIL] {name}: EXCEPTION: {type(exc).__name__}: {exc}")
            failures += 1

    print(f"\nContext failures: {failures}")
    return failures


# ---------------------------------------------------------------------------
# STEP 4: QWEN SEMANTIC TEST
# ---------------------------------------------------------------------------

def test_semantic() -> int:
    banner("STEP 4 — QWEN SEMANTIC CLASSIFICATION (LOCAL OLLAMA)")

    from app.services.extraction import VehicleExtractionService

    service = VehicleExtractionService()
    # Your project config may use http://ollama:11434 for Docker.
    # For local testing we force the host Ollama endpoint.
    service.llm.base_url = os.getenv(
        "LOCAL_OLLAMA_URL",
        "http://127.0.0.1:11434",
    )
    service.llm.model = os.getenv(
        "LOCAL_LLM_MODEL",
        "qwen2.5:7b",
    )

    cases = [
        ("literal vehicle type", "نقل", "نقل", None),
        ("note only", "جراش واحد يمين", None, "جراش واحد يمين"),
        ("type + condition", "نقل عام مصدومة", "نقل عام", "مصدومة"),
        ("type + condition", "شاص متربة", "شاص", "متربة"),
        ("ASR vehicle type candidate", "نات", "نقل", None),
        ("command must not become note", "تعديل", None, None),
    ]

    failures = 0

    try:
        print(f"LLM endpoint: {service.llm.base_url}")
        print(f"LLM model:    {service.llm.model}")
        print("Testing Qwen directly through the same LLMService used by the app.\n")

        if not service.llm.health():
            print("[FAIL] Ollama is not reachable.")
            print("Start Ollama on Windows, then rerun this step.")
            return 1

    except Exception as exc:
        print(f"[FAIL] Cannot check Ollama: {exc}")
        return 1

    for name, context, expected_type, expected_notes in cases:
        try:
            result = service._semantic_for_context("TEST000", context)
            actual_type = result.get("vehicle_type")
            actual_notes = result.get("notes")
        except Exception as exc:
            print(f"[FAIL] {name}: EXCEPTION: {type(exc).__name__}: {exc}")
            failures += 1
            continue

        print(f"\n{name}")
        print(f"  CONTEXT : {context}")
        print(f"  RESULT  : {json.dumps(result, ensure_ascii=False)}")

        type_ok = actual_type == expected_type
        notes_ok = actual_notes == expected_notes

        print(f"  type  -> expected={expected_type!r}, actual={actual_type!r} "
              f"{'PASS' if type_ok else 'FAIL'}")
        print(f"  notes -> expected={expected_notes!r}, actual={actual_notes!r} "
              f"{'PASS' if notes_ok else 'FAIL'}")

        failures += int(not type_ok or not notes_ok)

    print(f"\nSemantic failures: {failures}")
    return failures


# ---------------------------------------------------------------------------
# STEP 5: ASR ONLY
# ---------------------------------------------------------------------------

def test_asr(audio_path: str) -> int:
    banner("STEP 5 — ASR ONLY (COHERE)")

    audio = Path(audio_path)
    if not audio.exists():
        print(f"[FAIL] Audio file not found: {audio}")
        return 1

    try:
        from app.services.stt import STTService
    except Exception as exc:
        print(f"[FAIL] Could not import STTService: {exc}")
        return 1

    print(f"Audio: {audio}")
    print("This test stops after STT. No plate parser, no Qwen, no API.\n")

    try:
        stt = STTService()
    except Exception as exc:
        print(f"[FAIL] STTService initialization failed: {exc}")
        print("This usually means the local model path/config/dependencies are missing.")
        return 1

    try:
        result = stt.transcribe(str(audio))
    except TypeError:
        # Some versions accept a Path.
        try:
            result = stt.transcribe(audio)
        except Exception as exc:
            print(f"[FAIL] STT transcribe failed: {exc}")
            return 1
    except Exception as exc:
        print(f"[FAIL] STT transcribe failed: {exc}")
        return 1

    print("RAW STT RESULT:")
    try:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    except Exception:
        print(repr(result))

    # Helpful normalized transcript view when the service returns
    # TranscriptSegment objects.
    segments = getattr(result, "segments", None)
    if segments:
        print("\nSEGMENTS:")
        for i, seg in enumerate(segments):
            print(
                f"[{i}] {getattr(seg, 'start', '?')} -> "
                f"{getattr(seg, 'end', '?')} : "
                f"{getattr(seg, 'text', '')}"
            )

    print(
        "\nASR step complete. Judge THIS output first. "
        "Do not look at the final JSON yet."
    )
    return 0


# ---------------------------------------------------------------------------
# STEP 6: PLATE PARSER ON RAW TEXT
# ---------------------------------------------------------------------------

def test_exact_transcripts() -> int:
    banner("STEP 6 — PLATE PARSER AGAINST EXACT TEXT (NO ASR)")

    # This is the cleanest way to determine whether the parser itself is
    # responsible for failures. We intentionally bypass Cohere.
    from app.services.plate_candidate import extract_plate_candidates
    from app.services.plate_normalizer import normalize_plate_from_text

    cases = [
        ("T2", "ر و ب ر و ب 1729", ["روب1729"]),
        ("AAA", "ا ا ا 1234", ["ااا1234"]),
        ("T1 first", "ب ط ص ب ط ص 5745 نقل", ["بطص5745"]),
        ("T3", "ا ع ا 1234 دوكر حادث ا س ي 1254", ["اعا1234", "اسي1254"]),
        ("T4 first", "ح ع ح 5077 ب ط ص 1234 نقل", ["حعح5077", "بطص1234"]),
    ]

    failures = 0

    for name, text, expected in cases:
        try:
            segments = [make_segment(text)]
            candidates = extract_plate_candidates(segments)
            raw = [c.text for c in candidates]
            actual = [normalize_plate_from_text(x) for x in raw]
        except Exception as exc:
            print(f"[FAIL] {name}: EXCEPTION: {type(exc).__name__}: {exc}")
            failures += 1
            continue

        print(f"\n{name}: {text}")
        print(f"  candidates: {raw}")
        print(f"  normalized : {actual}")

        ok(name, expected, actual)
        failures += int(actual != expected)

    print(f"\nExact-text parser failures: {failures}")
    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        choices=[
            "normalizer",
            "candidate",
            "contexts",
            "exact-text",
            "semantic",
            "asr",
            "all",
        ],
        required=True,
    )
    parser.add_argument(
        "--audio",
        help="Path to local audio file. Required for --step asr/all.",
    )
    args = parser.parse_args()

    total_failures = 0

    if args.step in ("normalizer", "all"):
        total_failures += test_normalizer()

    if args.step in ("candidate", "all"):
        total_failures += test_candidate()

    if args.step in ("exact-text", "all"):
        total_failures += test_exact_transcripts()

    if args.step in ("contexts", "all"):
        total_failures += test_contexts()

    if args.step in ("semantic", "all"):
        total_failures += test_semantic()

    if args.step == "asr":
        if not args.audio:
            print("--audio is required for --step asr")
            return 2
        return test_asr(args.audio)

    if args.step == "all":
        if args.audio:
            total_failures += test_asr(args.audio)
        else:
            print("\n[INFO] --audio not supplied; skipping ASR test.")

    print("\n" + "=" * 78)
    if total_failures == 0:
        print("ALL REQUESTED LOCAL TESTS PASSED")
    else:
        print(f"TOTAL FAILURES: {total_failures}")
    print("=" * 78)

    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
