import time
from pathlib import Path

import torch

from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "whisper-large-v3-arabic-dialectal-v2"
)

AUDIO_PATH = (
    PROJECT_ROOT
    / "tests"
    / "sample2.aac"
)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("ARABIC DIALECTAL WHISPER TEST")
    print("=" * 70)

    print(f"Model: {MODEL_PATH}")
    print(f"Audio: {AUDIO_PATH}")

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_PATH}"
        )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = "cuda:0"
        torch_dtype = torch.float16

    else:

        device = "cpu"
        torch_dtype = torch.float32

    print(f"Device: {device}")

    # --------------------------------------------------------
    # PROCESSOR
    # --------------------------------------------------------

    print()
    print("Loading processor...")

    processor = (
        AutoProcessor.from_pretrained(
            str(MODEL_PATH)
        )
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("Loading model...")

    model = (
        AutoModelForSpeechSeq2Seq
        .from_pretrained(
            str(MODEL_PATH),
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
    )

    model.to(device)
    model.eval()

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    print("Creating ASR pipeline...")

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0 if device.startswith("cuda") else -1,
        torch_dtype=torch_dtype,
    )

    # --------------------------------------------------------
    # TRANSCRIBE
    # --------------------------------------------------------

    print()
    print("Transcribing...")
    print()

    start_time = time.time()

    result = asr(
        str(AUDIO_PATH),
        chunk_length_s=30,
        batch_size=1,
        generate_kwargs={
            "language": "ar",
            "task": "transcribe",
        },
    )

    elapsed = (
        time.time()
        - start_time
    )

    text = result["text"]

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print("=" * 70)
    print("TRANSCRIPT")
    print("=" * 70)

    print(text.strip())

    print()
    print("=" * 70)
    print("PERFORMANCE")
    print("=" * 70)

    print(
        f"Inference time: {elapsed:.2f} sec"
    )


if __name__ == "__main__":
    main()