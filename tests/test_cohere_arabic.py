import time
from pathlib import Path

import torch

from transformers import (
    AutoProcessor,
    CohereAsrForConditionalGeneration,
)

from transformers.audio_utils import load_audio


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "cohere-transcribe-arabic-07-2026"
)

AUDIO_PATH = (
    PROJECT_ROOT
    / "tests"
    / "sample4.m4a"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COHERE ARABIC ASR - LONG FORM TEST")
    print("=" * 70)

    print(f"Model: {MODEL_PATH}")
    print(f"Audio: {AUDIO_PATH}")

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_PATH}"
        )

    # ========================================================
    # DEVICE
    # ========================================================

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Device: {device}")
    print(f"Dtype:  {dtype}")

    # ========================================================
    # PROCESSOR
    # ========================================================

    print()
    print("Loading processor...")

    processor = (
        AutoProcessor.from_pretrained(
            str(MODEL_PATH)
        )
    )

    # ========================================================
    # MODEL
    # ========================================================

    print("Loading model...")

    model = (
        CohereAsrForConditionalGeneration
        .from_pretrained(
            str(MODEL_PATH),
            torch_dtype=dtype,
        )
    )

    model.to(device)
    model.eval()

    # ========================================================
    # AUDIO
    # ========================================================

    print()
    print("Loading audio...")

    audio = load_audio(
        str(AUDIO_PATH),
        sampling_rate=16000,
    )

    duration_s = (
        len(audio) / 16000
    )

    print(
        f"Audio duration: "
        f"{duration_s:.2f} seconds"
    )

    # ========================================================
    # PROCESS LONG AUDIO
    # ========================================================

    print()
    print("Preparing long-form audio...")
    print()

    inputs = processor(
        audio=audio,
        sampling_rate=16000,
        return_tensors="pt",
        language="ar",
    )

    # --------------------------------------------------------
    # VERY IMPORTANT
    #
    # Cohere provides this when the audio is split into
    # multiple chunks.
    # --------------------------------------------------------

    audio_chunk_index = (
        inputs.get("audio_chunk_index")
    )

    # Move tensors to device.
    inputs = {
        key: value.to(device)
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }

    if audio_chunk_index is not None:

        print(
            f"Detected "
            f"{len(audio_chunk_index)} "
            f"audio chunks."
        )

    else:

        print(
            "No chunk index returned."
        )

    # ========================================================
    # INFERENCE
    # ========================================================

    print()
    print("Transcribing...")
    print("This may take several minutes on CPU.")
    print()

    start_time = time.time()

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
        )

    elapsed = (
        time.time()
        - start_time
    )

    # ========================================================
    # DECODE
    # ========================================================

    if audio_chunk_index is not None:

        decoded = processor.decode(
            outputs,
            skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index,
            language="ar",
        )

    else:

        decoded = processor.decode(
            outputs,
            skip_special_tokens=True,
        )

    # Cohere's long-form decoder can return a list.
    if isinstance(decoded, list):

        text = "\n".join(
            item
            for item in decoded
            if isinstance(item, str)
        )

    else:

        text = str(decoded)

    text = text.strip()

    # ========================================================
    # RESULT
    # ========================================================

    print("=" * 70)
    print("COMPLETE TRANSCRIPT")
    print("=" * 70)
    print()

    print(text)

    print()
    print("=" * 70)
    print("PERFORMANCE")
    print("=" * 70)

    print(
        f"Audio duration : "
        f"{duration_s:.2f} sec"
    )

    print(
        f"Inference time : "
        f"{elapsed:.2f} sec"
    )

    if elapsed > 0:

        print(
            f"RTFx           : "
            f"{duration_s / elapsed:.2f}"
        )


if __name__ == "__main__":
    main()