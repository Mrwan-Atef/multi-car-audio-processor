from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = os.getenv(
    "STT_MODEL_ID",
    "CohereLabs/cohere-transcribe-arabic-07-2026",
)
MODEL_PATH = Path(
    os.getenv(
        "STT_MODEL_PATH",
        "/app/models/cohere-transcribe-arabic-07-2026",
    )
)
HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
    or None
)


def main() -> None:
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Preparing Cohere Arabic STT model")
    print("=" * 72)
    print(f"Model ID : {MODEL_ID}")
    print(f"Local dir: {MODEL_PATH}")

    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(MODEL_PATH),
            token=HF_TOKEN,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to download the STT model. "
            "Check internet access and, if required, set HF_TOKEN "
            "after accepting the model access terms on Hugging Face."
        ) from exc

    print("STT model is ready.")


if __name__ == "__main__":
    main()
