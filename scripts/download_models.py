# from pathlib import Path

# from huggingface_hub import snapshot_download


# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# MODELS_DIR = PROJECT_ROOT / "models"

# WHISPER_REPO = "Systran/faster-whisper-large-v3"


# def download_whisper() -> None:
#     destination = MODELS_DIR / "faster-whisper-large-v3"

#     if destination.exists() and (destination / "model.bin").exists():
#         print("=" * 60)
#         print("Faster-Whisper large-v3 is already downloaded.")
#         print("=" * 60)
#         print(f"Location: {destination}")
#         return

#     print("=" * 60)
#     print("Downloading Faster-Whisper large-v3")
#     print("=" * 60)

#     snapshot_download(
#         repo_id=WHISPER_REPO,
#         local_dir=str(destination),
#     )

#     print("=" * 60)
#     print("Download complete")
#     print("=" * 60)
#     print(f"Model location: {destination}")


# if __name__ == "__main__":
#     MODELS_DIR.mkdir(parents=True, exist_ok=True)
#     download_whisper()

import shutil
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

WHISPER_REPO = "Systran/faster-whisper-large-v3"
WHISPER_DIR = MODELS_DIR / "faster-whisper-large-v3"

# Local development model.
# This can be changed later for production.
OLLAMA_MODEL = "qwen2.5:7b"


def download_whisper() -> None:
    """
    Download Faster-Whisper large-v3 into the project models directory.
    """

    if (WHISPER_DIR / "model.bin").exists():
        print("\nFaster-Whisper large-v3 already exists.")
        print(f"Location: {WHISPER_DIR}")
        return

    print("\n" + "=" * 60)
    print("Downloading Faster-Whisper large-v3")
    print("=" * 60)

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_download(
        repo_id=WHISPER_REPO,
        local_dir=str(WHISPER_DIR),
    )

    print("\nWhisper download complete.")
    print(f"Location: {WHISPER_DIR}")


def check_ollama() -> None:
    """
    Make sure Ollama is installed and available in PATH.
    """

    if shutil.which("ollama") is None:
        raise RuntimeError(
            "Ollama was not found in PATH.\n"
            "Install Ollama and restart the terminal before "
            "running this script again."
        )


def download_llm() -> None:
    """
    Download the local Qwen model using Ollama.
    """

    check_ollama()

    print("\n" + "=" * 60)
    print(f"Downloading LLM: {OLLAMA_MODEL}")
    print("=" * 60)

    result = subprocess.run(
        [
            "ollama",
            "pull",
            OLLAMA_MODEL,
        ],
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to download Ollama model: {OLLAMA_MODEL}"
        )

    print("\nLLM download complete.")
    print(f"Model: {OLLAMA_MODEL}")


def main() -> None:
    print("=" * 60)
    print("MULTI-CAR AUDIO PROCESSOR - MODEL SETUP")
    print("=" * 60)

    download_whisper()
    download_llm()

    print("\n" + "=" * 60)
    print("ALL MODELS ARE READY")
    print("=" * 60)


if __name__ == "__main__":
    main()