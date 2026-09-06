import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(
    PROJECT_ROOT / ".env"
)


class Settings:

    APP_NAME = os.getenv(
        "APP_NAME",
        "multi-car-audio-processor",
    )

    APP_VERSION = os.getenv(
        "APP_VERSION",
        "1.0.0",
    )

    DEBUG = (
        os.getenv(
            "DEBUG",
            "true",
        ).lower()
        == "true"
    )

    STT_MODEL_PATH = os.getenv(
        "STT_MODEL_PATH",
        str(
            PROJECT_ROOT
            / "models"
            / "cohere-transcribe-arabic-07-2026"
        ),
    )

    STT_DEVICE = os.getenv(
        "STT_DEVICE",
        "cpu",
    )

    STT_COMPUTE_TYPE = os.getenv(
        "STT_COMPUTE_TYPE",
        "float32",
    )

    LLM_BASE_URL = os.getenv(
        "LLM_BASE_URL",
        "http://localhost:11434",
    )

    LLM_MODEL = os.getenv(
        "LLM_MODEL",
        "qwen2.5:7b",
    )

    LLM_TIMEOUT = float(
        os.getenv(
            "LLM_TIMEOUT",
            "900",
        )
    )

    MAX_AUDIO_SIZE_MB = int(
        os.getenv(
            "MAX_AUDIO_SIZE_MB",
            "100",
        )
    )


settings = Settings()