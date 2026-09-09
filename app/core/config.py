from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "multi-car-audio-processor"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    stt_model_id: str = "CohereLabs/cohere-transcribe-arabic-07-2026"
    stt_model_path: str = "models/cohere-transcribe-arabic-07-2026"
    stt_device: str = "cpu"
    stt_max_new_tokens: int = 256

    hf_token: str | None = None

    ollama_base_url: str = "http://ollama:11434"
    llm_model: str = "qwen2.5:7b"
    llm_timeout_seconds: float = 600.0

    upload_dir: str = "/tmp/ai-service/uploads"
    max_upload_mb: int = 200

    audio_sample_rate: int = 16000

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def model_path(self) -> Path:
        path = Path(self.stt_model_path)

        if path.is_absolute():
            return path

        return self.project_root / path


settings = Settings()