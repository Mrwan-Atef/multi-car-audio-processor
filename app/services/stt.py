from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

from app.core.config import settings
from app.models.transcript import TranscriptSegment, TranscriptResponse
from app.services.audio import convert_to_pcm_wav


class STTService:
    def __init__(self) -> None:
        self.processor = None
        self.model = None

    def load(self) -> None:
        model_path = settings.model_path

        if not model_path.exists():
            raise FileNotFoundError(
                f"STT model not found: {model_path}"
            )

        self.processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=True,
        )

        device = settings.stt_device.lower()

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "STT_DEVICE is set to cuda, but CUDA is not available."
            )

        target_device = "cuda" if device == "cuda" else "cpu"

        dtype = (
            torch.bfloat16
            if target_device == "cuda"
            else torch.float32
        )

        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            str(model_path),
            local_files_only=True,
            torch_dtype=dtype,
            device_map="auto" if target_device == "cuda" else None,
        )

        if target_device == "cpu":
            self.model.to("cpu")

        self.model.eval()

    def transcribe(self, audio_path: Path) -> TranscriptResponse:
        if self.processor is None or self.model is None:
            raise RuntimeError("STT model is not loaded.")

        wav_path = audio_path.with_suffix(".16k.wav")

        convert_to_pcm_wav(
            audio_path,
            wav_path,
            settings.audio_sample_rate,
        )

        audio, sample_rate = sf.read(
            str(wav_path),
            dtype="float32",
            always_2d=False,
        )

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        audio = np.asarray(audio, dtype=np.float32)

        inputs = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            language="ar",
        )

        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor):
                value = value.to(device=device)

                if value.is_floating_point():
                    value = value.to(dtype=dtype)

                inputs[key] = value

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=settings.stt_max_new_tokens,
            )

        audio_chunk_index = inputs.get("audio_chunk_index")

        text = self.processor.decode(
            output_ids,
            skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index,
            language="ar",
        )[0]

        text = text.strip()

        duration = len(audio) / float(sample_rate)

        return TranscriptResponse(
            text=text,
            segments=[
                TranscriptSegment(
                    text=text,
                    start=0.0,
                    end=duration,
                    words=[],
                )
            ],
        )