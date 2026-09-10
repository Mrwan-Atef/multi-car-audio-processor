from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from app.core.config import settings
from app.models.transcript import TranscriptSegment, TranscriptResponse
from app.services.audio import convert_to_pcm_wav
from app.services.vad import SpeechRegion, VADService


class STTService:
    def __init__(self) -> None:
        self.processor = None
        self.model = None
        self.vad = VADService()
        self.target_device = 'cpu'
        self.model_dtype = torch.float32

    def _select_cuda_dtype(self) -> torch.dtype:
        """Use BF16 on modern GPUs and FP16 on older GPUs such as T4."""
        major, _minor = torch.cuda.get_device_capability(0)
        if major >= 8:
            return torch.bfloat16
        return torch.float16

    def load(self) -> None:
        model_path = settings.model_path

        if not model_path.exists():
            raise FileNotFoundError(
                f'STT model not found: {model_path}. '
                'Run the model download step first.'
            )

        from transformers import (
            AutoProcessor,
            CohereAsrForConditionalGeneration,
        )

        self.processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=True,
        )

        device = settings.stt_device.lower().strip()

        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(
                'STT_DEVICE=cuda but CUDA is not available inside the container.'
            )

        self.target_device = 'cuda' if device == 'cuda' else 'cpu'
        self.model_dtype = (
            self._select_cuda_dtype()
            if self.target_device == 'cuda'
            else torch.float32
        )

        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            str(model_path),
            local_files_only=True,
            torch_dtype=self.model_dtype,
            device_map='auto' if self.target_device == 'cuda' else None,
        )

        if self.target_device == 'cpu':
            self.model.to('cpu')

        self.model.eval()

        if settings.vad_enabled:
            self.vad.load()

    def _transcribe_waveform(self, audio: np.ndarray, sample_rate: int) -> str:
        if self.processor is None or self.model is None:
            raise RuntimeError('STTService is not loaded.')

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            audio = np.mean(audio, axis=1)

        if audio.size == 0:
            return ''

        inputs = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors='pt',
            language='ar',
        )

        device = next(self.model.parameters()).device
        inputs = inputs.to(device=device)

        # Only floating tensors should be converted to the model dtype.
        # Metadata such as audio_chunk_index can remain integer tensors.
        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                inputs[key] = value.to(
                    device=device,
                    dtype=self.model_dtype,
                )

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=settings.stt_max_new_tokens,
            )

        audio_chunk_index = inputs.get('audio_chunk_index')

        decoded = self.processor.decode(
            output_ids,
            skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index,
            language='ar',
        )

        if isinstance(decoded, list):
            return str(decoded[0]).strip()

        return str(decoded).strip()

    @staticmethod
    def _region_seconds(
        region: SpeechRegion,
        sample_rate: int,
    ) -> tuple[float, float]:
        return (
            region.start_sample / float(sample_rate),
            region.end_sample / float(sample_rate),
        )

    def _detect_regions(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> list[SpeechRegion]:
        if not settings.vad_enabled:
            return [SpeechRegion(0, int(audio.size))]

        return self.vad.detect(audio, sample_rate)

    def transcribe(self, audio_path: Path) -> TranscriptResponse:
        if self.processor is None or self.model is None:
            raise RuntimeError('STTService is not loaded.')

        wav_path = audio_path.with_suffix('.16k.wav')
        convert_to_pcm_wav(
            audio_path,
            wav_path,
            settings.audio_sample_rate,
        )

        audio, sample_rate = sf.read(
            str(wav_path),
            dtype='float32',
            always_2d=False,
        )

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        audio = np.asarray(audio, dtype=np.float32)
        duration_ms = int(round(len(audio) * 1000.0 / sample_rate))

        regions = self._detect_regions(audio, sample_rate)

        # No detected speech means no transcript. This is preferable to
        # allowing an AED ASR model to hallucinate text from silence/noise.
        if not regions:
            return TranscriptResponse(
                text='',
                duration_ms=duration_ms,
                segments=[],
            )

        segments: list[TranscriptSegment] = []

        for region in regions:
            start = max(0, min(region.start_sample, len(audio)))
            end = max(start, min(region.end_sample, len(audio)))

            if end <= start:
                continue

            region_audio = audio[start:end]
            text = self._transcribe_waveform(
                region_audio,
                sample_rate,
            )

            if not text:
                continue

            start_seconds, end_seconds = self._region_seconds(
                region,
                sample_rate,
            )

            segments.append(
                TranscriptSegment(
                    text=text,
                    start=start_seconds,
                    end=end_seconds,
                    words=[],
                )
            )

        combined_text = ' '.join(
            segment.text
            for segment in segments
            if segment.text
        ).strip()

        return TranscriptResponse(
            text=combined_text,
            duration_ms=duration_ms,
            segments=segments,
        )
