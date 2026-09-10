from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from app.core.config import settings


@dataclass(frozen=True, slots=True)
class SpeechRegion:
    start_sample: int
    end_sample: int


class VADService:
    """Silero VAD wrapper used to preserve absolute audio timing."""

    def __init__(self) -> None:
        self.model = None
        self._get_speech_timestamps = None

    def load(self) -> None:
        try:
            from silero_vad import (
                get_speech_timestamps,
                load_silero_vad,
            )
        except ImportError as exc:
            raise RuntimeError(
                'silero-vad is not installed. '
                'Install the pinned project requirements.'
            ) from exc

        self.model = load_silero_vad(onnx=False)
        self._get_speech_timestamps = get_speech_timestamps

    @staticmethod
    def _merge_regions(
        regions: list[SpeechRegion],
        merge_gap_ms: int,
        total_samples: int,
        sample_rate: int,
    ) -> list[SpeechRegion]:
        if not regions:
            return []

        gap_samples = max(
            int(round(merge_gap_ms * sample_rate / 1000.0)),
            0,
        )

        merged: list[SpeechRegion] = []

        for region in regions:
            start = max(0, min(region.start_sample, total_samples))
            end = max(start, min(region.end_sample, total_samples))

            if end <= start:
                continue

            current = SpeechRegion(start, end)

            if not merged:
                merged.append(current)
                continue

            previous = merged[-1]
            gap = current.start_sample - previous.end_sample

            if gap <= gap_samples:
                merged[-1] = SpeechRegion(
                    previous.start_sample,
                    max(previous.end_sample, current.end_sample),
                )
            else:
                merged.append(current)

        return merged

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> list[SpeechRegion]:
        if self.model is None or self._get_speech_timestamps is None:
            raise RuntimeError('VADService is not loaded.')

        if sample_rate not in (8000, 16000):
            raise ValueError(
                f'Silero VAD expects 8 kHz or 16 kHz audio, got {sample_rate}.'
            )

        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim != 1:
            waveform = np.mean(waveform, axis=1)

        if waveform.size == 0:
            return []

        speech_timestamps = self._get_speech_timestamps(
            torch.from_numpy(waveform),
            self.model,
            sampling_rate=sample_rate,
            threshold=settings.vad_threshold,
            min_speech_duration_ms=settings.vad_min_speech_duration_ms,
            min_silence_duration_ms=settings.vad_min_silence_duration_ms,
            speech_pad_ms=settings.vad_speech_pad_ms,
            return_seconds=False,
        )

        regions = [
            SpeechRegion(
                start_sample=int(item['start']),
                end_sample=int(item['end']),
            )
            for item in speech_timestamps
        ]

        # Short pauses inside a spoken plate should not split one plate into
        # several ASR requests. Long pauses, however, are valuable because
        # they let us anchor later plates to a much tighter time window.
        return self._merge_regions(
            regions,
            merge_gap_ms=settings.vad_merge_gap_ms,
            total_samples=waveform.size,
            sample_rate=sample_rate,
        )
