from pathlib import Path

import torch

from transformers import (
    AutoProcessor,
    CohereAsrForConditionalGeneration,
)

from transformers.audio_utils import load_audio

from app.core.config import settings
from app.models.transcript import TranscriptSegment


class SpeechToTextService:

    def __init__(self):

        self.model_path = Path(
            settings.STT_MODEL_PATH
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"STT model not found: "
                f"{self.model_path}"
            )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            and settings.STT_DEVICE != "cpu"
            else "cpu"
        )

        self.dtype = (
            torch.float16
            if self.device == "cuda"
            else torch.float32
        )

        print(
            f"Model path: {self.model_path}"
        )

        print(
            f"Device: {self.device}"
        )

        print(
            f"Dtype: {self.dtype}"
        )

        print()
        print(
            "Loading Cohere processor..."
        )

        self.processor = (
            AutoProcessor.from_pretrained(
                str(self.model_path)
            )
        )

        print(
            "Loading Cohere model..."
        )

        self.model = (
            CohereAsrForConditionalGeneration
            .from_pretrained(
                str(self.model_path),
                torch_dtype=self.dtype,
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    def transcribe(
        self,
        audio_path: str,
    ) -> list[TranscriptSegment]:

        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {path}"
            )

        # ----------------------------------------------------
        # LOAD AUDIO
        # ----------------------------------------------------

        audio = load_audio(
            str(path),
            sampling_rate=16000,
        )

        duration = (
            len(audio) / 16000.0
        )

        # ----------------------------------------------------
        # LONG-FORM PROCESSING
        # ----------------------------------------------------

        inputs = self.processor(
            audio=audio,
            sampling_rate=16000,
            return_tensors="pt",
            language="ar",
        )

        audio_chunk_index = inputs.get(
            "audio_chunk_index"
        )

        # Move tensor inputs to the model device.
        inputs = {
            key: (
                value.to(
                    self.device,
                    dtype=self.dtype,
                )
                if torch.is_floating_point(value)
                else value.to(self.device)
            )
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        with torch.inference_mode():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
            )

        # ----------------------------------------------------
        # DECODE LONG-FORM RESULT
        # ----------------------------------------------------

        if audio_chunk_index is not None:

            decoded = self.processor.decode(
                outputs,
                skip_special_tokens=True,
                audio_chunk_index=audio_chunk_index,
                language="ar",
            )

        else:

            decoded = self.processor.decode(
                outputs,
                skip_special_tokens=True,
            )

        if isinstance(
            decoded,
            list,
        ):
            text = " ".join(
                item.strip()
                for item in decoded
                if isinstance(item, str)
                and item.strip()
            )
        else:
            text = str(decoded).strip()

        if not text:
            return []

        # ----------------------------------------------------
        # IMPORTANT
        #
        # Cohere gives us the correct text structure, but this
        # path does not rely on word timestamps.
        #
        # Plate detection is based on token order.
        # ----------------------------------------------------

        return [
            TranscriptSegment(
                start=0.0,
                end=max(
                    duration,
                    0.001,
                ),
                text=text,
                words=[],
            )
        ]