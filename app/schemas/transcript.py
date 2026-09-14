from pydantic import BaseModel, ConfigDict, Field


class TranscriptWord(BaseModel):
    model_config = ConfigDict(extra='ignore')

    text: str
    start: float
    end: float


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(extra='ignore')

    text: str
    start: float
    end: float
    words: list[TranscriptWord] = Field(default_factory=list)


class TranscriptResponse(BaseModel):
    model_config = ConfigDict(extra='ignore')

    text: str
    # Full source recording duration, relative to the recording start.
    duration_ms: int = 0
    segments: list[TranscriptSegment] = Field(default_factory=list)
