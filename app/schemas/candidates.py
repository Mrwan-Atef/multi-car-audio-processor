from pydantic import BaseModel, Field


class PlateCandidate(BaseModel):
    text: str = Field(min_length=1)

    start: float = Field(ge=0)
    end: float = Field(gt=0)

    segment_index: int = Field(ge=0)

    token_start: int = Field(ge=0)
    token_end: int = Field(ge=0)


class SuspiciousCandidate(BaseModel):
    text: str = Field(min_length=1)

    start: float = Field(ge=0)
    end: float = Field(gt=0)

    segment_index: int = Field(ge=0)

    token_start: int = Field(ge=0)
    token_end: int = Field(ge=0)

    reason: str = Field(min_length=1)