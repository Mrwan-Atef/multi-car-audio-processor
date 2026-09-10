from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Vehicle(BaseModel):
    model_config = ConfigDict(extra='ignore')

    plate_number: str

    # Representative point in the plate utterance. This is the midpoint of
    # timestamp_start_ms and timestamp_end_ms.
    timestamp_ms: int = Field(
        ...,
        ge=0,
        description='Estimated plate-event time in milliseconds from recording start.',
    )
    timestamp_start_ms: int = Field(
        ...,
        ge=0,
        description='Estimated start of the plate utterance in milliseconds from recording start.',
    )
    timestamp_end_ms: int = Field(
        ...,
        ge=0,
        description='Estimated end of the plate utterance in milliseconds from recording start.',
    )

    vehicle_type: str | None = None
    notes: str | None = None


class VehicleExtractionResponse(BaseModel):
    model_config = ConfigDict(extra='ignore')

    recording_duration_ms: int = Field(
        0,
        ge=0,
        description='Full recording duration in milliseconds.',
    )
    timestamp_reference: Literal['recording_start'] = 'recording_start'
    cars: list[Vehicle] = Field(default_factory=list)
