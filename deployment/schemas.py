"""Pydantic response schemas for the LeafFusionNet API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, RootModel


class APIModel(BaseModel):
    """Base API model with JSON serialization configured for responses."""

    model_config = ConfigDict(extra="forbid")


class ServiceStatusResponse(APIModel):
    """Service identity and availability response."""

    project: str
    status: str


class HealthResponse(APIModel):
    """Service health response."""

    status: str
    model_loaded: bool


class ClassesResponse(APIModel):
    """Ordered model class labels response."""

    classes: list[str]


class ModelInfoResponse(RootModel[dict[str, Any]]):
    """Model metadata loaded directly from the model metadata artifact."""


class PredictionItem(APIModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True
    )

    class_name: str = Field(
        serialization_alias="class"
    )

    confidence: float = Field(
        ge=0.0,
        le=100.0
    )


class PredictionResponse(APIModel):
    """Prediction result for one uploaded image, enriched with disease info."""

    prediction: str
    confidence: float = Field(ge=0.0, le=100.0)
    display_name: str
    crop: str
    severity: str
    description: str
    treatment: list[str]
    prevention: list[str]
    top5: list[PredictionItem]
    inference_time_ms: float = Field(ge=0.0)


class ErrorResponse(APIModel):
    """Standard JSON error response."""

    detail: str

