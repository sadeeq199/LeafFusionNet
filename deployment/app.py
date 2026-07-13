"""FastAPI application for serving the trained LeafFusionNet model."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import CORS_ORIGINS, MAX_UPLOAD_BYTES, SUPPORTED_EXTENSIONS, VERCEL_ORIGIN_REGEX
from .model_loader import get_class_names, get_metadata, is_model_loaded, load_artifacts
from .predictor import InvalidImageError, predict_image
from .schemas import (
    ClassesResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ServiceStatusResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load all deployment artifacts exactly once during application startup."""
    logger.info("Starting LeafFusionNet API")
    load_artifacts()
    yield
    logger.info("Stopping LeafFusionNet API")


app = FastAPI(
    title="LeafFusionNet API",
    description="Production inference API for LeafFusionNet plant disease classification.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=VERCEL_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _error_response(status_code: int, detail: str) -> JSONResponse:
    """Build a consistent JSON error response."""
    return JSONResponse(status_code=status_code, content=ErrorResponse(detail=detail).model_dump())


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, __: RequestValidationError) -> JSONResponse:
    """Return a concise JSON error for invalid request bodies."""
    return _error_response(status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid request. Provide an image in the 'file' field.")


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Return consistent JSON errors for HTTP and missing-route errors."""
    detail = str(exc.detail) if exc.detail else "Request failed."
    return _error_response(exc.status_code, detail)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """Log unexpected errors without exposing internal details to clients."""
    logger.exception("Unhandled API error: %s", exc)
    return _error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, "An internal server error occurred.")


@app.get("/", response_model=ServiceStatusResponse, tags=["Service"])
async def root() -> ServiceStatusResponse:
    """Return the API service identity and current status."""
    return ServiceStatusResponse(project="LeafFusionNet", status="running")


@app.get("/health", response_model=HealthResponse, tags=["Service"])
async def health() -> HealthResponse:
    """Report whether the model was loaded successfully during startup."""
    return HealthResponse(status="healthy" if is_model_loaded() else "unhealthy", model_loaded=is_model_loaded())


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info() -> ModelInfoResponse:
    """Return the model metadata artifact loaded during startup."""
    return ModelInfoResponse(get_metadata())


@app.get("/classes", response_model=ClassesResponse, tags=["Model"])
async def classes() -> ClassesResponse:
    """Return the ordered class names used by model output indices."""
    return ClassesResponse(classes=list(get_class_names()))


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Prediction"],
)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Validate an image upload and return its top-five disease predictions."""
    filename = file.filename or ""
    logger.info("Prediction request received for file '%s'", filename)
    extension = filename.lower().rsplit(".", 1)
    suffix = f".{extension[-1]}" if len(extension) == 2 else ""
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Only JPG, JPEG, and PNG files are accepted.",
        )

    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()

    if not contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The uploaded file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )

    try:
        result = predict_image(contents)
    except InvalidImageError as exc:
        logger.warning("Invalid image received for file '%s': %s", filename, exc)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("Prediction failed for file '%s'", filename)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction could not be completed.") from exc

    logger.info("Prediction completed for file '%s': %s", filename, result.prediction)
    return result
