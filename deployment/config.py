"""Configuration for the LeafFusionNet deployment service."""

from __future__ import annotations

import os
from pathlib import Path


DEPLOYMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOYMENT_DIR.parent

MODEL_PATH = PROJECT_ROOT / "models" / "LeafFusionNet.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "results" / "reports" / "class_names.json"
MODEL_METADATA_PATH = PROJECT_ROOT / "results" / "reports" / "model_metadata.json"

IMAGE_SIZE: tuple[int, int] = (224, 224)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

# Comma-separated exact origins can be supplied by Render, for example:
# CORS_ORIGINS=https://my-app.vercel.app,https://www.example.com
_configured_origins = os.getenv("CORS_ORIGINS", "")
CORS_ORIGINS = [
    origin.strip()
    for origin in _configured_origins.split(",")
    if origin.strip()
]
if not CORS_ORIGINS:
    CORS_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

# Allows Vercel preview and production deployments. Configure CORS_ORIGINS as
# well when a custom frontend domain is used.
VERCEL_ORIGIN_REGEX = r"https://([a-zA-Z0-9-]+\.)*vercel\.app"

