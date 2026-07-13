"""One-time loading and accessors for LeafFusionNet deployment artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import tensorflow as tf

from .config import (
    CLASS_NAMES_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)

_model: tf.keras.Model | None = None
_class_names: tuple[str, ...] = ()
_metadata: dict[str, Any] = {}


def _read_json(path: Path) -> Any:
    """Read and parse a UTF-8 JSON artifact.

    Args:
        path: Path to the JSON artifact.

    Raises:
        RuntimeError: If the artifact cannot be read or parsed.
    """
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load artifact '{path}': {exc}") from exc

from pathlib import Path
import os

print("=" * 60)
print("Current working directory:", os.getcwd())
print("Project root exists:", PROJECT_ROOT)
print("Models dir exists:", (PROJECT_ROOT / "models").exists())

models_dir = PROJECT_ROOT / "models"
if models_dir.exists():
    print("Files inside models:")
    for f in models_dir.iterdir():
        print(" -", f.name)
else:
    print("Models directory does not exist")

print("=" * 60)

def load_artifacts() -> None:
    """Load the Keras model and its metadata once for the service lifetime.

    Raises:
        RuntimeError: If a required artifact is unavailable or invalid.
    """
    global _model, _class_names, _metadata

    if _model is not None:
        return

    required_paths = (MODEL_PATH, CLASS_NAMES_PATH, MODEL_METADATA_PATH)
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise RuntimeError(f"Required deployment artifacts are missing: {', '.join(missing_paths)}")

    logger.info("Loading LeafFusionNet model from %s", MODEL_PATH)
    try:
        class_names = _read_json(CLASS_NAMES_PATH)
        metadata = _read_json(MODEL_METADATA_PATH)
        if not isinstance(class_names, list) or not all(isinstance(item, str) for item in class_names):
            raise RuntimeError("class_names.json must contain a JSON array of strings.")
        if not isinstance(metadata, dict):
            raise RuntimeError("model_metadata.json must contain a JSON object.")

        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        output_classes = model.output_shape[-1]
        if output_classes != len(class_names):
            raise RuntimeError(
                "Model output class count does not match class_names.json "
                f"({output_classes} != {len(class_names)})."
            )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        logger.exception("LeafFusionNet model loading failed")
        raise RuntimeError(f"LeafFusionNet startup failed: {exc}") from exc

    _model = model
    _class_names = tuple(class_names)
    _metadata = metadata
    logger.info("LeafFusionNet model loaded successfully with %d classes", len(_class_names))


def get_model() -> tf.keras.Model:
    """Return the already-loaded model.

    Raises:
        RuntimeError: If startup loading has not completed.
    """
    if _model is None:
        raise RuntimeError("Model is not loaded.")
    return _model


def get_class_names() -> tuple[str, ...]:
    """Return the ordered class labels loaded at startup."""
    if not _class_names:
        raise RuntimeError("Class names are not loaded.")
    return _class_names


def get_metadata() -> dict[str, Any]:
    """Return a copy of model metadata loaded at startup."""
    if not _metadata:
        raise RuntimeError("Model metadata is not loaded.")
    return dict(_metadata)


def is_model_loaded() -> bool:
    """Return whether the model has completed startup loading."""
    return _model is not None
