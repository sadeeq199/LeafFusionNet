"""Deployment metadata generation utilities for LeafFusionNet."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import keras
import tensorflow as tf

from config.hyperparameters import (
    BATCH_SIZE,
    CLASSIFIER_ACTIVATION,
    IMG_SIZE,
    INITIAL_EPOCHS,
    INITIAL_LEARNING_RATE,
    LOSS_FUNCTION,
    OPTIMIZER,
    RANDOM_SEED,
)
from config.paths import PROJECT_ROOT, REPORTS_DIR, TRAIN_DIR, VALID_DIR


_IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png"})
_MODEL_VERSION = "1.0.0"
_PROJECT_NAME = "LeafFusionNet"
_ARCHITECTURE_NAME = "Dual-Branch CNN with Feature Fusion"
_DATASET_NAME = "PlantVillage"


def generate_class_names(
    train_dir: Path = TRAIN_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    """Save training class names in TensorFlow directory-loader index order.

    Args:
        train_dir: Training dataset root containing one directory per class.
        reports_dir: Directory where ``class_names.json`` is written.

    Returns:
        Path to the generated class-name JSON file.

    Raises:
        FileNotFoundError: If the training dataset directory is missing.
        ValueError: If no class directories are found.
        OSError: If the JSON output cannot be written.
    """
    class_names = _get_class_names(train_dir)
    return _write_json(reports_dir / "class_names.json", class_names)


def generate_model_metadata(
    model: tf.keras.Model,
    saved_model_path: Path,
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    """Save deployment-facing metadata for a saved Keras classification model.

    Args:
        model: Trained Keras model that has been saved successfully.
        saved_model_path: Path to the saved Keras model file.
        reports_dir: Directory where ``model_metadata.json`` is written.

    Returns:
        Path to the generated model-metadata JSON file.

    Raises:
        ValueError: If the model has no defined input or output class shape.
        FileNotFoundError: If the supplied saved-model file is missing.
        OSError: If the JSON output cannot be written.
    """
    if model is None:
        raise ValueError("model must not be None.")
    if not saved_model_path.is_file():
        raise FileNotFoundError(f"Saved model does not exist: {saved_model_path}")

    input_shape = model.input_shape
    output_shape = model.output_shape
    if isinstance(input_shape, list) or len(input_shape) < 2:
        raise ValueError("Model must expose one input shape with feature dimensions.")
    if isinstance(output_shape, list) or len(output_shape) < 2 or output_shape[-1] is None:
        raise ValueError("Model must expose one output shape with a class dimension.")

    metadata = {
        "model_name": model.name,
        "model_version": _MODEL_VERSION,
        "framework": "TensorFlow",
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "python_version": platform.python_version(),
        "input_shape": [int(dimension) for dimension in input_shape[1:]],
        "num_classes": int(output_shape[-1]),
        "classifier_activation": _get_classifier_activation(model),
        "saved_model": _relative_project_path(saved_model_path),
    }
    return _write_json(reports_dir / "model_metadata.json", metadata)


def generate_dataset_summary(
    train_dir: Path = TRAIN_DIR,
    valid_dir: Path = VALID_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    """Compute and save image and class counts for training and validation data.

    Args:
        train_dir: Training dataset root containing class directories.
        valid_dir: Validation dataset root containing class directories.
        reports_dir: Directory where ``dataset_summary.json`` is written.

    Returns:
        Path to the generated dataset-summary JSON file.

    Raises:
        FileNotFoundError: If a dataset directory is missing.
        ValueError: If either dataset has no class directories.
        OSError: If the JSON output cannot be written.
    """
    train_classes = _get_class_names(train_dir)
    validation_classes = _get_class_names(valid_dir)
    summary = {
        "train_images": _count_images(train_dir),
        "validation_images": _count_images(valid_dir),
        "train_classes": len(train_classes),
        "validation_classes": len(validation_classes),
        "image_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "random_seed": RANDOM_SEED,
    }
    return _write_json(reports_dir / "dataset_summary.json", summary)


def generate_experiment_metadata(
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    """Save configuration metadata describing the LeafFusionNet experiment.

    Args:
        reports_dir: Directory where ``experiment.json`` is written.

    Returns:
        Path to the generated experiment-metadata JSON file.

    Raises:
        OSError: If the JSON output cannot be written.
    """
    experiment = {
        "project": _PROJECT_NAME,
        "architecture": _ARCHITECTURE_NAME,
        "dataset": _DATASET_NAME,
        "framework": "TensorFlow",
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "python": platform.python_version(),
        "optimizer": OPTIMIZER,
        "loss": LOSS_FUNCTION,
        "learning_rate": INITIAL_LEARNING_RATE,
        "epochs": INITIAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": RANDOM_SEED,
    }
    return _write_json(reports_dir / "experiment.json", experiment)


def _get_class_names(dataset_dir: Path) -> list[str]:
    """Return alphabetically ordered class directories used by Keras loaders."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    class_names = sorted(
        (path.name for path in dataset_dir.iterdir() if path.is_dir()),
    )
    if not class_names:
        raise ValueError(f"No class directories found in dataset: {dataset_dir}")
    return class_names


def _count_images(dataset_dir: Path) -> int:
    """Count supported image files below a validated dataset directory."""
    return sum(
        1
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )


def _get_classifier_activation(model: tf.keras.Model) -> str:
    """Return the classifier activation name, with a configuration fallback."""
    for layer in reversed(model.layers):
        activation = getattr(layer, "activation", None)
        if activation is not None:
            return str(getattr(activation, "__name__", CLASSIFIER_ACTIVATION))
    return CLASSIFIER_ACTIVATION


def _relative_project_path(path: Path) -> str:
    """Return a project-relative POSIX path when possible."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_json(output_path: Path, content: Any) -> Path:
    """Write JSON content with consistent UTF-8 encoding and formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    return output_path


__all__ = [
    "generate_class_names",
    "generate_dataset_summary",
    "generate_experiment_metadata",
    "generate_model_metadata",
]
