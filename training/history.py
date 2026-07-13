"""Training artifact generation utilities for LeafFusionNet.

This module serializes Keras ``History`` objects and creates publication-ready
training curves without modifying model training behavior.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config.paths import FIGURES_DIR, REPORTS_DIR


_DPI = 300
_HISTORY_FILENAME = "training_history.json"
_SUMMARY_FILENAME = "training_summary.json"


def save_training_history(
    history: tf.keras.callbacks.History,
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    """Save all epoch metrics in a Keras History object as formatted JSON.

    Args:
        history: Completed Keras history returned by ``model.fit``.
        reports_dir: Destination directory for the history JSON file.

    Returns:
        Path to the saved ``training_history.json`` file.

    Raises:
        ValueError: If the history is missing or does not contain epoch metrics.
        OSError: If the report directory or JSON file cannot be written.
    """
    history_data = _get_history_data(history)
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / _HISTORY_FILENAME
    output_path.write_text(
        json.dumps(_to_json_value(history_data), indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def save_training_summary(
    history: tf.keras.callbacks.History,
    training_time_seconds: float,
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    """Calculate and save aggregate metrics from a completed training run.

    Args:
        history: Completed Keras history returned by ``model.fit``.
        training_time_seconds: Wall-clock duration of the ``model.fit`` call.
        reports_dir: Destination directory for the summary JSON file.

    Returns:
        Path to the saved ``training_summary.json`` file.

    Raises:
        ValueError: If expected training or validation metrics are unavailable,
            or if the supplied duration is negative.
        OSError: If the report directory or JSON file cannot be written.
    """
    if training_time_seconds < 0:
        raise ValueError("training_time_seconds must not be negative.")

    history_data = _get_history_data(history)
    loss = _get_metric(history_data, "loss")
    accuracy = _get_metric(history_data, "accuracy")
    validation_loss = _get_metric(history_data, "val_loss")
    validation_accuracy = _get_metric(history_data, "val_accuracy")
    _validate_metric_lengths(loss, accuracy, validation_loss, validation_accuracy)

    summary = {
        "epochs": len(loss),
        "best_epoch": int(np.argmax(validation_accuracy)) + 1,
        "best_train_accuracy": float(np.max(accuracy)),
        "best_validation_accuracy": float(np.max(validation_accuracy)),
        "best_validation_loss": float(np.min(validation_loss)),
        "final_train_accuracy": float(accuracy[-1]),
        "final_validation_accuracy": float(validation_accuracy[-1]),
        "training_time_seconds": float(training_time_seconds),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / _SUMMARY_FILENAME
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return output_path


def plot_accuracy_curve(
    history: tf.keras.callbacks.History,
    figures_dir: Path = FIGURES_DIR,
) -> Path:
    """Create a high-resolution training and validation accuracy curve.

    Args:
        history: Completed Keras history returned by ``model.fit``.
        figures_dir: Destination directory for the accuracy PNG file.

    Returns:
        Path to the saved ``accuracy_curve.png`` file.

    Raises:
        ValueError: If training or validation accuracy metrics are unavailable.
        OSError: If the figure directory or PNG file cannot be written.
    """
    history_data = _get_history_data(history)
    return _plot_training_validation_curve(
        training_values=_get_metric(history_data, "accuracy"),
        validation_values=_get_metric(history_data, "val_accuracy"),
        title="Training and Validation Accuracy",
        y_label="Accuracy",
        output_path=figures_dir / "accuracy_curve.png",
    )


def plot_loss_curve(
    history: tf.keras.callbacks.History,
    figures_dir: Path = FIGURES_DIR,
) -> Path:
    """Create a high-resolution training and validation loss curve.

    Args:
        history: Completed Keras history returned by ``model.fit``.
        figures_dir: Destination directory for the loss PNG file.

    Returns:
        Path to the saved ``loss_curve.png`` file.

    Raises:
        ValueError: If training or validation loss metrics are unavailable.
        OSError: If the figure directory or PNG file cannot be written.
    """
    history_data = _get_history_data(history)
    return _plot_training_validation_curve(
        training_values=_get_metric(history_data, "loss"),
        validation_values=_get_metric(history_data, "val_loss"),
        title="Training and Validation Loss",
        y_label="Loss",
        output_path=figures_dir / "loss_curve.png",
    )


def plot_learning_rate_curve(
    history: tf.keras.callbacks.History,
    figures_dir: Path = FIGURES_DIR,
) -> Path | None:
    """Create a learning-rate curve when the History object contains one.

    Args:
        history: Completed Keras history returned by ``model.fit``.
        figures_dir: Destination directory for the learning-rate PNG file.

    Returns:
        Path to ``learning_rate_curve.png`` when learning-rate history exists;
        otherwise ``None``.

    Raises:
        ValueError: If a present learning-rate metric has no values.
        OSError: If the figure directory or PNG file cannot be written.
    """
    history_data = _get_history_data(history)
    learning_rate = history_data.get("learning_rate", history_data.get("lr"))
    if learning_rate is None:
        return None
    if len(learning_rate) == 0:
        raise ValueError("Learning-rate history exists but contains no values.")

    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "learning_rate_curve.png"
    epochs = np.arange(1, len(learning_rate) + 1)
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(epochs, learning_rate, color="#6a4c93", linewidth=2, label="Learning Rate")
    axis.set(
        title="Learning Rate per Epoch",
        xlabel="Epoch",
        ylabel="Learning Rate",
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _get_history_data(history: tf.keras.callbacks.History) -> dict[str, list[Any]]:
    """Validate and return epoch metrics from a Keras History object."""
    if history is None:
        raise ValueError("history must not be None.")
    history_data = getattr(history, "history", None)
    if not isinstance(history_data, dict) or not history_data:
        raise ValueError("History object does not contain epoch metrics.")
    return history_data


def _get_metric(history_data: Mapping[str, Sequence[Any]], metric_name: str) -> list[float]:
    """Return one non-empty history metric as native Python floats."""
    values = history_data.get(metric_name)
    if values is None or len(values) == 0:
        raise ValueError(f"Training history does not contain '{metric_name}' values.")
    return [float(value) for value in values]


def _validate_metric_lengths(*metrics: Sequence[float]) -> None:
    """Ensure all required epoch metric sequences have matching lengths."""
    lengths = {len(metric) for metric in metrics}
    if len(lengths) != 1:
        raise ValueError("Training and validation metric histories have inconsistent lengths.")


def _plot_training_validation_curve(
    training_values: Sequence[float],
    validation_values: Sequence[float],
    title: str,
    y_label: str,
    output_path: Path,
) -> Path:
    """Save a consistently styled training-versus-validation metric curve."""
    _validate_metric_lengths(training_values, validation_values)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(training_values) + 1)
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(epochs, training_values, color="#1d3557", linewidth=2, label="Training")
    axis.plot(epochs, validation_values, color="#e63946", linewidth=2, label="Validation")
    axis.set(title=title, xlabel="Epoch", ylabel=y_label)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _to_json_value(value: Any) -> Any:
    """Convert NumPy and TensorFlow scalar values into JSON-compatible values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tf.Tensor):
        return _to_json_value(value.numpy())
    if isinstance(value, np.ndarray):
        return [_to_json_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    return value


__all__ = [
    "plot_accuracy_curve",
    "plot_learning_rate_curve",
    "plot_loss_curve",
    "save_training_history",
    "save_training_summary",
]
