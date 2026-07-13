"""Generate reproducible test-set evaluation reports for LeafFusionNet.

Run this module from the project root with ``python -m evaluation.generate_reports``.
It loads the best Keras checkpoint and writes metrics, per-image predictions,
model summary, and publication-quality matplotlib figures under ``results``.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from config.hyperparameters import NUM_CLASSES
from config.paths import CHECKPOINTS_DIR, FIGURES_DIR, REPORTS_DIR, TEST_DIR
from data.dataset import build_test_dataset

BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.keras"
_DPI = 300
_IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}


def validate_evaluation_inputs(
    model_path: Path = BEST_MODEL_PATH,
    test_dir: Path = TEST_DIR,
) -> None:
    """Validate filesystem inputs required for standalone evaluation.

    Args:
        model_path: Path to the serialized Keras model checkpoint.
        test_dir: Directory containing one subdirectory per test class.

    Raises:
        FileNotFoundError: If the model checkpoint or test directory is absent.
        ValueError: If the test directory does not contain ``NUM_CLASSES``
            class directories.
    """
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test dataset directory does not exist: {test_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Best model checkpoint does not exist: {model_path}")

    class_directories = [path for path in test_dir.iterdir() if path.is_dir()]
    if len(class_directories) != NUM_CLASSES:
        raise ValueError(
            f"Test dataset has {len(class_directories)} class directories; "
            f"expected NUM_CLASSES={NUM_CLASSES}."
        )


def collect_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect true labels, predicted labels, and probabilities for a dataset.

    Args:
        model: Loaded Keras classification model.
        dataset: Deterministic labeled test dataset.

    Returns:
        A tuple containing true label indices, predicted label indices, and the
        full class-probability matrix.

    Raises:
        ValueError: If the model output does not have ``NUM_CLASSES`` columns
            or the dataset contains no images.
    """
    true_labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for images, labels in dataset:
        batch_probabilities = np.asarray(model.predict(images, verbose=0))
        if batch_probabilities.ndim != 2 or batch_probabilities.shape[1] != NUM_CLASSES:
            raise ValueError(
                "Model output must have shape (batch_size, NUM_CLASSES); "
                f"received {batch_probabilities.shape}."
            )
        true_labels.append(np.asarray(labels, dtype=np.int64).reshape(-1))
        probabilities.append(batch_probabilities)

    if not true_labels:
        raise ValueError("Test dataset contains no images to evaluate.")

    y_true = np.concatenate(true_labels)
    y_probabilities = np.concatenate(probabilities)
    return y_true, np.argmax(y_probabilities, axis=1), y_probabilities


def get_test_metadata(test_dir: Path = TEST_DIR) -> tuple[list[str], list[str]]:
    """Return the class names and image paths in Keras directory-loader order.

    Args:
        test_dir: Root test directory containing one directory per class.

    Returns:
        A tuple of alphabetically ordered class names and their alphabetically
        ordered image paths, grouped by class.

    Raises:
        ValueError: If a class directory contains no supported image files.
    """
    class_paths = sorted(path for path in test_dir.iterdir() if path.is_dir())
    class_names = [path.name for path in class_paths]
    image_paths: list[str] = []
    for class_path in class_paths:
        files = sorted(
            path for path in class_path.rglob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not files:
            raise ValueError(f"Test class directory contains no supported images: {class_path}")
        image_paths.extend(str(path) for path in files)
    return class_names, image_paths


def build_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Calculate scalar and aggregate classification metrics.

    Args:
        y_true: Ground-truth integer class labels.
        y_pred: Predicted integer class labels.

    Returns:
        JSON-serializable accuracy, macro-average, and weighted-average metrics.
    """
    metrics: dict[str, Any] = {"accuracy": float(accuracy_score(y_true, y_pred))}
    for average, key in (("macro", "macro_average"), ("weighted", "weighted_average")):
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        metrics[key] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
        }
    metrics["precision"] = metrics["weighted_average"]["precision"]
    metrics["recall"] = metrics["weighted_average"]["recall"]
    metrics["f1_score"] = metrics["weighted_average"]["f1_score"]
    return metrics


def save_text_reports(
    model: tf.keras.Model,
    class_names: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict[str, Any],
    reports_dir: Path = REPORTS_DIR,
) -> None:
    """Write classification, JSON metrics, and Keras summary report files.

    Args:
        model: Loaded Keras model whose summary is saved.
        class_names: Class names ordered by label index.
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        metrics: JSON-serializable aggregate metrics.
        reports_dir: Target directory for the report files.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, zero_division=0
    )
    (reports_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    summary_stream = io.StringIO()
    model.summary(print_fn=lambda line: summary_stream.write(f"{line}\n"))
    (reports_dir / "model_summary.txt").write_text(summary_stream.getvalue(), encoding="utf-8")


def save_predictions_csv(
    image_paths: Sequence[str],
    class_names: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    reports_dir: Path = REPORTS_DIR,
) -> None:
    """Save a per-image prediction report as a CSV file.

    Args:
        image_paths: Deterministically ordered test image paths.
        class_names: Class names ordered by label index.
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        probabilities: Prediction probability matrix.
        reports_dir: Target directory for ``predictions.csv``.

    Raises:
        ValueError: If image paths and prediction results have different sizes.
    """
    if len(image_paths) != len(y_true):
        raise ValueError(
            f"Found {len(image_paths)} image paths but collected {len(y_true)} predictions."
        )
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Image Path", "True Class", "Predicted Class", "Confidence", "Correct Prediction"])
        for path, true_label, predicted_label, probability in zip(
            image_paths, y_true, y_pred, probabilities, strict=True
        ):
            writer.writerow([
                path,
                class_names[int(true_label)],
                class_names[int(predicted_label)],
                f"{float(np.max(probability)):.6f}",
                bool(true_label == predicted_label),
            ])


def save_figures(
    dataset: tf.data.Dataset,
    class_names: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    figures_dir: Path = FIGURES_DIR,
) -> None:
    """Create high-resolution confusion, distribution, confidence, and example figures.

    Args:
        dataset: Test dataset used to retrieve display images in prediction order.
        class_names: Class names ordered by label index.
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        probabilities: Prediction probability matrix.
        figures_dir: Target directory for PNG figures.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = np.arange(len(class_names))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    _save_confusion_matrix(matrix, class_names, figures_dir / "confusion_matrix.png", False)
    _save_confusion_matrix(matrix, class_names, figures_dir / "normalized_confusion_matrix.png", True)

    confidence = np.max(probabilities, axis=1)
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.hist(confidence, bins=20, color="#2a9d8f", edgecolor="white")
    axis.set(title="Prediction Confidence Distribution", xlabel="Confidence", ylabel="Number of Images")
    figure.tight_layout()
    figure.savefig(figures_dir / "confidence_histogram.png", dpi=_DPI, bbox_inches="tight")
    plt.close(figure)

    counts = np.bincount(y_true, minlength=len(class_names))
    figure, axis = plt.subplots(figsize=(max(12, len(class_names) * 0.4), 7))
    axis.bar(labels, counts, color="#457b9d")
    axis.set(title="Test Set Class Distribution", xlabel="Class", ylabel="Number of Images")
    axis.set_xticks(labels, class_names, rotation=90, fontsize=8)
    figure.tight_layout()
    figure.savefig(figures_dir / "class_distribution.png", dpi=_DPI, bbox_inches="tight")
    plt.close(figure)

    images = np.concatenate([batch_images.numpy() for batch_images, _ in dataset])
    _save_prediction_examples(images, class_names, y_true, y_pred, confidence, figures_dir)


def _save_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
    normalized: bool,
) -> None:
    """Save one annotated confusion matrix figure."""

    if normalized:
        display_matrix = matrix.astype(np.float64)
        row_totals = display_matrix.sum(axis=1, keepdims=True)
        display_matrix = np.divide(
            display_matrix,
            row_totals,
            out=np.zeros_like(display_matrix),
            where=row_totals != 0,
        )
        title = "Normalized Confusion Matrix"
        value_format = ".2f"

    else:
        display_matrix = matrix.astype(np.int64)
        title = "Confusion Matrix"
        value_format = "d"

    figure, axis = plt.subplots(
        figsize=(
            max(12, len(class_names) * 0.45),
            max(10, len(class_names) * 0.40),
        )
    )

    image = axis.imshow(
        display_matrix,
        interpolation="nearest",
        cmap="Blues",
    )

    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set(
        title=title,
        xlabel="Predicted Class",
        ylabel="True Class",
    )

    axis.set_xticks(
        np.arange(len(class_names)),
        class_names,
        rotation=90,
        fontsize=7,
    )

    axis.set_yticks(
        np.arange(len(class_names)),
        class_names,
        fontsize=7,
    )

    threshold = display_matrix.max() / 2 if display_matrix.size else 0

    for row, column in np.ndindex(display_matrix.shape):

        value = display_matrix[row, column]

        if normalized:
            text = f"{value:.2f}"
        else:
            text = str(int(value))

        axis.text(
            column,
            row,
            text,
            ha="center",
            va="center",
            fontsize=6,
            color="white" if value > threshold else "black",
        )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=_DPI,
        bbox_inches="tight",
    )

    plt.close(figure)


def _save_prediction_examples(
    images: np.ndarray, class_names: Sequence[str], y_true: np.ndarray, y_pred: np.ndarray,
    confidence: np.ndarray, figures_dir: Path
) -> None:
    """Save a representative grid of predictions without changing evaluation order."""
    sample_count = min(16, len(images))
    indices = np.linspace(0, len(images) - 1, sample_count, dtype=int)
    figure, axes = plt.subplots(4, 4, figsize=(14, 14))
    for axis in axes.flat:
        axis.axis("off")
    for axis, index in zip(axes.flat, indices, strict=False):
        correct = y_true[index] == y_pred[index]
        axis.imshow(np.clip(images[index], 0.0, 1.0))
        axis.set_title(
            f"True: {class_names[y_true[index]]}\nPred: {class_names[y_pred[index]]} ({confidence[index]:.1%})",
            color="#228b22" if correct else "#c1121f", fontsize=8,
        )
        axis.axis("off")
    figure.suptitle("Test Set Prediction Examples", fontsize=16)
    figure.tight_layout()
    figure.savefig(figures_dir / "prediction_examples.png", dpi=_DPI, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run the complete independent test-set evaluation and report generation.

    Raises:
        FileNotFoundError: If the test directory or best checkpoint is missing.
        ValueError: If class counts, model outputs, or prediction alignment are invalid.
    """
    validate_evaluation_inputs()
    dataset = build_test_dataset()
    class_names, image_paths = get_test_metadata()
    if len(class_names) != NUM_CLASSES:
        raise ValueError(f"Test dataset exposes {len(class_names)} classes; expected {NUM_CLASSES}.")
    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    y_true, y_pred, probabilities = collect_predictions(model, dataset)
    metrics = build_metrics(y_true, y_pred)
    save_text_reports(model, class_names, y_true, y_pred, metrics)
    save_predictions_csv(image_paths, class_names, y_true, y_pred, probabilities)
    save_figures(dataset, class_names, y_true, y_pred, probabilities)
    print(f"Evaluation complete. Reports: {REPORTS_DIR}; figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
