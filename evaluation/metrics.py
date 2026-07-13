"""Classification metric computation utilities for LeafFusionNet."""

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute weighted classification metrics.

    Parameters
    ----------
    y_true : Any
        Ground-truth class labels.
    y_pred : Any
        Predicted class labels.

    Returns
    -------
    dict[str, float]
        Dictionary containing accuracy, precision, recall, and F1 score.

    Raises
    ------
    ValueError
        If ``y_true`` and ``y_pred`` have different lengths.
    """
    true_labels = np.asarray(y_true)
    predicted_labels = np.asarray(y_pred)

    if len(true_labels) != len(predicted_labels):
        msg = "y_true and y_pred must have the same length."
        raise ValueError(msg)

    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "precision": float(
            precision_score(
                true_labels,
                predicted_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                true_labels,
                predicted_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }


__all__ = ["compute_metrics"]
