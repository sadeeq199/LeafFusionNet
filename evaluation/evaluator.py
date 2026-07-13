"""Model evaluation utilities for the LeafFusionNet project."""

import numpy as np
import tensorflow as tf

from evaluation.metrics import compute_metrics


def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> dict[str, float]:
    """Evaluate a model on a labeled TensorFlow dataset.

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model that outputs class probability predictions.
    dataset : tf.data.Dataset
        Labeled dataset yielding batches of images and ground-truth labels.

    Returns
    -------
    dict[str, float]
        Classification metrics computed from collected ground-truth and
        predicted class labels.

    Raises
    ------
    ValueError
        If ``model`` or ``dataset`` is ``None``.
    """
    if model is None:
        msg = "model must not be None."
        raise ValueError(msg)
    if dataset is None:
        msg = "dataset must not be None."
        raise ValueError(msg)

    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)

        y_true.extend(np.asarray(labels).astype(int).tolist())
        y_pred.extend(predicted_labels.astype(int).tolist())

    return compute_metrics(y_true, y_pred)


__all__ = ["evaluate_model"]
