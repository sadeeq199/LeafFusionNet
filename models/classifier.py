"""Classification head for the LeafFusionNet architecture.

This module maps fused feature vectors to class probabilities using only the
configured classifier output layer. It contains no feature extraction, feature
fusion, or model assembly logic.
"""

import tensorflow as tf

from config.hyperparameters import CLASSIFIER_ACTIVATION, CLASSIFIER_UNITS


def build_classifier(fused_features: tf.Tensor) -> tf.Tensor:
    """Build the LeafFusionNet classification head.

    Parameters
    ----------
    fused_features : tf.Tensor
        Fused feature vector produced by the feature fusion block.

    Returns
    -------
    tf.Tensor
        Prediction tensor containing class probabilities.
    """
    return tf.keras.layers.Dense(
        units=CLASSIFIER_UNITS,
        activation=CLASSIFIER_ACTIVATION,
        name="classifier",
    )(fused_features)


__all__ = ["build_classifier"]
