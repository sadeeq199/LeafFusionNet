"""Feature fusion block for the LeafFusionNet architecture.

This module combines feature maps from the two custom convolutional branches
into a single fused feature vector. It contains no feature extraction,
classifier, or model assembly logic.
"""

import tensorflow as tf

from config.hyperparameters import FUSION_DROPOUT_1, FUSION_UNITS


def build_feature_fusion(
    branch_a_features: tf.Tensor,
    branch_b_features: tf.Tensor,
) -> tf.Tensor:
    """Build the LeafFusionNet feature fusion block.

    Parameters
    ----------
    branch_a_features : tf.Tensor
        Feature map produced by Branch A.
    branch_b_features : tf.Tensor
        Feature map produced by Branch B.

    Returns
    -------
    tf.Tensor
        Fused feature vector after global average pooling, concatenation,
        dense projection, batch normalization, ReLU, and dropout.
    """
    branch_a_vector = tf.keras.layers.GlobalAveragePooling2D(
        name="branch_a_global_average_pooling"
    )(branch_a_features)
    branch_b_vector = tf.keras.layers.GlobalAveragePooling2D(
        name="branch_b_global_average_pooling"
    )(branch_b_features)

    x = tf.keras.layers.Concatenate(name="feature_concatenation")(
        [branch_a_vector, branch_b_vector]
    )
    x = tf.keras.layers.Dense(
        units=FUSION_UNITS,
        name="fusion_dense",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="fusion_batch_norm")(x)
    x = tf.keras.layers.ReLU(name="fusion_relu")(x)
    return tf.keras.layers.Dropout(
        rate=FUSION_DROPOUT_1,
        name="fusion_dropout",
    )(x)


__all__ = ["build_feature_fusion"]
