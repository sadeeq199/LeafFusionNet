"""Complete LeafFusionNet model assembly.

This module assembles the custom LeafFusionNet architecture from existing
project building blocks. It contains no feature extraction, feature fusion, or
classifier implementation details.
"""

import tensorflow as tf

from config.hyperparameters import INPUT_SHAPE
from models.backbones import build_branch_a, build_branch_b, build_stem
from models.classifier import build_classifier
from models.fusion import build_feature_fusion


def build_leaffusionnet() -> tf.keras.Model:
    """Build the complete LeafFusionNet model.

    Returns
    -------
    tf.keras.Model
        Functional Keras model named ``LeafFusionNet``.

    Notes
    -----
    The model is assembled exclusively from existing project building blocks:
    stem, dual branches, feature fusion, and classifier head.
    """
    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="input_image")

    stem_features = build_stem(inputs)
    branch_a_features = build_branch_a(stem_features)
    branch_b_features = build_branch_b(stem_features)
    fused_features = build_feature_fusion(branch_a_features, branch_b_features)
    outputs = build_classifier(fused_features)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="LeafFusionNet",
    )


__all__ = ["build_leaffusionnet"]
