"""Feature extraction blocks for the LeafFusionNet architecture.

This module implements the custom stem and dual-branch convolutional feature
extractors from scratch using standard TensorFlow Keras layers only. It does
not contain model heads, pooling, dropout, feature fusion, pretrained models,
or transfer learning components.
"""

import tensorflow as tf

from config.hyperparameters import (
    BRANCH_A_FILTERS,
    BRANCH_A_KERNEL_SIZES,
    BRANCH_B_DILATION_RATE,
    BRANCH_B_FILTERS,
    BRANCH_B_KERNEL_SIZES,
    PADDING,
    STEM_FILTERS,
    STEM_KERNEL_SIZE,
    STEM_STRIDES,
)

_PROJECTION_KERNEL_SIZE = (1, 1)


def build_stem(inputs: tf.Tensor) -> tf.Tensor:
    """Build the LeafFusionNet stem block.

    Parameters
    ----------
    inputs : tf.Tensor
        Input image tensor with shape compatible with ``224 x 224 x 3``.

    Returns
    -------
    tf.Tensor
        Stem feature map after convolution, batch normalization, and ReLU.
    """
    x = tf.keras.layers.Conv2D(
        filters=STEM_FILTERS,
        kernel_size=STEM_KERNEL_SIZE,
        strides=STEM_STRIDES,
        padding=PADDING,
        name="stem_conv",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="stem_batch_norm")(x)
    return tf.keras.layers.ReLU(name="stem_relu")(x)


def build_branch_a(inputs: tf.Tensor) -> tf.Tensor:
    """Build Branch A of the LeafFusionNet feature extractor.

    Parameters
    ----------
    inputs : tf.Tensor
        Stem feature map tensor passed into Branch A.

    Returns
    -------
    tf.Tensor
        Branch A feature map after residual addition and ReLU activation.
    """
    shortcut = inputs

    x = tf.keras.layers.Conv2D(
        filters=BRANCH_A_FILTERS[0],
        kernel_size=BRANCH_A_KERNEL_SIZES[0],
        padding=PADDING,
        name="branch_a_conv_1",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="branch_a_batch_norm_1")(x)
    x = tf.keras.layers.ReLU(name="branch_a_relu_1")(x)

    x = tf.keras.layers.Conv2D(
        filters=BRANCH_A_FILTERS[1],
        kernel_size=BRANCH_A_KERNEL_SIZES[1],
        padding=PADDING,
        name="branch_a_conv_2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="branch_a_batch_norm_2")(x)

    if inputs.shape[-1] != BRANCH_A_FILTERS[1]:
        shortcut = tf.keras.layers.Conv2D(
            filters=BRANCH_A_FILTERS[1],
            kernel_size=_PROJECTION_KERNEL_SIZE,
            padding=PADDING,
            name="branch_a_projection",
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(
            name="branch_a_projection_batch_norm"
        )(shortcut)

    x = tf.keras.layers.Add(name="branch_a_residual_add")([x, shortcut])
    return tf.keras.layers.ReLU(name="branch_a_relu_out")(x)


def build_branch_b(inputs: tf.Tensor) -> tf.Tensor:
    """Build Branch B of the LeafFusionNet feature extractor.

    Parameters
    ----------
    inputs : tf.Tensor
        Stem feature map tensor passed into Branch B.

    Returns
    -------
    tf.Tensor
        Branch B feature map after residual addition and ReLU activation.
    """
    shortcut = inputs

    x = tf.keras.layers.Conv2D(
        filters=BRANCH_B_FILTERS[0],
        kernel_size=BRANCH_B_KERNEL_SIZES[0],
        padding=PADDING,
        name="branch_b_conv_1",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="branch_b_batch_norm_1")(x)
    x = tf.keras.layers.ReLU(name="branch_b_relu_1")(x)

    x = tf.keras.layers.Conv2D(
        filters=BRANCH_B_FILTERS[1],
        kernel_size=BRANCH_B_KERNEL_SIZES[1],
        padding=PADDING,
        dilation_rate=BRANCH_B_DILATION_RATE,
        name="branch_b_conv_2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="branch_b_batch_norm_2")(x)

    if inputs.shape[-1] != BRANCH_B_FILTERS[1]:
        shortcut = tf.keras.layers.Conv2D(
            filters=BRANCH_B_FILTERS[1],
            kernel_size=_PROJECTION_KERNEL_SIZE,
            padding=PADDING,
            name="branch_b_projection",
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(
            name="branch_b_projection_batch_norm"
        )(shortcut)

    x = tf.keras.layers.Add(name="branch_b_residual_add")([x, shortcut])
    return tf.keras.layers.ReLU(name="branch_b_relu_out")(x)


__all__ = ["build_stem", "build_branch_a", "build_branch_b"]
