"""Image preprocessing utilities for the LeafFusionNet project."""

import tensorflow as tf


def preprocess_image(image: tf.Tensor) -> tf.Tensor:
    """Convert an image tensor to normalized floating-point values.

    Parameters
    ----------
    image : tf.Tensor
        Input image tensor with pixel values expected in the range ``[0, 255]``.

    Returns
    -------
    tf.Tensor
        Image tensor converted to ``tf.float32`` and normalized to the range
        ``[0, 1]`` while preserving the input tensor shape.

    Notes
    -----
    This function performs only type conversion and pixel-value normalization.
    It does not resize, augment, clip, denoise, or otherwise alter image
    content.
    """
    return tf.cast(image, tf.float32) / 255.0


__all__ = ["preprocess_image"]
