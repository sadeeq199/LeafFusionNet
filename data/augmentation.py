"""Data augmentation utilities for the LeafFusionNet project."""

import tensorflow as tf

_AUGMENTATION_PIPELINE = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(mode="horizontal"),
        tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),
    ],
    name="leaf_fusion_augmentation",
)


def augment_image(image: tf.Tensor, training: bool) -> tf.Tensor:
    """Apply training-time image augmentation.

    Parameters
    ----------
    image : tf.Tensor
        Input image tensor to augment.
    training : bool
        Whether augmentation should be applied. If ``False``, the input image
        is returned unchanged.

    Returns
    -------
    tf.Tensor
        The augmented image tensor during training, or the original image
        tensor when not training.

    Notes
    -----
    This function applies only random horizontal flipping and a small random
    zoom during training. Rotation and all other augmentation types are
    intentionally excluded.
    """
    if not training:
        return image

    return _AUGMENTATION_PIPELINE(image, training=True)


__all__ = ["augment_image"]
