"""Single-image inference utilities for the LeafFusionNet project."""

from pathlib import Path

import numpy as np
import tensorflow as tf

from config.hyperparameters import IMG_SIZE
from data.preprocessing import preprocess_image


def predict_image(
    model: tf.keras.Model,
    image_path: str,
    class_names: list[str] | tuple[str, ...],
) -> tuple[str, float]:
    """Predict the class label for a single image.

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model used for prediction.
    image_path : str
        Path to the image file on disk.
    class_names : list[str] | tuple[str, ...]
        Ordered class names corresponding to the model output indices.

    Returns
    -------
    tuple[str, float]
        Predicted class name and confidence score.

    Raises
    ------
    ValueError
        If ``model`` is ``None`` or ``class_names`` is empty.
    FileNotFoundError
        If ``image_path`` does not exist.
    """
    if model is None:
        msg = "model must not be None."
        raise ValueError(msg)
    if not class_names:
        msg = "class_names must not be empty."
        raise ValueError(msg)

    path = Path(image_path)
    if not path.is_file():
        msg = f"Image file does not exist: {path}"
        raise FileNotFoundError(msg)

    image = tf.keras.utils.load_img(
        path,
        target_size=(IMG_SIZE, IMG_SIZE),
    )
    image_array = tf.keras.utils.img_to_array(image)
    image_tensor = tf.expand_dims(tf.convert_to_tensor(image_array), axis=0)
    processed_image = preprocess_image(image_tensor)

    predictions = model.predict(processed_image, verbose=0)
    predicted_index = int(np.argmax(predictions, axis=1)[0])
    confidence_score = float(np.max(predictions, axis=1)[0])

    return class_names[predicted_index], confidence_score


__all__ = ["predict_image"]
