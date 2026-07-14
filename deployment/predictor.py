"""Image validation, preprocessing, and prediction for LeafFusionNet."""

from __future__ import annotations

import io
import threading
import time

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from .config import IMAGE_SIZE, SUPPORTED_IMAGE_FORMATS
from .disease_info import DISEASE_INFO
from .model_loader import get_class_names, get_model
from .plant_validator import validate_plant_image
from .schemas import PredictionItem, PredictionResponse

_prediction_lock = threading.Lock()


class InvalidImageError(ValueError):
    """Raised when an uploaded image is unsupported or cannot be decoded."""


def _validate_image(data: bytes) -> None:
    """Verify that image bytes are a complete, supported image.

    Args:
        data: Raw upload bytes.

    Raises:
        InvalidImageError: If the bytes do not represent a supported image.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in SUPPORTED_IMAGE_FORMATS:
                raise InvalidImageError("Only JPG, JPEG, and PNG image files are supported.")
            image.verify()
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("The uploaded file is not a valid, complete image.") from exc


def _preprocess_image(data: bytes) -> tf.Tensor:
    """Apply the exact existing inference preprocessing to upload bytes.

    This mirrors ``inference/predict.py``: Keras RGB decoding, resize to
    224x224, conversion to an array, batch expansion, and division by 255.

    Args:
        data: Validated raw image bytes.

    Returns:
        A normalized model-ready image batch.
    """
    image = tf.keras.utils.load_img(
        io.BytesIO(data),
        target_size=IMAGE_SIZE,
        color_mode="rgb",
    )
    image_array = tf.keras.utils.img_to_array(image)
    image_tensor = tf.expand_dims(tf.convert_to_tensor(image_array), axis=0)
    return tf.cast(image_tensor, tf.float32) / 255.0


def predict_image(data: bytes) -> PredictionResponse:
    """Run a top-five LeafFusionNet prediction for validated image bytes.

    Args:
        data: Raw image upload bytes.

    Returns:
        Structured prediction response with percentage confidences.
    """
    _validate_image(data)
    is_plant, message = validate_plant_image(data)
    if not is_plant:
        raise InvalidImageError(message)

    model_input = _preprocess_image(data)
    class_names = get_class_names()

    started_at = time.perf_counter()
    with _prediction_lock:
        probabilities = get_model()(model_input, training=False).numpy()[0]
    inference_time_ms = (time.perf_counter() - started_at) * 1000

    top_indices = np.argsort(probabilities)[::-1][: min(5, len(class_names))]
    top5 = [
        PredictionItem(class_name=class_names[int(index)], confidence=float(probabilities[index] * 100))
        for index in top_indices
    ]
    best = top5[0]
    predicted_class = best.class_name

    try:
        disease = DISEASE_INFO[predicted_class]
    except KeyError as exc:
        raise RuntimeError(
            f"No disease info entry found for predicted class '{predicted_class}'. "
            "DISEASE_INFO is out of sync with the model's class labels."
        ) from exc

    return PredictionResponse(
        prediction=predicted_class,
        confidence=best.confidence,
        display_name=disease["display_name"],
        crop=disease["crop"],
        severity=disease["severity"],
        description=disease["description"],
        treatment=disease["treatment"],
        prevention=disease["prevention"],
        top5=top5,
        inference_time_ms=inference_time_ms,
    )
