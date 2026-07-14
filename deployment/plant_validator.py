"""Local plant image validation using EfficientNetB0 ImageNet predictions.

This module performs a lightweight, local sanity check before an uploaded
image is handed to LeafFusionNet for disease classification. It does not
retrain or modify LeafFusionNet in any way; it only decides whether an image
is *obviously* not a plant (e.g. a person, a car, a phone) and should be
rejected before wasting a LeafFusionNet inference call.
"""

from __future__ import annotations

import io
import re

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import (
    decode_predictions,
    preprocess_input,
)

_EFFICIENTNET_IMAGE_SIZE = (224, 224)
_PLANT_DETECTED_MESSAGE = "Plant detected"
_INVALID_PLANT_MESSAGE = "Please upload a clear photo of a single plant leaf."
_TOP_K = 5

# Minimum cumulative confidence, summed across all Top-5 positive-keyword
# matches, required to accept an image as a plant. Summing rather than
# checking a single class lets corroborating evidence across multiple
# Top-5 slots (e.g. "corn" and "maize" both appearing) count in the
# image's favor, while still rejecting images whose only plant-like
# signal is a single low-confidence class.
_MIN_PLANT_SCORE = 0.20

# Any Top-5 ImageNet prediction matching one of these words causes an
# immediate rejection. These are common non-plant subjects (people, animals,
# vehicles, electronics, household objects, buildings) that should never be
# routed to LeafFusionNet.
_NEGATIVE_KEYWORDS = (
    "person",
    "man",
    "woman",
    "boy",
    "girl",
    "face",
    "human",
    "dog",
    "cat",
    "horse",
    "bird",
    "car",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
    "phone",
    "cellular",
    "smartphone",
    "laptop",
    "keyboard",
    "monitor",
    "television",
    "screen",
    "remote",
    "mouse",
    "book",
    "wallet",
    "watch",
    "shoe",
    "bottle",
    "cup",
    "chair",
    "table",
    "bed",
    "building",
    "house",
    "airplane",
    "train",
    "boat",
)

# Any Top-5 ImageNet prediction matching one of these words is treated as
# evidence of plant-leaf/crop content, but ONLY when paired with a
# sufficiently high confidence score (see _POSITIVE_CONFIDENCE_THRESHOLD).
# This list is intentionally narrow and focused on plant leaves and the
# specific crops LeafFusionNet classifies. Broad or tangential categories
# (generic trees, forests, wood, flowers, fungi, and non-leaf produce such
# as apples or bananas) are deliberately excluded, since matching on them
# says little about whether the image actually shows a leaf and hurts
# precision.
_POSITIVE_KEYWORDS = (
    "leaf",
    "vine",
    "corn",
    "maize",
    "wheat",
    "rice",
    "soybean",
    "bean",
    "pea",
    "potato",
    "tomato",
    "pepper",
    "eggplant",
    "broccoli",
    "cabbage",
    "cauliflower",
    "lettuce",
    "spinach",
    "pumpkin",
    "cucumber",
    "artichoke",
    "grape",
)


def _load_efficientnet() -> tf.keras.Model:
    """Load the reusable EfficientNetB0 ImageNet validation model."""
    try:
        return EfficientNetB0(weights="imagenet", include_top=True)
    except Exception as exc:
        raise RuntimeError("EfficientNetB0 plant validator failed to load.") from exc


_PLANT_VALIDATOR_MODEL = _load_efficientnet()


def _normalize_class_name(class_name: str) -> str:
    """Normalize a decoded ImageNet class name for keyword matching.

    Args:
        class_name: Raw decoded class name (e.g. ``"granny_smith"``).

    Returns:
        The class name lowercased with underscores replaced by spaces.
    """
    return class_name.lower().replace("_", " ")


def _matches_any_keyword(normalized_name: str, keywords: tuple[str, ...]) -> bool:
    """Check whether any keyword appears as a whole word in a class name.

    Word-boundary matching is used (rather than plain substring matching) so
    that short keywords such as ``"bus"`` or ``"nut"`` do not accidentally
    match unrelated class names that merely contain that substring.

    Args:
        normalized_name: A lowercased, space-normalized class name.
        keywords: The keyword tuple to check against.

    Returns:
        ``True`` if any keyword matches as a whole word, else ``False``.
    """
    return any(
        re.search(rf"\b{re.escape(keyword)}\b", normalized_name) is not None
        for keyword in keywords
    )


def validate_plant_image(image_bytes: bytes) -> tuple[bool, str]:
    """Validate whether uploaded image bytes likely contain plant content.

    The Top-5 EfficientNetB0 ImageNet predictions are checked against a
    negative keyword list and a positive keyword list:

    - If any Top-5 class matches a negative keyword (person, car, phone,
      etc.), the image is immediately rejected.
    - Otherwise, a ``plant_score`` is computed as the sum of the confidence
      values of every Top-5 class that matches a positive (plant-related)
      keyword. If ``plant_score >= _MIN_PLANT_SCORE``, the image is
      accepted.
    - Otherwise, the image is rejected. An unrecognized or weakly-supported
      ImageNet category is not treated as evidence of a plant. This
      validator prioritizes precision: false positives (letting a
      non-plant image through to LeafFusionNet) are considered worse than
      false negatives (rejecting a real leaf).

    Args:
        image_bytes: Raw uploaded image bytes.

    Returns:
        A ``(is_valid, message)`` tuple. ``is_valid`` is ``False`` only when
        the image cannot be decoded or a negative keyword is matched.

    Raises:
        RuntimeError: If EfficientNetB0 prediction or decoding fails
            unexpectedly.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb_image = image.convert("RGB")
            resized_image = rgb_image.resize(
                _EFFICIENTNET_IMAGE_SIZE, resample=Image.Resampling.BILINEAR
            )
    except (UnidentifiedImageError, OSError, ValueError):
        return False, _INVALID_PLANT_MESSAGE

    image_array = tf.keras.utils.img_to_array(resized_image)
    batch = np.expand_dims(image_array, axis=0)
    batch = preprocess_input(batch)

    try:
        predictions = _PLANT_VALIDATOR_MODEL(batch, training=False).numpy()
        decoded_predictions = decode_predictions(predictions, top=_TOP_K)[0]

        print("\n===== EfficientNet Top-5 =====")
        for _, class_name, confidence in decoded_predictions:
            print(f"{class_name}: {confidence:.4f}")
        print("==============================\n")

    except Exception as exc:
        raise RuntimeError("EfficientNetB0 plant validation failed.") from exc

    normalized_predictions = [
        (_normalize_class_name(class_name), float(confidence))
        for _, class_name, confidence in decoded_predictions
    ]

    for normalized_name, _confidence in normalized_predictions:
        if _matches_any_keyword(normalized_name, _NEGATIVE_KEYWORDS):
            return False, _INVALID_PLANT_MESSAGE

    plant_score = sum(
        confidence
        for normalized_name, confidence in normalized_predictions
        if _matches_any_keyword(normalized_name, _POSITIVE_KEYWORDS)
    )

    if plant_score >= _MIN_PLANT_SCORE:
        return True, _PLANT_DETECTED_MESSAGE

    return False, _INVALID_PLANT_MESSAGE


__all__ = ["validate_plant_image"]
