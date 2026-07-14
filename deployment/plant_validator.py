"""Local non-plant rejection using EfficientNetB0 ImageNet predictions.

This module performs a lightweight, local sanity check before an uploaded
image is handed to LeafFusionNet for disease classification. It does not
retrain or modify LeafFusionNet in any way.

Design (per the new validation pipeline):

- EfficientNetB0 is used ONLY to reject images that are OBVIOUSLY not a
  plant at all (a person, a laptop, a phone, a car, a screenshot, etc.).
- EfficientNetB0 is NEVER used to positively confirm that an image is a
  plant. ImageNet was not trained on leaf-disease imagery, and it very
  often predicts unrelated categories for real leaf photos (e.g. "conch",
  "snail", "acorn", "pot", "walking_stick", "custard_apple"). Treating
  those as rejections caused many valid leaf photos to be rejected.
- If EfficientNetB0 does not detect an obvious non-plant object, this
  validator does NOT reject and does NOT accept -- it simply defers.
  LeafFusionNet's own top-1 confidence is the sole authority on whether
  the image is actually a valid leaf (handled downstream, e.g. in
  predictor.py, against LEAFFUSION_MIN_CONFIDENCE).

There is deliberately no positive keyword list, no plant_score, and no
positive confidence accumulation in this module anymore.
"""

from __future__ import annotations

import io
import logging
import re

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import (
    decode_predictions,
    preprocess_input,
)

logger = logging.getLogger(__name__)

_EFFICIENTNET_IMAGE_SIZE = (224, 224)
_INVALID_PLANT_MESSAGE = "Please upload a clear photo of a single plant leaf."
_TOP_K = 5

# Any Top-5 ImageNet prediction matching one of these words causes an
# immediate rejection. These are obvious non-plant subjects (people,
# animals, vehicles, electronics, screens, household objects, buildings)
# that should never be routed to LeafFusionNet. This is the ONLY
# rejection criterion EfficientNetB0 is used for -- it never rejects
# because it is merely "unsure".
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
    "desktop computer",
    "television",
    "screen",
    "website",
    "web site",
    "menu",
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
    """Check whether any keyword appears as a whole word/phrase in a class name.

    Word-boundary matching is used (rather than plain substring matching) so
    that short keywords such as ``"bus"`` do not accidentally match
    unrelated class names that merely contain that substring.

    Args:
        normalized_name: A lowercased, space-normalized class name.
        keywords: The keyword tuple to check against.

    Returns:
        ``True`` if any keyword matches as a whole word/phrase, else ``False``.
    """
    return any(
        re.search(rf"\b{re.escape(keyword)}\b", normalized_name) is not None
        for keyword in keywords
    )


def is_obvious_non_plant(image_bytes: bytes) -> tuple[bool, str]:
    """Check whether uploaded image bytes are an OBVIOUS non-plant image.

    This runs EfficientNetB0's Top-5 ImageNet predictions and checks them
    only against a negative (non-plant) keyword list.

    - If any Top-5 class matches a negative keyword (person, car, phone,
      screen, website, etc.), the image is rejected.
    - Otherwise, the image is NOT rejected here. This function makes no
      claim that the image IS a plant -- that determination belongs to
      LeafFusionNet's own confidence score.

    Args:
        image_bytes: Raw uploaded image bytes.

    Returns:
        A ``(is_valid, message)`` tuple. ``is_valid`` is ``False`` only when
        the image cannot be decoded or a negative keyword is matched.
        ``message`` is only meaningful when ``is_valid`` is ``False``.

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
    except Exception as exc:
        raise RuntimeError("EfficientNetB0 plant validation failed.") from exc

    normalized_predictions = [
        (_normalize_class_name(class_name), float(confidence))
        for _, class_name, confidence in decoded_predictions
    ]

    logger.info(
        "EfficientNet Top-5: %s",
        ", ".join(f"{name}={conf:.4f}" for name, conf in normalized_predictions),
    )

    for normalized_name, _confidence in normalized_predictions:
        if _matches_any_keyword(normalized_name, _NEGATIVE_KEYWORDS):
            logger.info(
                "EfficientNet decision: REJECT (matched non-plant keyword in '%s')",
                normalized_name,
            )
            return False, _INVALID_PLANT_MESSAGE

    logger.info("EfficientNet decision: no obvious non-plant object detected; deferring to LeafFusionNet")
    return True, ""


__all__ = ["is_obvious_non_plant"]
