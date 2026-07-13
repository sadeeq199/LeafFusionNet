"""Gemini-based image suitability validation for LeafFusionNet inference.

This module validates whether uploaded image bytes contain exactly one clear,
visible plant leaf suitable for disease diagnosis before the TensorFlow
classifier is invoked.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from google import genai
from google.genai import types

_API_KEY = os.getenv("GEMINI_API_KEY")
if not _API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is required for Gemini image validation but was not set."
    )

_CLIENT = genai.Client(api_key=_API_KEY)
_MODEL_NAMES = (
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
)
_DEFAULT_MIME_TYPE = "image/jpeg"
_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

_VALIDATION_PROMPT = """
You are a strict image validation service for a plant disease diagnosis API.

Decide whether the uploaded image is suitable for plant leaf disease
classification.

Return valid=true ONLY IF all conditions are satisfied:
- The image contains exactly one plant leaf.
- The image is clear and not blurred.
- The leaf is visible.
- The leaf occupies a reasonable portion of the image.
- The image is appropriate for plant disease diagnosis.

Return valid=false if the image contains or appears to be any of these:
- person
- phone
- laptop
- keyboard
- car
- pet
- food
- random object
- blurred image
- dark image
- multiple leaves
- no leaf
- artificial image
- screenshot
- document
- drawing

Answer ONLY valid JSON with this exact shape:
{
  "valid": true,
  "reason": "Valid plant leaf"
}

or:
{
  "valid": false,
  "reason": "The uploaded image does not contain a plant leaf."
}

Do not return markdown, comments, explanations, or extra text.
""".strip()


def _detect_mime_type(image_bytes: bytes) -> str:
    """Detect a supported image MIME type from binary signatures."""
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return "image/gif"
    return _DEFAULT_MIME_TYPE


def _parse_validation_response(response_text: str | None) -> tuple[bool, str]:
    """Parse Gemini JSON response into a validation result."""
    if not response_text:
        return False, "Gemini returned an empty validation response."

    try:
        payload: Any = json.loads(response_text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_PATTERN.search(response_text)
        if match is None:
            return False, "Gemini returned a malformed validation response."
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return False, "Gemini returned a malformed validation response."

    if not isinstance(payload, dict):
        return False, "Gemini returned an invalid validation response."

    valid = payload.get("valid")
    reason = payload.get("reason")

    if not isinstance(valid, bool) or not isinstance(reason, str):
        return False, "Gemini returned an invalid validation response."

    if valid:
        return True, "Valid plant leaf"

    clean_reason = reason.strip()
    if clean_reason:
        return False, clean_reason
    return False, "حط صورة ورقه نبات يا خول."


def _is_not_found_error(exc: Exception) -> bool:
    """Return whether a Gemini SDK exception represents a 404 response."""
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if status_code == 404:
        return True

    error_text = str(exc).upper()
    return "404" in error_text or "NOT_FOUND" in error_text


def validate_leaf_image(image_bytes: bytes) -> tuple[bool, str]:
    """Validate whether image bytes contain one diagnosable plant leaf.

    Args:
        image_bytes: Raw uploaded image bytes.

    Returns:
        A tuple containing a validation flag and a human-readable reason. Valid
        images return ``(True, "Valid plant leaf")``. Invalid images return
        ``(False, "<reason>")``.

    Raises:
        RuntimeError: If Gemini validation fails because the API is temporarily
            unavailable or the request cannot be completed.
    """
    if not image_bytes:
        return False, "The uploaded image is empty."

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=_detect_mime_type(image_bytes),
    )

    first_not_found_error: Exception | None = None

    for model_name in _MODEL_NAMES:
        try:
            response = _CLIENT.models.generate_content(
                model=model_name,
                contents=[_VALIDATION_PROMPT, image_part],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
            return _parse_validation_response(response.text)
        except Exception as exc:
            if _is_not_found_error(exc):
                if first_not_found_error is None:
                    first_not_found_error = exc
                continue
            raise RuntimeError(
                "Gemini image validation is temporarily unavailable."
            ) from exc

    raise RuntimeError(
        "Gemini image validation failed because no configured Gemini model is available."
    ) from first_not_found_error


__all__ = ["validate_leaf_image"]
