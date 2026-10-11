"""OpenCV preprocessing for low-quality smartphone prescription photos."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode the uploaded image.")
    return bgr


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_prescription_image(
    image_bytes: bytes,
) -> Tuple[np.ndarray, np.ndarray, Image.Image]:
    """Enhance a prescription photo for OCR.

    Pipeline: decode → grayscale → CLAHE → Gaussian denoise → light binarization.

    Returns:
        original_rgb, processed_rgb, processed_pil
    """
    original_bgr = _bytes_to_bgr(image_bytes)
    original_rgb = _bgr_to_rgb(original_bgr)

    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Adaptive threshold helps faint ink without destroying printed text.
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )
    # Blend CLAHE and binary so handwriting stays readable in the UI.
    blended = cv2.addWeighted(denoised, 0.72, binary, 0.28, 0)
    processed_rgb = cv2.cvtColor(blended, cv2.COLOR_GRAY2RGB)
    processed_pil = Image.fromarray(processed_rgb)
    return original_rgb, processed_rgb, processed_pil
