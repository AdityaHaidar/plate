import cv2
import numpy as np


def crop_region(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Crop region dari image. Clamp agar tidak out of bounds."""
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2]


def preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    """Grayscale + Otsu threshold untuk OCR."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def trim_plate_crop(vehicle_crop: np.ndarray, px1, py1, px2, py2) -> np.ndarray:
    """Trim bagian atas plat (area kota) agar OCR lebih akurat."""
    w, h = px2 - px1, py2 - py1
    px1_new = max(0, px1 + int(0.03 * w))
    px2_new = min(vehicle_crop.shape[1], px2 - int(0.03 * w))
    py1_new = max(0, py1 + int(0.08 * h))
    py2_new = min(vehicle_crop.shape[0], py2 - int(0.35 * h))
    return vehicle_crop[py1_new:py2_new, px1_new:px2_new]