import numpy as np
import easyocr
from utils.image_utils import preprocess_for_ocr, trim_plate_crop
from utils.plate_format import clean_plate_text, format_indonesian_plate


class OCRReader:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True)

    def read_plate(self, vehicle_crop: np.ndarray, px1: int, py1: int, px2: int, py2: int) -> str | None:
        """
        Baca plat dari vehicle_crop dengan koordinat plat.
        Returns formatted plate string atau None jika gagal.
        """
        ocr_crop = trim_plate_crop(vehicle_crop, px1, py1, px2, py2)

        if ocr_crop.size == 0:
            return None

        thresh = preprocess_for_ocr(ocr_crop)

        results = self.reader.readtext(
            thresh,
            paragraph=True,
            width_ths=1.5,
            x_ths=1.0,
            y_ths=0.5
        )

        parts = []
        for result in results:
            try:
                text = result[1] if len(result) >= 2 else ""
                cleaned = clean_plate_text(text)
                if cleaned:
                    parts.append(cleaned)
            except Exception:
                continue

        combined = ''.join(parts)
        return format_indonesian_plate(combined)