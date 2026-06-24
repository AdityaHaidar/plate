import numpy as np
from ultralytics import YOLO


class PlateDetector:
    def __init__(self, weights: str = "./models/license_plate_detector.pt"):
        self.model = YOLO(weights)

    def detect(self, vehicle_crop: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Returns list of (x1, y1, x2, y2) relative to vehicle_crop.
        """
        results = self.model(vehicle_crop)[0]
        plates = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plates.append((x1, y1, x2, y2))
        return plates