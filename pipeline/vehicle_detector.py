import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck


class VehicleDetector:
    def __init__(self, weights: str = "yolov8n.pt"):
        self.model = YOLO(weights)

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """
        Returns list of (x1, y1, x2, y2, conf) untuk setiap kendaraan terdeteksi.
        """
        results = self.model(frame)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append((x1, y1, x2, y2, conf))
        return detections