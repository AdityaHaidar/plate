import numpy as np
import sys
import os

# Pastikan folder sort/ ada di path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sort'))
from sort import Sort


class VehicleTracker:
    def __init__(self, max_age: int = 10, min_hits: int = 3, iou_threshold: float = 0.3):
        self.tracker = Sort(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold
        )

    def update(self, detections: list[tuple[int, int, int, int, float]]) -> np.ndarray:
        """
        Input:  list of (x1, y1, x2, y2, conf)
        Output: numpy array shape (N, 5) → [x1, y1, x2, y2, track_id]
        """
        if len(detections) == 0:
            return self.tracker.update(np.empty((0, 5)))

        dets = np.array(detections, dtype=float)
        return self.tracker.update(dets)