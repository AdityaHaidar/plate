import cv2
import numpy as np

COLORS = [
    (0, 255, 0), (255, 165, 0), (0, 165, 255),
    (255, 0, 255), (0, 255, 255), (255, 255, 0)
]


def draw_vehicle_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    track_id: int,
    plate_text: str | None = None
) -> None:
    color = COLORS[track_id % len(COLORS)]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"ID:{track_id}"
    if plate_text:
        label += f"  {plate_text}"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)


def embed_plate_crop(
    frame: np.ndarray,
    plate_crop: np.ndarray,
    x1: int, y2: int,
    max_w: int = 120,
    max_h: int = 40
) -> None:
    if plate_crop is None or plate_crop.size == 0:
        return

    h, w = plate_crop.shape[:2]
    if h == 0 or w == 0:
        return

    scale = min(max_w / w, max_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

    thumb = cv2.resize(plate_crop, (new_w, new_h))
    if len(thumb.shape) == 2:
        thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
    else:
        thumb = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)

    fh, fw = frame.shape[:2]
    fx = max(0, x1)
    fy = max(0, y2 + 4)

    x_end = min(fx + new_w, fw)
    y_end = min(fy + new_h, fh)

    paste_w = x_end - fx
    paste_h = y_end - fy

    # Pastikan area paste benar-benar positif sebelum assign
    if paste_w <= 0 or paste_h <= 0:
        return

    frame[fy:y_end, fx:x_end] = thumb[:paste_h, :paste_w]
    cv2.rectangle(frame, (fx, fy), (x_end - 1, y_end - 1), (255, 255, 255), 1)