import cv2
import numpy as np
import streamlit as st
import tempfile
import os
import time

from pipeline.vehicle_detector import VehicleDetector
from pipeline.plate_detector import PlateDetector
from pipeline.ocr_reader import OCRReader
from pipeline.tracker import VehicleTracker
from pipeline.visualizer import draw_vehicle_box, embed_plate_crop
from utils.image_utils import crop_region

# ==================================================
# CONFIG
# ==================================================

st.set_page_config(page_title="Vehicle Plate Video", layout="wide")
st.title("🎥 Vehicle Detection + Plate OCR — Video Pipeline")

OCR_EVERY_N_FRAMES = 5   # jalankan OCR setiap N frame per kendaraan
PLATE_CACHE_TTL    = 30  # frame, sebelum re-OCR ulang

# ==================================================
# LOAD MODELS (cached)
# ==================================================

@st.cache_resource
def load_pipeline():
    return (
        VehicleDetector("yolov8n.pt"),
        PlateDetector("./models/license_plate_detector.pt"),
        OCRReader(),
        VehicleTracker()
    )

vehicle_detector, plate_detector, ocr_reader, tracker = load_pipeline()

# ==================================================
# UI
# ==================================================

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

col_left, col_right = st.columns([3, 1])
with col_left:
    run_btn = st.button("▶ Proses Video", type="primary", disabled=uploaded_video is None)
with col_right:
    show_crops = st.checkbox("Tampilkan crop plat di video", value=True)

frame_display   = st.empty()
progress_bar    = st.progress(0)
status_text     = st.empty()

# ==================================================
# MAIN PROCESSING
# ==================================================

if uploaded_video and run_btn:

    # Simpan ke tempfile
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    tfile.flush()
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25

    # Output video writer
    out_path = tempfile.mktemp(suffix="_output.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    fw       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (fw, fh))

    # Cache plat per track_id: {track_id: {"text": str, "last_frame": int, "crop": ndarray}}
    plate_cache: dict[int, dict] = {}

    frame_idx = 0

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ── 1. Deteksi kendaraan ──────────────────────────────────────
        detections = vehicle_detector.detect(frame_rgb)

        # ── 2. SORT tracking ─────────────────────────────────────────
        tracked = tracker.update(detections)  # (N, 5) → x1,y1,x2,y2,id

        # ── 3. Per kendaraan: deteksi plat + OCR ─────────────────────
        output_frame = frame_rgb.copy()

        for trk in tracked:
            x1, y1, x2, y2, tid = map(int, trk)
            tid = int(tid)

            vehicle_crop = crop_region(frame_rgb, x1, y1, x2, y2)
            if vehicle_crop.size == 0:
                continue

            cached = plate_cache.get(tid, {})
            age    = frame_idx - cached.get("last_frame", -9999)
            plate_text  = cached.get("text")
            plate_thumb = cached.get("crop")

            # Jalankan plate detector + OCR setiap N frame
            if age >= OCR_EVERY_N_FRAMES:
                plates = plate_detector.detect(vehicle_crop)

                if plates:
                    px1, py1, px2, py2 = plates[0]
                    new_text = ocr_reader.read_plate(vehicle_crop, px1, py1, px2, py2)

                    raw_crop = crop_region(vehicle_crop, px1, py1, px2, py2)

                    if new_text:
                        plate_text = new_text

                    plate_cache[tid] = {
                        "text":       plate_text,
                        "last_frame": frame_idx,
                        "crop":       raw_crop
                    }
                    plate_thumb = raw_crop

            # ── 4. Visualisasi ────────────────────────────────────────
            draw_vehicle_box(output_frame, x1, y1, x2, y2, tid, plate_text)

            if show_crops and plate_thumb is not None:
                embed_plate_crop(output_frame, plate_thumb, x1, y2)

        # ── 5. Tulis frame output ─────────────────────────────────────
        out_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
        writer.write(out_bgr)

        # ── 6. Preview setiap 5 frame ─────────────────────────────────
        if frame_idx % 5 == 0:
            frame_display.image(output_frame, use_container_width=True)
            progress = frame_idx / max(total_frames, 1)
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Frame {frame_idx}/{total_frames} | Kendaraan terdeteksi: {len(tracked)}")

        frame_idx += 1

    cap.release()
    writer.release()

    # Windows: beri jeda agar file handle benar-benar dilepas sebelum unlink
    time.sleep(0.5)
    try:
        os.unlink(video_path)
    except PermissionError:
        pass  # Biarkan OS membersihkan tempfile saat proses selesai

    status_text.text("✅ Selesai! Siap diunduh.")
    progress_bar.progress(1.0)

    with open(out_path, "rb") as f:
        st.download_button(
            label="⬇ Download Video Hasil",
            data=f,
            file_name="output_plate_detection.mp4",
            mime="video/mp4"
        )

    os.unlink(out_path)