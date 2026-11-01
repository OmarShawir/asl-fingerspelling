"""Use MediaPipe to locate a single hand in the unmirrored camera frame."""

import time
from http.client import HTTPException
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import is_zipfile

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


TRACKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_tracker(path: Path) -> Path:
    """Download the official tracker once. An interrupted download can be retried."""
    if path.is_file() and is_zipfile(path):
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading Google's hand tracker...", flush=True)
    try:
        with urlopen(TRACKER_URL, timeout=45) as response:
            bundle = response.read()
        if not is_zipfile(BytesIO(bundle)):
            raise ValueError("Incomplete tracker bundle")
    except (OSError, HTTPException, ValueError) as error:
        raise SystemExit(
            f"Download failed ({error}). Retry, or save the model from\n{TRACKER_URL}\nto {path}."
        ) from error

    temp = NamedTemporaryFile(dir=path.parent, delete=False)
    temp_path = Path(temp.name)
    try:
        with temp:
            temp.write(bundle)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
    return path


def open_tracker(settings):
    model_path = ensure_tracker(settings["model_path"])
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=settings["detection_confidence"],
        min_hand_presence_confidence=settings["presence_confidence"],
        min_tracking_confidence=settings["tracking_confidence"],
    )
    return vision.HandLandmarker.create_from_options(options)


def next_timestamp_ms(previous_ms):
    """MediaPipe VIDEO mode requires increasing timestamps."""
    return max(previous_ms + 1, time.monotonic_ns() // 1_000_000)


def extract_points(tracker, frame, timestamp_ms):
    """Return (21, 3) image-space points in MediaPipe order, or NaNs for a missed hand."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = tracker.detect_for_video(image, timestamp_ms)
    if not result.hand_landmarks:
        return np.full((21, 3), np.nan, dtype=np.float32)
    hand = result.hand_landmarks[0]
    return np.array([(p.x, p.y, p.z) for p in hand], dtype=np.float32)


def draw_hand(frame, points):
    """Draw on the preview after extracting the original coordinates."""
    if not np.isfinite(points).all():
        return
    height, width = frame.shape[:2]
    pixels = (points[:, :2] * (width - 1, height - 1)).astype(int)
    for edge in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, pixels[edge.start], pixels[edge.end], (60, 210, 90), 2, cv2.LINE_AA)
    for point in pixels:
        cv2.circle(frame, point, 4, (30, 185, 255), -1, cv2.LINE_AA)
