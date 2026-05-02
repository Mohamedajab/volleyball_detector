from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .config import (
    DEFAULT_FPS,
    MODELS_DIR,
    OUTPUT_DIR,
    SAMPLE_DATA_DIR,
    SUPPORTED_VIDEO_TYPES,
    UPLOADS_DIR,
)


def ensure_workspace_directories() -> None:
    """Create runtime folders used by the app."""
    for directory in (UPLOADS_DIR, OUTPUT_DIR, MODELS_DIR, SAMPLE_DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def is_supported_video(filename: str) -> bool:
    suffix = Path(filename).suffix.lower().lstrip(".")
    return suffix in SUPPORTED_VIDEO_TYPES


def _safe_stem(stem: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return cleaned or "uploaded_video"


def save_uploaded_video(uploaded_file, upload_dir: Path = UPLOADS_DIR) -> Path:
    """Persist a Streamlit uploaded video and return the local path."""
    ensure_workspace_directories()
    if not is_supported_video(uploaded_file.name):
        supported = ", ".join(SUPPORTED_VIDEO_TYPES)
        raise ValueError(f"Unsupported video type. Please upload one of: {supported}.")

    suffix = Path(uploaded_file.name).suffix.lower()
    payload = uploaded_file.getvalue()
    digest = hashlib.sha1(payload).hexdigest()[:10]
    output_path = upload_dir / f"{_safe_stem(Path(uploaded_file.name).stem)}_{digest}{suffix}"
    output_path.write_bytes(payload)
    return output_path


def read_video_metadata(video_path: Path | str) -> dict:
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    return {
        "fps": float(fps),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": float(duration),
        "path": str(path),
    }


def extract_frame(video_path: Path | str, frame_number: int = 0) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frame_number = max(0, int(frame_number))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def frame_generator(video_path: Path | str) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_number = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame_number, frame
            frame_number += 1
    finally:
        cap.release()


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def resize_for_display(frame: np.ndarray, max_width: int = 900) -> tuple[np.ndarray, float]:
    """Resize a frame for UI display and return (resized_frame, scale)."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame.copy(), 1.0

    scale = max_width / float(width)
    resized = cv2.resize(frame, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale

