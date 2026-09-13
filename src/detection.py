from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import MODEL_PRIORITY_FILENAMES, MODELS_DIR, POSE_MODEL_FILENAME, PROJECT_ROOT

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - handled in the Streamlit UI.
    YOLO = None


PLAYER_CLASS_TOKENS = ("person", "player", "athlete")
BALL_CLASS_TOKENS = ("ball", "sports_ball", "volleyball")
POSE_COM_KEYPOINTS = (5, 6, 11, 12)  # shoulders and hips in YOLO pose format


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int | None
    class_name: str


def _candidate_model_paths(filename: str) -> list[Path]:
    return [MODELS_DIR / filename, PROJECT_ROOT / filename]


def find_detection_model_path() -> Path | None:
    for filename in MODEL_PRIORITY_FILENAMES:
        for path in _candidate_model_paths(filename):
            if path.exists():
                return path
    return None


def find_pose_model_path() -> Path | None:
    for path in _candidate_model_paths(POSE_MODEL_FILENAME):
        if path.exists():
            return path
    return None


def load_detection_model() -> tuple[Any, str]:
    """Load the preferred player detector, falling back to YOLOv8 nano if needed."""
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed. Run: pip install -r requirements.txt")

    model_path = find_detection_model_path()
    if model_path is not None:
        return YOLO(str(model_path)), str(model_path)

    # Ultralytics downloads this on first use when network is available.
    return YOLO("yolov8n.pt"), "yolov8n.pt (Ultralytics fallback)"


def load_pose_model() -> tuple[Any | None, str | None]:
    if YOLO is None:
        return None, None

    model_path = find_pose_model_path()
    if model_path is None:
        return None, None
    return YOLO(str(model_path)), str(model_path)


def load_ball_model() -> tuple[Any | None, str | None]:
    """Load a general YOLO model for ball detection when available."""
    if YOLO is None:
        return None, None

    for path in _candidate_model_paths("yolov8n.pt"):
        if path.exists():
            return YOLO(str(path)), str(path)

    try:
        return YOLO("yolov8n.pt"), "yolov8n.pt (Ultralytics ball fallback)"
    except Exception:
        return None, None


def _normalise_class_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def _names_to_dict(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {idx: str(value) for idx, value in enumerate(names)}
    return {}


def _looks_like_player_class(class_name: str) -> bool:
    normalised = _normalise_class_name(class_name)
    return any(token in normalised for token in PLAYER_CLASS_TOKENS)


def _model_has_player_classes(names: dict[int, str]) -> bool:
    return any(_looks_like_player_class(name) for name in names.values())


def detect_players(model: Any, frame: np.ndarray, confidence_threshold: float = 0.35) -> list[Detection]:
    """Run YOLO and return detections that represent players/people."""
    results = model.predict(frame, conf=confidence_threshold, verbose=False)
    if not results:
        return []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    names = _names_to_dict(getattr(result, "names", None) or getattr(model, "names", None))
    has_player_classes = _model_has_player_classes(names)

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.full(len(xyxy), -1)

    detections: list[Detection] = []
    for bbox, conf, class_id in zip(xyxy, confidences, classes):
        class_id_int = int(class_id) if class_id >= 0 else None
        class_name = names.get(class_id_int, "player")

        # COCO models should only keep people. Custom volleyball models often have
        # one unlabeled player class, so when no known person/player class exists,
        # we accept all boxes as candidate players.
        if has_player_classes and not _looks_like_player_class(class_name):
            continue

        detections.append(
            Detection(
                bbox=tuple(float(value) for value in bbox),
                confidence=float(conf),
                class_id=class_id_int,
                class_name=class_name,
            )
        )
    return detections


def estimate_pose_vertical_signal(pose_model: Any, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> float | None:
    """Estimate a selected player's centre-of-mass y coordinate from pose keypoints."""
    if pose_model is None:
        return None

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width - 1, x2), min(height - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    try:
        results = pose_model.predict(crop, conf=0.25, verbose=False)
    except Exception:
        return None

    if not results:
        return None
    keypoints = getattr(results[0], "keypoints", None)
    if keypoints is None or getattr(keypoints, "xy", None) is None or len(keypoints.xy) == 0:
        return None

    points = keypoints.xy[0].detach().cpu().numpy()
    valid_y_values: list[float] = []
    for index in POSE_COM_KEYPOINTS:
        if index >= len(points):
            continue
        px, py = points[index]
        if px > 0 and py > 0:
            valid_y_values.append(float(py + y1))

    if len(valid_y_values) < 2:
        return None
    return float(np.mean(valid_y_values))




def _looks_like_ball_class(class_name: str) -> bool:
    normalised = _normalise_class_name(class_name)
    return any(token in normalised for token in BALL_CLASS_TOKENS)


def detect_balls(model: Any, frame: np.ndarray, confidence_threshold: float = 0.15) -> list[Detection]:
    """Detect ball candidates. Works best with a model trained on volleyballs."""
    if model is None:
        return []

    try:
        results = model.predict(frame, conf=confidence_threshold, verbose=False)
    except Exception:
        return []
    if not results:
        return []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    names = _names_to_dict(getattr(result, "names", None) or getattr(model, "names", None))
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.full(len(xyxy), -1)

    detections: list[Detection] = []
    for bbox, conf, class_id in zip(xyxy, confidences, classes):
        class_id_int = int(class_id) if class_id >= 0 else None
        class_name = names.get(class_id_int, "")
        if not _looks_like_ball_class(class_name):
            continue

        x1, y1, x2, y2 = [float(value) for value in bbox]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width < 3 or height < 3:
            continue
        detections.append(Detection((x1, y1, x2, y2), float(conf), class_id_int, class_name))

    return sorted(detections, key=lambda item: item.confidence, reverse=True)
