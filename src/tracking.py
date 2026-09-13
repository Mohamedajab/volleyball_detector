from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .calibration import map_pixel_to_court
from .config import COURT_BOUNDARY_MARGIN_M, COURT_LENGTH_M, COURT_WIDTH_M, DEFAULT_FPS
from .detection import Detection, detect_players, estimate_pose_vertical_signal


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    centroid: tuple[float, float]
    confidence: float
    class_name: str
    missed_frames: int = 0
    last_frame: int = -1


@dataclass(frozen=True)
class TrackObservation:
    frame_number: int
    timestamp_seconds: float
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_name: str


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))


def bbox_foot_point(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, _, x2, y2 = bbox
    return (float((x1 + x2) / 2.0), float(y2))


class CentroidTracker:
    """Small fallback tracker using nearest-neighbour matching between frames."""

    def __init__(self, max_distance_px: float = 90.0, max_missing_frames: int = 18):
        self.max_distance_px = float(max_distance_px)
        self.max_missing_frames = int(max_missing_frames)
        self.next_track_id = 1
        self.tracks: dict[int, Track] = {}

    def reset(self) -> None:
        self.next_track_id = 1
        self.tracks.clear()

    def _create_track(self, detection: Detection, frame_number: int) -> Track:
        track = Track(
            track_id=self.next_track_id,
            bbox=detection.bbox,
            centroid=bbox_foot_point(detection.bbox),
            confidence=detection.confidence,
            class_name=detection.class_name,
            missed_frames=0,
            last_frame=frame_number,
        )
        self.tracks[track.track_id] = track
        self.next_track_id += 1
        return track

    def update(self, detections: list[Detection], frame_number: int, timestamp_seconds: float) -> list[TrackObservation]:
        for track in self.tracks.values():
            track.missed_frames += 1

        matches: list[tuple[int, int]] = []
        if self.tracks and detections:
            candidate_distances: list[tuple[float, int, int]] = []
            for track_id, track in self.tracks.items():
                for detection_index, detection in enumerate(detections):
                    distance = float(np.linalg.norm(np.asarray(track.centroid) - np.asarray(bbox_foot_point(detection.bbox))))
                    candidate_distances.append((distance, track_id, detection_index))

            used_tracks: set[int] = set()
            used_detections: set[int] = set()
            for distance, track_id, detection_index in sorted(candidate_distances, key=lambda item: item[0]):
                if distance > self.max_distance_px:
                    continue
                if track_id in used_tracks or detection_index in used_detections:
                    continue
                used_tracks.add(track_id)
                used_detections.add(detection_index)
                matches.append((track_id, detection_index))

        matched_track_ids: set[int] = set()
        matched_detection_indices: set[int] = set()
        observations: list[TrackObservation] = []

        for track_id, detection_index in matches:
            detection = detections[detection_index]
            track = self.tracks[track_id]
            track.bbox = detection.bbox
            track.centroid = bbox_foot_point(detection.bbox)
            track.confidence = detection.confidence
            track.class_name = detection.class_name
            track.missed_frames = 0
            track.last_frame = frame_number
            matched_track_ids.add(track_id)
            matched_detection_indices.add(detection_index)
            observations.append(
                TrackObservation(frame_number, timestamp_seconds, track_id, detection.bbox, detection.confidence, detection.class_name)
            )

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indices:
                continue
            track = self._create_track(detection, frame_number)
            matched_track_ids.add(track.track_id)
            observations.append(
                TrackObservation(frame_number, timestamp_seconds, track.track_id, detection.bbox, detection.confidence, detection.class_name)
            )

        stale_ids = [track_id for track_id, track in self.tracks.items() if track.missed_frames > self.max_missing_frames]
        for track_id in stale_ids:
            del self.tracks[track_id]

        return sorted(observations, key=lambda observation: observation.track_id)


def _names_to_dict(names) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def _is_player_class(class_name: str, names: dict[int, str]) -> bool:
    lowered = class_name.lower()
    known_player_labels = ("person", "player", "athlete")
    has_known_player_label = any(any(token in value.lower() for token in known_player_labels) for value in names.values())
    return not has_known_player_label or any(token in lowered for token in known_player_labels)


def _observations_from_tracked_result(result, model, frame_number: int, timestamp_seconds: float) -> list[TrackObservation]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0 or getattr(boxes, "id", None) is None:
        return []

    names = _names_to_dict(getattr(result, "names", None) or getattr(model, "names", None))
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.full(len(xyxy), -1)
    track_ids = boxes.id.detach().cpu().numpy()

    observations: list[TrackObservation] = []
    for bbox, confidence, class_id, track_id in zip(xyxy, confidences, classes, track_ids):
        class_id_int = int(class_id) if class_id >= 0 else None
        class_name = names.get(class_id_int, "player")
        if not _is_player_class(class_name, names):
            continue
        observations.append(
            TrackObservation(
                frame_number=frame_number,
                timestamp_seconds=timestamp_seconds,
                track_id=int(track_id),
                bbox=tuple(float(value) for value in bbox),
                confidence=float(confidence),
                class_name=class_name,
            )
        )
    return sorted(observations, key=lambda observation: observation.track_id)


def _court_note(court_x: float | None, court_y: float | None) -> str:
    if court_x is None or court_y is None:
        return "court mapping unavailable"
    margin = COURT_BOUNDARY_MARGIN_M
    if court_x < -margin or court_x > COURT_WIDTH_M + margin or court_y < -margin or court_y > COURT_LENGTH_M + margin:
        return "mapped outside court bounds"
    return ""


def observation_to_record(
    observation: TrackObservation,
    pixel_to_court_matrix: np.ndarray,
    frame: np.ndarray | None = None,
    pose_model=None,
    selected_track_id: int | None = None,
) -> dict:
    x1, y1, x2, y2 = observation.bbox
    center_x, center_y = bbox_center(observation.bbox)
    foot_x, foot_y = bbox_foot_point(observation.bbox)

    court_x = court_y = None
    try:
        mapped = map_pixel_to_court([(foot_x, foot_y)], pixel_to_court_matrix)[0]
        court_x, court_y = float(mapped[0]), float(mapped[1])
    except Exception:
        pass

    vertical_signal_y = center_y
    vertical_signal_source = "bbox_center"
    if selected_track_id is not None and observation.track_id == selected_track_id and frame is not None and pose_model is not None:
        pose_signal = estimate_pose_vertical_signal(pose_model, frame, observation.bbox)
        if pose_signal is not None and np.isfinite(pose_signal):
            vertical_signal_y = float(pose_signal)
            vertical_signal_source = "pose_com"

    return {
        "frame_number": int(observation.frame_number),
        "timestamp_seconds": round(float(observation.timestamp_seconds), 4),
        "track_id": int(observation.track_id),
        "bbox_x1": float(x1),
        "bbox_y1": float(y1),
        "bbox_x2": float(x2),
        "bbox_y2": float(y2),
        "bbox_center_x": float(center_x),
        "bbox_center_y": float(center_y),
        "foot_pixel_x": float(foot_x),
        "foot_pixel_y": float(foot_y),
        "court_x_m": court_x,
        "court_y_m": court_y,
        "detection_confidence": float(observation.confidence),
        "class_name": observation.class_name,
        "vertical_signal_y": float(vertical_signal_y),
        "vertical_signal_source": vertical_signal_source,
        "notes": _court_note(court_x, court_y),
    }


def run_tracking_on_video(
    video_path: str,
    detector_model,
    pixel_to_court_matrix: np.ndarray,
    confidence_threshold: float,
    max_distance_px: float,
    max_missing_frames: int,
    frame_limit: int | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    pose_model=None,
    selected_track_id: int | None = None,
    tracker_backend: str = "bytetrack.yaml",
) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames_to_process = min(total_frames, frame_limit) if frame_limit else total_frames
    frames_to_process = max(frames_to_process, 0)

    use_yolo_tracker = tracker_backend != "centroid fallback"
    tracker = CentroidTracker(max_distance_px=max_distance_px, max_missing_frames=max_missing_frames)
    records: list[dict] = []
    frame_number = 0
    yolo_tracker_failed = False

    try:
        while True:
            if frame_limit is not None and frame_number >= frame_limit:
                break

            ok, frame = cap.read()
            if not ok:
                break

            timestamp = frame_number / fps if fps > 0 else frame_number / DEFAULT_FPS
            observations: list[TrackObservation] = []
            if use_yolo_tracker and not yolo_tracker_failed:
                try:
                    results = detector_model.track(
                        frame,
                        persist=frame_number > 0,
                        tracker=tracker_backend,
                        conf=confidence_threshold,
                        verbose=False,
                    )
                    observations = _observations_from_tracked_result(results[0], detector_model, frame_number, timestamp) if results else []
                except Exception:
                    yolo_tracker_failed = True
                    observations = []

            if not use_yolo_tracker or yolo_tracker_failed:
                detections = detect_players(detector_model, frame, confidence_threshold=confidence_threshold)
                observations = tracker.update(detections, frame_number, timestamp)

            for observation in observations:
                records.append(
                    observation_to_record(
                        observation=observation,
                        pixel_to_court_matrix=pixel_to_court_matrix,
                        frame=frame,
                        pose_model=pose_model,
                        selected_track_id=selected_track_id,
                    )
                )

            frame_number += 1
            if progress_callback and frames_to_process > 0:
                backend_label = "centroid fallback" if yolo_tracker_failed or not use_yolo_tracker else tracker_backend
                progress_callback(
                    min(frame_number / frames_to_process, 1.0),
                    f"Tracking frame {frame_number} of {frames_to_process} ({backend_label})",
                )
    finally:
        cap.release()

    return records

