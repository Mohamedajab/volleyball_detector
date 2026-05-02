from __future__ import annotations

import cv2
import numpy as np

from .config import (
    ATTACK_LINE_Y_M,
    CENTER_LINE_Y_M,
    COURT_CORNERS_M,
    COURT_LENGTH_M,
    COURT_WIDTH_M,
)


def validate_corner_points(points: list[tuple[float, float]], frame_shape: tuple[int, int] | None = None) -> tuple[bool, str]:
    if len(points) != 4:
        return False, "Exactly four court corners are required."

    array = np.asarray(points, dtype=np.float32)
    if not np.isfinite(array).all():
        return False, "Corner coordinates must be finite numbers."

    if frame_shape is not None:
        height, width = frame_shape[:2]
        if np.any(array[:, 0] < 0) or np.any(array[:, 0] >= width) or np.any(array[:, 1] < 0) or np.any(array[:, 1] >= height):
            return False, "All corners must be inside the video frame."

    distances = np.linalg.norm(array[:, None, :] - array[None, :, :], axis=2)
    distances += np.eye(4) * 1e9
    if np.min(distances) < 8:
        return False, "Two or more court corners are too close together."

    x = array[:, 0]
    y = array[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if area < 200:
        return False, "Selected corners do not form a large enough court polygon."

    return True, "Court corners look valid."


def compute_homography(points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    ok, message = validate_corner_points(points)
    if not ok:
        raise ValueError(message)

    source = np.asarray(points, dtype=np.float32)
    destination = np.asarray(COURT_CORNERS_M, dtype=np.float32)
    pixel_to_court = cv2.getPerspectiveTransform(source, destination)
    court_to_pixel = cv2.getPerspectiveTransform(destination, source)
    return pixel_to_court, court_to_pixel


def map_pixel_to_court(pixel_points: list[tuple[float, float]] | np.ndarray, pixel_to_court_matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(points, pixel_to_court_matrix)
    return mapped.reshape(-1, 2)


def map_court_to_pixel(court_points: list[tuple[float, float]] | np.ndarray, court_to_pixel_matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(court_points, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(points, court_to_pixel_matrix)
    return mapped.reshape(-1, 2)


def court_line_segments() -> list[tuple[tuple[float, float], tuple[float, float], str]]:
    return [
        ((0.0, 0.0), (COURT_WIDTH_M, 0.0), "baseline"),
        ((COURT_WIDTH_M, 0.0), (COURT_WIDTH_M, COURT_LENGTH_M), "sideline"),
        ((COURT_WIDTH_M, COURT_LENGTH_M), (0.0, COURT_LENGTH_M), "baseline"),
        ((0.0, COURT_LENGTH_M), (0.0, 0.0), "sideline"),
        ((0.0, CENTER_LINE_Y_M), (COURT_WIDTH_M, CENTER_LINE_Y_M), "centre"),
        ((0.0, ATTACK_LINE_Y_M[0]), (COURT_WIDTH_M, ATTACK_LINE_Y_M[0]), "attack"),
        ((0.0, ATTACK_LINE_Y_M[1]), (COURT_WIDTH_M, ATTACK_LINE_Y_M[1]), "attack"),
    ]


def draw_court_lines(frame: np.ndarray, court_to_pixel_matrix: np.ndarray, thickness: int = 2) -> np.ndarray:
    annotated = frame.copy()
    for start, end, line_type in court_line_segments():
        pixel_points = map_court_to_pixel([start, end], court_to_pixel_matrix)
        p1 = tuple(np.round(pixel_points[0]).astype(int))
        p2 = tuple(np.round(pixel_points[1]).astype(int))
        color = (0, 255, 255) if line_type in {"baseline", "sideline"} else (255, 255, 255)
        line_thickness = thickness + 1 if line_type == "centre" else thickness
        cv2.line(annotated, p1, p2, color, line_thickness, cv2.LINE_AA)
    return annotated


def draw_calibration_points(frame: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
    labels = ["near left", "near right", "far right", "far left"]
    annotated = frame.copy()
    for index, (x, y) in enumerate(points):
        point = (int(round(x)), int(round(y)))
        cv2.circle(annotated, point, 7, (0, 180, 255), -1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"{index + 1}: {labels[index]}",
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
    if len(points) > 1:
        int_points = np.asarray(points, dtype=np.int32)
        cv2.polylines(annotated, [int_points], isClosed=len(points) == 4, color=(0, 180, 255), thickness=2)
    return annotated

