from __future__ import annotations

import cv2
import numpy as np

from .detection import Detection


def detect_ball_candidates_classical(frame: np.ndarray, previous_frame: np.ndarray | None = None) -> list[Detection]:
    """Find small ball-like candidates using motion, brightness, and circularity.

    This is not a replacement for a trained volleyball detector, but it gives the
    app a practical fallback for back-view practice footage where the ball is a
    small bright moving object.
    """
    if frame is None or frame.size == 0:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    masks = []
    bright = cv2.inRange(blur, 170, 255)
    masks.append(bright)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_sat_bright = cv2.inRange(hsv, np.array([0, 0, 135]), np.array([179, 95, 255]))
    masks.append(low_sat_bright)

    if previous_frame is not None and previous_frame.shape == frame.shape:
        prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        prev_blur = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        motion = cv2.absdiff(blur, prev_blur)
        _, motion_mask = cv2.threshold(motion, 18, 255, cv2.THRESH_BINARY)
        masks.append(cv2.bitwise_and(low_sat_bright, motion_mask))

    combined = masks[0]
    for mask in masks[1:]:
        combined = cv2.bitwise_or(combined, mask)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = frame.shape[:2]
    frame_area = width * height
    candidates: list[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 12 or area > frame_area * 0.004:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 4 or h < 4 or w > width * 0.12 or h > height * 0.12:
            continue
        aspect = w / max(h, 1)
        if aspect < 0.45 or aspect > 2.2:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
        if circularity < 0.2:
            continue
        score = float(min(0.99, 0.25 + circularity * 0.5 + min(area / 250.0, 0.24)))
        candidates.append(Detection((float(x), float(y), float(x + w), float(y + h)), score, None, "ball_classical"))

    return sorted(candidates, key=lambda item: item.confidence, reverse=True)
