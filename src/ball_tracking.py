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

    # Motion must restrict the mask; OR-ing it with brightness discards it.
    if previous_frame is None:
        return []
    combined = masks[-1]
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


class BallTracker:
    """Associate candidates over time; never report extrapolated detections."""

    def __init__(self):
        self.position = None
        self.velocity = np.zeros(2)
        self.time = None
        self.confirmations = 0

    def update(self, detections, classical, timestamp, shape):
        diagonal = float(np.hypot(shape[0], shape[1]))
        if self.time is not None and timestamp - self.time > 0.25:
            self.position = None
            self.confirmations = 0
        candidates = detections if self.position is None else [*detections, *classical]
        if not candidates:
            return None
        dt = max(timestamp - self.time, 1 / 120) if self.time is not None else 1 / 30
        prediction = self.position + self.velocity * dt if self.position is not None else None
        ranked = []
        for detection in candidates:
            x1, y1, x2, y2 = detection.bbox
            center = np.array([(x1+x2)/2, (y1+y2)/2])
            distance = np.linalg.norm(center - prediction) if prediction is not None else 0.0
            gate = diagonal * min(0.18, 0.025 + dt * 1.5)
            if distance > gate:
                continue
            penalty = 0.5 if detection.class_name == "ball_classical" else 0.0
            ranked.append((distance / gate + penalty - detection.confidence, detection, center))
        if not ranked:
            return None
        _, detection, center = min(ranked, key=lambda value: value[0])
        if self.position is not None:
            self.velocity = 0.5 * self.velocity + 0.5 * (center-self.position) / dt
        else:
            self.velocity = np.zeros(2)
        self.position, self.time = center, timestamp
        self.confirmations += 1
        return detection if self.confirmations >= 2 else None
