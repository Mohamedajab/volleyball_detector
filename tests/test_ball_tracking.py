import cv2
import numpy as np
from src.ball_tracking import BallTracker, detect_ball_candidates_classical
from src.detection import Detection


def test_stationary_bright_object_is_not_a_moving_ball():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(frame, (100, 100), 8, (255, 255, 255), -1)
    assert detect_ball_candidates_classical(frame, frame.copy()) == []


def test_ball_rejects_teleport_and_requires_confirmation():
    tracker = BallTracker()
    def ball(x):
        return Detection((x, 50, x+10, 60), 0.8, 32, "sports ball")
    assert tracker.update([ball(50)], [], 0, (480, 640)) is None
    assert tracker.update([ball(54)], [], 1/30, (480, 640)) is not None
    assert tracker.update([ball(500)], [], 2/30, (480, 640)) is None
