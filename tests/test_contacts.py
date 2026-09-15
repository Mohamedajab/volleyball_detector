import numpy as np
import pandas as pd
from src.volleyball_metrics import estimate_contacts


def test_contact_requires_hand_proximity_and_reports_calibrated_height():
    players = pd.DataFrame([{
        "frame_number": frame, "timestamp_seconds": frame / 30,
        "track_id": 1, "raw_track_id": 8, "roster_id": "P1",
        "court_x_m": 4.0, "court_y_m": 5.0,
        "bbox_x1": 400, "bbox_y1": 300, "bbox_x2": 500, "bbox_y2": 650,
        "bbox_center_x": 450, "bbox_center_y": 475,
        "left_wrist_x": 450, "left_wrist_y": 400,
        "right_wrist_x": 460, "right_wrist_y": 400,
    } for frame in (9, 10, 11)])
    balls = pd.DataFrame([
        {"frame_number": 9, "timestamp_seconds": .3, "ball_center_x": 430,
         "ball_center_y": 410, "court_x_m": 4, "court_y_m": 5, "confidence": .9},
        {"frame_number": 10, "timestamp_seconds": 1/3, "ball_center_x": 450,
         "ball_center_y": 400, "court_x_m": 4, "court_y_m": 5, "confidence": .9},
        {"frame_number": 11, "timestamp_seconds": 11/30, "ball_center_x": 446,
         "ball_center_y": 370, "court_x_m": 4, "court_y_m": 5, "confidence": .9},
    ])
    projection = np.array([[100, 0, 0, 50], [0, 0, -100, 700], [0, 0, 0, 1]], dtype=float)
    events = estimate_contacts(players, balls, ["P1"], 30, projection)
    assert len(events) == 1
    assert events.iloc[0].action_guess == "set contact candidate"
    assert 2.8 < events.iloc[0].contact_height_estimate_m < 3.2
    assert events.iloc[0].event_confidence > .5


def test_ball_passing_far_from_hands_is_not_contact():
    players = pd.DataFrame([{
        "frame_number": 2, "timestamp_seconds": .1, "track_id": 1,
        "raw_track_id": 1, "roster_id": "P1", "court_x_m": 4,
        "court_y_m": 5, "bbox_x1": 100, "bbox_y1": 100, "bbox_x2": 200,
        "bbox_y2": 400, "bbox_center_x": 150, "bbox_center_y": 250,
        "left_wrist_x": 160, "left_wrist_y": 160,
        "right_wrist_x": 140, "right_wrist_y": 160,
    }])
    balls = pd.DataFrame([
        {"frame_number": f, "timestamp_seconds": f/30, "ball_center_x": 600+f*5,
         "ball_center_y": 50, "court_x_m": 1, "court_y_m": 1, "confidence": .9}
        for f in (1, 2, 3)
    ])
    assert estimate_contacts(players, balls, ["P1"], 30).empty
