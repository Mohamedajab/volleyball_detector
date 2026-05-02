from src.detection import Detection
from src.tracking import CentroidTracker


def test_centroid_tracker_keeps_id_for_nearby_detection():
    tracker = CentroidTracker(max_distance_px=50, max_missing_frames=2)

    first = tracker.update([Detection((0, 0, 20, 40), 0.9, 0, "person")], frame_number=0, timestamp_seconds=0.0)
    second = tracker.update([Detection((4, 0, 24, 40), 0.9, 0, "person")], frame_number=1, timestamp_seconds=0.033)

    assert first[0].track_id == second[0].track_id
