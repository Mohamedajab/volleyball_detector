import pandas as pd
from src.roster import team_records


def row(frame, raw, x, y=4):
    return dict(frame_number=frame, timestamp_seconds=frame/30, track_id=raw,
                court_x_m=x, court_y_m=y, detection_confidence=0.9)


def test_crossing_people_keep_identity():
    rows = []
    for frame in range(60):
        rows += [row(frame, 11, 2+frame*0.08), row(frame, 22, 7-frame*0.08)]
    team = team_records(pd.DataFrame(rows), [11, 22])
    assert team.groupby("raw_track_id").track_id.nunique().max() == 1
    assert team[team.raw_track_id == 11].track_id.unique().tolist() == [1]


def test_far_side_and_extra_people_never_expand_roster():
    rows = [row(f, p, p) for f in range(4) for p in range(1, 9)]
    rows += [row(f, 100, 3, 12) for f in range(4)]
    team = team_records(pd.DataFrame(rows), range(1, 7))
    assert set(team.raw_track_id) == set(range(1, 7))
    assert team.groupby("frame_number").size().max() == 6


def test_short_track_fragment_recovers_but_long_gap_stays_missing():
    data = pd.DataFrame([row(0, 10, 3), row(1, 20, 3.05), row(120, 99, 3.1)])
    team = team_records(data, [10])
    assert team.raw_track_id.tolist() == [10, 20]
    assert team.track_id.tolist() == [1, 1]
