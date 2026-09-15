import pandas as pd
from src.stat_sheet import build_stat_sheet


def test_only_reviewed_events_count_and_kills_are_attempts():
    events = pd.DataFrame([
        dict(Player="P1", Event="Kill", Reviewed=True),
        dict(Player="P1", Event="Attack error", Reviewed=True),
        dict(Player="P1", Event="Attack attempt", Reviewed=True),
        dict(Player="P1", Event="Ace", Reviewed=False),
    ])
    row = build_stat_sheet(events, ["P1"]).iloc[0]
    assert row["Attack attempts"] == 3
    assert row["Hitting percentage"] == 0
    assert row["Aces"] == 0
