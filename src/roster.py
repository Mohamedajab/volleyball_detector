"""Six persistent people, independent of changing volleyball court zones."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROSTER_ORDER = [f"P{i}" for i in range(1, 7)]


def assign_near_side_roster_slots(df: pd.DataFrame, seed_ids=None) -> pd.DataFrame:
    """Lock the roster once, with gated global assignment for short gaps.

    Raw tracker IDs provide continuity. Ambiguous or long gaps stay unassigned;
    no replacement people are automatically added to the locked roster.
    """
    out = df.copy().reset_index(drop=True)
    if "raw_track_id" not in out:
        out["raw_track_id"] = out.get("track_id", pd.Series(dtype=int))
    out["team_player"] = False
    out["team_side"] = "other"
    out["roster_id"] = ""
    out["roster_zone"] = ""
    if out.empty:
        return out
    valid = out[np.isfinite(out.court_x_m) & np.isfinite(out.court_y_m)]
    valid = valid[valid.court_x_m.between(-0.75, 9.75) & valid.court_y_m.between(-1.5, 9.0)]
    if valid.empty:
        return out
    groups = list(valid.groupby("frame_number", sort=True))
    if seed_ids is None:
        initial = next((g for _, g in groups if len(g) >= 6), groups[0][1])
        initial = initial.sort_values("detection_confidence", ascending=False).head(6)
        seed_ids = initial.sort_values(["court_x_m", "court_y_m"]).raw_track_id.astype(int).tolist()
    seed_ids = list(dict.fromkeys(map(int, seed_ids)))
    if not 1 <= len(seed_ids) <= 6:
        raise ValueError("Select between one and six near-side players.")
    states = {}
    for _, group in groups:
        rows = list(group.itertuples())
        costs = np.full((len(seed_ids), len(rows) + len(seed_ids)), 100.0)
        for slot, seed in enumerate(seed_ids):
            state = states.get(slot)
            for j, row in enumerate(rows):
                point = np.array([row.court_x_m, row.court_y_m], dtype=float)
                raw = int(row.raw_track_id)
                if state is None:
                    if raw == seed:
                        costs[slot, j] = 0.0
                    continue
                dt = float(row.timestamp_seconds) - state["time"]
                if dt <= 0 or dt > 2.0:
                    continue
                if np.linalg.norm(point - state["point"]) > 0.6 + 9.0 * dt:
                    continue
                same = raw == state["raw"]
                if not same and dt > 0.75:
                    continue
                prediction = state["point"] + state["velocity"] * min(dt, 0.3)
                costs[slot, j] = np.linalg.norm(point - prediction) + (0.0 if same else 1.5)
            costs[slot, len(rows) + slot] = 4.0
        slots, columns = linear_sum_assignment(costs)
        for slot, j in zip(slots, columns):
            if j >= len(rows) or costs[slot, j] >= 4.0:
                continue
            row = rows[j]
            point = np.array([row.court_x_m, row.court_y_m], dtype=float)
            previous = states.get(slot)
            velocity = np.zeros(2)
            if previous is not None:
                dt = float(row.timestamp_seconds) - previous["time"]
                velocity = 0.5 * previous["velocity"] + 0.5 * (point - previous["point"]) / dt
            states[slot] = dict(point=point, velocity=velocity, raw=int(row.raw_track_id), time=float(row.timestamp_seconds))
            out.at[row.Index, "team_player"] = True
            out.at[row.Index, "team_side"] = "near_side_backview"
            out.at[row.Index, "roster_id"] = ROSTER_ORDER[slot]
    return out


def roster_ids_from_dataframe(df):
    return [value for value in ROSTER_ORDER if "roster_id" in df and value in set(df.roster_id)]


def team_records(df, seed_ids=None):
    result = assign_near_side_roster_slots(df, seed_ids)
    result = result[result.team_player].copy()
    result["track_id"] = result.roster_id.str[1:].astype(int)
    return result
