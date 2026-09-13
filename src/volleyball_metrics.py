from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CENTER_LINE_Y_M, COURT_LENGTH_M, COURT_WIDTH_M, ESTIMATED_PLAYER_HEIGHT_M, OUTPUT_DIR

NEAR_SIDE_LABEL = "near_side_backview"
BALL_DIAMETER_M = 0.21
CONTACT_RADIUS_M = 1.7
CONTACT_RADIUS_PX = 85.0
CONTACT_COOLDOWN_FRAMES = 8


def select_near_side_team(df: pd.DataFrame, max_players: int = 6) -> list[int]:
    """Pick the camera-side team from a back-view clip using calibrated court positions."""
    if df.empty or "court_y_m" not in df:
        return []

    candidates = []
    for track_id, group in df.dropna(subset=["court_y_m"]).groupby("track_id"):
        median_y = float(group["court_y_m"].median())
        detections = int(len(group))
        in_near_half_ratio = float((group["court_y_m"] <= CENTER_LINE_Y_M + 0.75).mean())
        if in_near_half_ratio < 0.55:
            continue
        candidates.append(
            {
                "track_id": int(track_id),
                "median_y": median_y,
                "detections": detections,
                "near_half_ratio": in_near_half_ratio,
            }
        )

    candidates.sort(key=lambda item: (-item["detections"], item["median_y"]))
    return [item["track_id"] for item in candidates[:max_players]]


def tag_team_players(df: pd.DataFrame, team_track_ids: list[int]) -> pd.DataFrame:
    output = df.copy()
    team_ids = {int(track_id) for track_id in team_track_ids}
    output["team_player"] = output["track_id"].astype(int).isin(team_ids) if not output.empty else False
    output["team_side"] = np.where(output.get("team_player", False), NEAR_SIDE_LABEL, "other") if not output.empty else []
    return output


def estimate_player_table(df: pd.DataFrame, team_track_ids: list[int], fps: float) -> pd.DataFrame:
    rows = []
    for track_id in team_track_ids:
        group = df[df["track_id"] == track_id].dropna(subset=["court_x_m", "court_y_m"]).sort_values("frame_number")
        if group.empty:
            rows.append({"track_id": track_id, "frames": 0, "time_s": 0.0, "distance_m": 0.0, "avg_speed_mps": 0.0, "zone": "unknown"})
            continue

        distance = 0.0
        previous = None
        for row in group.itertuples(index=False):
            current = (float(row.court_x_m), float(row.court_y_m), float(row.timestamp_seconds))
            if previous is not None:
                dt = max(current[2] - previous[2], 1.0 / max(fps, 1.0))
                step = math.hypot(current[0] - previous[0], current[1] - previous[1])
                if step / dt <= 10.0:
                    distance += step
            previous = current

        median_x = float(group["court_x_m"].median())
        median_y = float(group["court_y_m"].median())
        row_zone = "front row" if median_y >= CENTER_LINE_Y_M - 3.0 else "back row"
        if median_x < COURT_WIDTH_M / 3:
            column = "left"
        elif median_x > COURT_WIDTH_M * 2 / 3:
            column = "right"
        else:
            column = "middle"
        time_s = len(group) / max(fps, 1.0)
        rows.append(
            {
                "track_id": int(track_id),
                "frames": int(len(group)),
                "time_s": round(time_s, 2),
                "distance_m": round(distance, 2),
                "avg_speed_mps": round(distance / time_s, 2) if time_s > 0 else 0.0,
                "avg_court_x_m": round(float(group["court_x_m"].mean()), 2),
                "avg_court_y_m": round(float(group["court_y_m"].mean()), 2),
                "zone": f"{row_zone} {column}",
            }
        )
    return pd.DataFrame(rows)


def ball_records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    columns = [
        "frame_number",
        "timestamp_seconds",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "ball_center_x",
        "ball_center_y",
        "court_x_m",
        "court_y_m",
        "confidence",
        "notes",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records).sort_values("frame_number").reset_index(drop=True)


def estimate_ball_record(ball_detection, frame_number: int, timestamp_seconds: float, pixel_to_court_matrix) -> dict:
    from .calibration import map_pixel_to_court

    x1, y1, x2, y2 = ball_detection.bbox
    center_x = float((x1 + x2) / 2.0)
    center_y = float((y1 + y2) / 2.0)
    court_x = court_y = np.nan
    note = "ball mapped as image centre; not a true 3D ball position"
    try:
        mapped = map_pixel_to_court([(center_x, center_y)], pixel_to_court_matrix)[0]
        court_x, court_y = float(mapped[0]), float(mapped[1])
    except Exception:
        note = "ball detected but court mapping unavailable"
    return {
        "frame_number": int(frame_number),
        "timestamp_seconds": round(float(timestamp_seconds), 4),
        "bbox_x1": float(x1),
        "bbox_y1": float(y1),
        "bbox_x2": float(x2),
        "bbox_y2": float(y2),
        "ball_center_x": center_x,
        "ball_center_y": center_y,
        "court_x_m": court_x,
        "court_y_m": court_y,
        "confidence": float(ball_detection.confidence),
        "notes": note,
    }


def estimate_contacts(player_df: pd.DataFrame, ball_df: pd.DataFrame, team_track_ids: list[int], fps: float) -> pd.DataFrame:
    columns = [
        "frame_number",
        "timestamp_seconds",
        "track_id",
        "action_guess",
        "court_x_m",
        "court_y_m",
        "ball_court_x_m",
        "ball_court_y_m",
        "contact_height_estimate_m",
        "confidence_note",
    ]
    if player_df.empty or ball_df.empty or not team_track_ids:
        return pd.DataFrame(columns=columns)

    team = player_df[player_df["track_id"].astype(int).isin({int(track_id) for track_id in team_track_ids})].copy()
    if team.empty:
        return pd.DataFrame(columns=columns)

    player_by_frame = {int(frame): group for frame, group in team.groupby("frame_number")}
    events = []
    last_contact_frame = -10_000
    previous_ball = None

    for ball in ball_df.itertuples(index=False):
        frame_number = int(ball.frame_number)
        players = player_by_frame.get(frame_number)
        if players is None or frame_number - last_contact_frame < CONTACT_COOLDOWN_FRAMES:
            previous_ball = ball
            continue

        candidates = []
        for player in players.itertuples(index=False):
            court_distance = np.inf
            if pd.notna(ball.court_x_m) and pd.notna(ball.court_y_m) and pd.notna(player.court_x_m) and pd.notna(player.court_y_m):
                court_distance = math.hypot(float(ball.court_x_m) - float(player.court_x_m), float(ball.court_y_m) - float(player.court_y_m))
            pixel_distance = math.hypot(float(ball.ball_center_x) - float(player.bbox_center_x), float(ball.ball_center_y) - float(player.bbox_center_y))
            if court_distance <= CONTACT_RADIUS_M or pixel_distance <= CONTACT_RADIUS_PX:
                candidates.append((court_distance, pixel_distance, player))

        if not candidates:
            previous_ball = ball
            continue

        candidates.sort(key=lambda item: (item[0], item[1]))
        _, _, player = candidates[0]
        bbox_height_px = max(1.0, float(player.bbox_y2) - float(player.bbox_y1))
        pixels_above_feet = max(0.0, float(player.foot_pixel_y) - float(ball.ball_center_y))
        contact_height = min(3.8, max(0.0, (pixels_above_feet / bbox_height_px) * ESTIMATED_PLAYER_HEIGHT_M))

        if pd.notna(ball.court_y_m) and float(ball.court_y_m) >= CENTER_LINE_Y_M - 1.0:
            action = "attack/block touch guess"
        elif previous_ball is not None and pd.notna(previous_ball.court_y_m) and pd.notna(ball.court_y_m):
            action = "set/pass guess" if float(ball.court_y_m) > float(previous_ball.court_y_m) else "dig/reception guess"
        else:
            action = "contact guess"

        events.append(
            {
                "frame_number": frame_number,
                "timestamp_seconds": float(ball.timestamp_seconds),
                "track_id": int(player.track_id),
                "action_guess": action,
                "court_x_m": float(player.court_x_m) if pd.notna(player.court_x_m) else np.nan,
                "court_y_m": float(player.court_y_m) if pd.notna(player.court_y_m) else np.nan,
                "ball_court_x_m": float(ball.court_x_m) if pd.notna(ball.court_x_m) else np.nan,
                "ball_court_y_m": float(ball.court_y_m) if pd.notna(ball.court_y_m) else np.nan,
                "contact_height_estimate_m": round(contact_height, 2),
                "confidence_note": "Heuristic proximity event; requires a volleyball-trained ball model for high accuracy.",
            }
        )
        last_contact_frame = frame_number
        previous_ball = ball

    return pd.DataFrame(events, columns=columns)


def estimate_team_box_score(player_table: pd.DataFrame, contacts_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in player_table.itertuples(index=False):
        player_contacts = contacts_df[contacts_df["track_id"] == int(row.track_id)] if not contacts_df.empty else pd.DataFrame()
        attacks = int(player_contacts["action_guess"].str.contains("attack", na=False).sum()) if not player_contacts.empty else 0
        sets = int(player_contacts["action_guess"].str.contains("set", na=False).sum()) if not player_contacts.empty else 0
        receptions = int(player_contacts["action_guess"].str.contains("dig|reception|pass", na=False, regex=True).sum()) if not player_contacts.empty else 0
        max_height = player_contacts["contact_height_estimate_m"].max() if not player_contacts.empty else np.nan
        rows.append(
            {
                "track_id": int(row.track_id),
                "zone": row.zone,
                "tracked_time_s": row.time_s,
                "movement_m": row.distance_m,
                "avg_speed_mps": row.avg_speed_mps,
                "contacts": int(len(player_contacts)),
                "attack_touches": attacks,
                "set_touches": sets,
                "reception_pass_touches": receptions,
                "max_contact_height_m": round(float(max_height), 2) if pd.notna(max_height) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_volleyball_outputs(player_table: pd.DataFrame, ball_df: pd.DataFrame, contacts_df: pd.DataFrame, box_score_df: pd.DataFrame) -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "team_players": OUTPUT_DIR / "team_players.csv",
        "ball_tracking": OUTPUT_DIR / "ball_tracking.csv",
        "ball_contacts": OUTPUT_DIR / "ball_contacts.csv",
        "team_box_score": OUTPUT_DIR / "team_box_score.csv",
    }
    player_table.to_csv(paths["team_players"], index=False)
    ball_df.to_csv(paths["ball_tracking"], index=False)
    contacts_df.to_csv(paths["ball_contacts"], index=False)
    box_score_df.to_csv(paths["team_box_score"], index=False)
    return paths
