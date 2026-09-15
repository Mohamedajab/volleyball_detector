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


from .roster import assign_near_side_roster_slots, roster_ids_from_dataframe


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


def estimate_player_table(df: pd.DataFrame, player_ids: list, fps: float, id_column: str = "track_id") -> pd.DataFrame:
    rows = []
    for player_id in player_ids:
        group = df[df[id_column].astype(str) == str(player_id)].dropna(subset=["court_x_m", "court_y_m"]).sort_values("frame_number")
        if group.empty:
            rows.append({id_column: player_id, "frames": 0, "time_s": 0.0, "distance_m": 0.0, "avg_speed_mps": 0.0, "zone": "unknown"})
            continue

        distance = 0.0
        previous = None
        for row in group.itertuples(index=False):
            current = (float(row.court_x_m), float(row.court_y_m), float(row.timestamp_seconds))
            if previous is not None:
                dt = max(current[2] - previous[2], 1.0 / max(fps, 1.0))
                step = math.hypot(current[0] - previous[0], current[1] - previous[1])
                if dt <= 0.25 and step / dt <= 10.0:
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
        roster_zone = ""
        if "roster_zone" in group and group["roster_zone"].astype(str).str.len().any():
            roster_zone = group["roster_zone"].mode().iloc[0]
        raw_ids = ""
        if "raw_track_id" in group:
            raw_ids = ",".join(str(int(value)) for value in sorted(group["raw_track_id"].dropna().astype(int).unique())[:12])
        rows.append(
            {
                id_column: player_id,
                "display_id": str(player_id),
                "raw_track_ids_seen": raw_ids,
                "frames": int(len(group)),
                "time_s": round(time_s, 2),
                "distance_m": round(distance, 2),
                "avg_speed_mps": round(distance / time_s, 2) if time_s > 0 else 0.0,
                "avg_court_x_m": round(float(group["court_x_m"].mean()), 2),
                "avg_court_y_m": round(float(group["court_y_m"].mean()), 2),
                "zone": roster_zone or f"{row_zone} {column}",
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


def estimate_contacts(
    player_df: pd.DataFrame,
    ball_df: pd.DataFrame,
    player_ids: list,
    fps: float,
    projection_matrix_3d=None,
) -> pd.DataFrame:
    """Find reviewable hand-ball contacts from proximity and trajectory change."""
    from .camera_3d import estimate_height_above_ground

    columns = [
        "frame_number", "timestamp_seconds", "player_id", "raw_track_id",
        "action_guess", "court_x_m", "court_y_m", "ball_court_x_m",
        "ball_court_y_m", "contact_height_estimate_m",
        "height_reprojection_error_px", "hand_distance_px",
        "trajectory_change_score", "event_confidence", "confidence_note",
    ]
    if player_df.empty or ball_df.empty or not player_ids:
        return pd.DataFrame(columns=columns)

    id_column = "roster_id" if "roster_id" in player_df else "track_id"
    wanted = {str(value) for value in player_ids}
    team = player_df[player_df[id_column].astype(str).isin(wanted)].copy()
    if team.empty:
        return pd.DataFrame(columns=columns)

    balls = ball_df.sort_values("frame_number").reset_index(drop=True)
    players_by_frame = {int(frame): group for frame, group in team.groupby("frame_number")}
    events = []
    last_contact_frame = -10_000

    for index in range(1, len(balls) - 1):
        ball = balls.iloc[index]
        before, after = balls.iloc[index - 1], balls.iloc[index + 1]
        frame_number = int(ball.frame_number)
        if frame_number - last_contact_frame < CONTACT_COOLDOWN_FRAMES:
            continue
        if float(ball.timestamp_seconds - before.timestamp_seconds) > 0.2:
            continue
        if float(after.timestamp_seconds - ball.timestamp_seconds) > 0.2:
            continue

        incoming = np.array([
            ball.ball_center_x - before.ball_center_x,
            ball.ball_center_y - before.ball_center_y,
        ], dtype=float)
        outgoing = np.array([
            after.ball_center_x - ball.ball_center_x,
            after.ball_center_y - ball.ball_center_y,
        ], dtype=float)
        incoming_norm = float(np.linalg.norm(incoming))
        outgoing_norm = float(np.linalg.norm(outgoing))
        if incoming_norm < 1 or outgoing_norm < 1:
            continue
        direction_change = 1.0 - float(
            np.clip((incoming @ outgoing) / (incoming_norm * outgoing_norm), -1.0, 1.0)
        )
        speed_change = abs(outgoing_norm - incoming_norm) / max(incoming_norm, outgoing_norm)
        trajectory_score = float(np.clip(0.65 * direction_change + 0.35 * speed_change, 0, 1))

        nearby = []
        for offset in (0, -1, 1):
            group = players_by_frame.get(frame_number + offset)
            if group is not None:
                nearby.append(group)
        if not nearby:
            continue
        players = pd.concat(nearby).sort_values("frame_number")
        ball_pixel = np.array([ball.ball_center_x, ball.ball_center_y], dtype=float)

        candidates = []
        for player in players.itertuples(index=False):
            bbox_height = max(1.0, float(player.bbox_y2) - float(player.bbox_y1))
            hands = []
            for prefix in ("left_wrist", "right_wrist"):
                x, y = getattr(player, prefix + "_x", np.nan), getattr(player, prefix + "_y", np.nan)
                if pd.notna(x) and pd.notna(y):
                    hands.append(np.array([x, y], dtype=float))
            pose_hands = bool(hands)
            if not hands:
                hands = [np.array([player.bbox_center_x, player.bbox_y1 + 0.18 * bbox_height])]
            hand_distance = min(float(np.linalg.norm(ball_pixel - hand)) for hand in hands)
            threshold = bbox_height * (0.38 if pose_hands else 0.25)
            if hand_distance <= threshold:
                candidates.append((hand_distance / threshold, hand_distance, pose_hands, player))
        if not candidates:
            continue

        normalised_distance, hand_distance, pose_hands, player = min(candidates, key=lambda value: value[0])
        # A visible hand-ball overlap is strong evidence; a trajectory change adds
        # support and rejects balls merely passing near a player.
        if trajectory_score < 0.12 and normalised_distance > 0.45:
            continue

        height = np.nan
        residual = np.nan
        height_note = "height unavailable: select both net-tape endpoints"
        if projection_matrix_3d is not None and pd.notna(player.court_x_m) and pd.notna(player.court_y_m):
            fitted, residual = estimate_height_above_ground(
                projection_matrix_3d,
                (float(player.court_x_m), float(player.court_y_m)),
                (float(ball.ball_center_x), float(ball.ball_center_y)),
            )
            tolerance = max(10.0, (float(player.bbox_x2) - float(player.bbox_x1)) * 0.35)
            if fitted is not None and residual <= tolerance:
                height = round(float(fitted), 2)
                height_note = f"vertical fit residual {residual:.1f} px"
            else:
                height_note = f"height rejected: vertical fit residual {residual:.1f} px"

        high_contact = pd.notna(height) and height >= 2.25
        front_row = pd.notna(player.court_y_m) and float(player.court_y_m) >= CENTER_LINE_Y_M - 3.2
        outgoing_rises = outgoing[1] < -max(2.0, outgoing_norm * 0.2)
        if high_contact and front_row and not outgoing_rises:
            action = "attack contact candidate"
        elif outgoing_rises and (high_contact or pose_hands):
            action = "set contact candidate"
        elif high_contact and front_row:
            action = "block contact candidate"
        else:
            action = "pass/dig contact candidate"

        confidence = float(np.clip(
            0.45 * (1 - normalised_distance)
            + 0.30 * trajectory_score
            + 0.15 * float(ball.confidence)
            + 0.10 * pose_hands,
            0, 1,
        ))
        events.append({
            "frame_number": frame_number,
            "timestamp_seconds": float(ball.timestamp_seconds),
            "player_id": str(getattr(player, id_column)),
            "raw_track_id": int(player.raw_track_id),
            "action_guess": action,
            "court_x_m": float(player.court_x_m),
            "court_y_m": float(player.court_y_m),
            "ball_court_x_m": float(ball.court_x_m) if pd.notna(ball.court_x_m) else np.nan,
            "ball_court_y_m": float(ball.court_y_m) if pd.notna(ball.court_y_m) else np.nan,
            "contact_height_estimate_m": height,
            "height_reprojection_error_px": round(float(residual), 2) if pd.notna(residual) else np.nan,
            "hand_distance_px": round(hand_distance, 2),
            "trajectory_change_score": round(trajectory_score, 3),
            "event_confidence": round(confidence, 3),
            "confidence_note": (
                ("pose wrists; " if pose_hands else "bbox hand fallback; ")
                + height_note + "; requires video review"
            ),
        })
        last_contact_frame = frame_number

    return pd.DataFrame(events, columns=columns)


def estimate_team_box_score(player_table: pd.DataFrame, contacts_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in player_table.itertuples(index=False):
        row_id = str(getattr(row, "display_id", getattr(row, "roster_id", getattr(row, "track_id", ""))))
        player_contacts = contacts_df[contacts_df["player_id"].astype(str) == row_id] if not contacts_df.empty and "player_id" in contacts_df else pd.DataFrame()
        attacks = int(player_contacts["action_guess"].str.contains("attack", na=False).sum()) if not player_contacts.empty else 0
        sets = int(player_contacts["action_guess"].str.contains("set", na=False).sum()) if not player_contacts.empty else 0
        receptions = int(player_contacts["action_guess"].str.contains("dig|reception|pass", na=False, regex=True).sum()) if not player_contacts.empty else 0
        max_height = player_contacts["contact_height_estimate_m"].max() if not player_contacts.empty else np.nan
        rows.append(
            {
                "player_id": row_id,
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
