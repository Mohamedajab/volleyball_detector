from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

from .config import ATTACK_LINE_Y_M, CENTER_LINE_Y_M, COURT_LENGTH_M, COURT_WIDTH_M

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _records_iter(frame_records) -> list[dict]:
    if frame_records is None:
        return []
    if isinstance(frame_records, pd.DataFrame):
        return frame_records.to_dict("records")
    return list(frame_records)


def draw_player_boxes(frame: np.ndarray, frame_records, selected_track_id: int | None = None) -> np.ndarray:
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    for record in _records_iter(frame_records):
        x1 = max(0, min(width - 1, int(round(record["bbox_x1"]))))
        y1 = max(0, min(height - 1, int(round(record["bbox_y1"]))))
        x2 = max(0, min(width - 1, int(round(record["bbox_x2"]))))
        y2 = max(0, min(height - 1, int(round(record["bbox_y2"]))))
        if x2 <= x1 or y2 <= y1:
            continue
        track_id = int(record["track_id"])
        is_selected = selected_track_id is not None and track_id == int(selected_track_id)
        is_team = bool(record.get("team_player", False))
        color = (0, 140, 255) if is_selected else ((255, 210, 70) if is_team else (40, 220, 90))
        thickness = 3 if is_selected or is_team else 2
        roster_id = str(record.get("roster_id", "")).strip()
        label = roster_id if is_team and roster_id else (f"Team ID {track_id}" if is_team else f"ID {track_id}")
        if is_selected:
            label = f"Selected Player ID: {track_id}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        label_y = max(22, y1 - 8)
        cv2.putText(annotated, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
    return annotated


def draw_selected_trail(frame: np.ndarray, trail_points: list[tuple[float, float]], max_points: int = 120) -> np.ndarray:
    annotated = frame.copy()
    recent_points = trail_points[-max_points:]
    if len(recent_points) < 2:
        return annotated

    points = [(int(round(x)), int(round(y))) for x, y in recent_points if np.isfinite(x) and np.isfinite(y)]
    for index in range(1, len(points)):
        alpha = index / max(len(points), 1)
        color = (0, int(120 + 100 * alpha), 255)
        cv2.line(annotated, points[index - 1], points[index], color, 3, cv2.LINE_AA)
    cv2.circle(annotated, points[-1], 5, (0, 220, 255), -1, cv2.LINE_AA)
    return annotated


def draw_stats_overlay(frame: np.ndarray, stats: dict) -> np.ndarray:
    annotated = frame.copy()
    overlay = annotated.copy()
    height, width = annotated.shape[:2]
    box_width = min(470, width - 24)
    box_height = 132
    cv2.rectangle(overlay, (12, 12), (12 + box_width, 12 + box_height), (15, 20, 28), -1)
    annotated = cv2.addWeighted(overlay, 0.72, annotated, 0.28, 0)

    lines = [
        f"Selected Player ID: {stats.get('selected_track_id', 'N/A')}",
        f"Frame: {stats.get('frame_number', 0)}",
        f"Movement distance: {stats.get('movement_distance_m', 0.0):.2f} m",
        f"Estimated jumps: {stats.get('jump_count', 0)}",
    ]
    for index, text in enumerate(lines):
        cv2.putText(
            annotated,
            text,
            (28, 42 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (245, 248, 250),
            2,
            cv2.LINE_AA,
        )
    return annotated


def draw_contact_overlay(frame: np.ndarray, contact: dict | None) -> np.ndarray:
    if not contact:
        return frame
    annotated = frame.copy()
    height = contact.get("contact_height_estimate_m")
    height_text = f" | {float(height):.2f} m" if pd.notna(height) else ""
    confidence = float(contact.get("event_confidence", 0))
    action = str(contact.get("action_guess", "contact")).replace(" contact candidate", "").upper()
    text = f"{contact.get('player_id', '')} {action}{height_text} | confidence {confidence:.0%}"
    cv2.rectangle(annotated, (12, 154), (min(720, annotated.shape[1] - 12), 202), (15, 20, 28), -1)
    cv2.putText(annotated, text, (26, 187), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (70, 220, 255), 2, cv2.LINE_AA)
    return annotated


def generate_top_down_court(df: pd.DataFrame, selected_track_id: int, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = df[df["track_id"] == selected_track_id].dropna(subset=["court_x_m", "court_y_m"]).sort_values("frame_number")

    fig, ax = plt.subplots(figsize=(5.4, 9.2))
    ax.set_facecolor("#17443a")
    fig.patch.set_facecolor("#ffffff")

    court = plt.Rectangle((0, 0), COURT_WIDTH_M, COURT_LENGTH_M, facecolor="#d8a85f", edgecolor="white", linewidth=2.2)
    ax.add_patch(court)
    ax.plot([0, COURT_WIDTH_M], [CENTER_LINE_Y_M, CENTER_LINE_Y_M], color="white", linewidth=2)
    for y_value in ATTACK_LINE_Y_M:
        ax.plot([0, COURT_WIDTH_M], [y_value, y_value], color="white", linewidth=1.6, linestyle="--")

    if not selected.empty:
        x = selected["court_x_m"].to_numpy(dtype=float)
        y = selected["court_y_m"].to_numpy(dtype=float)
        frames = selected["frame_number"].to_numpy(dtype=float)
        ax.plot(x, y, color="#0b1f33", linewidth=1.6, alpha=0.75)
        scatter = ax.scatter(x, y, c=frames, cmap="viridis", s=22, edgecolor="white", linewidth=0.25)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Frame")
    else:
        ax.text(
            COURT_WIDTH_M / 2,
            COURT_LENGTH_M / 2,
            "No selected-player court positions",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
        )

    ax.set_xlim(-0.7, COURT_WIDTH_M + 0.7)
    ax.set_ylim(-0.7, COURT_LENGTH_M + 0.7)
    ax.set_aspect("equal")
    ax.set_xlabel("Court width (m)")
    ax.set_ylabel("Court length (m)")
    ax.set_title(f"Selected Player {selected_track_id} Movement Path")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path



def draw_ball_overlay(frame: np.ndarray, ball_record: dict | None, ball_trail: list[tuple[float, float]]) -> np.ndarray:
    annotated = frame.copy()
    points = [(int(round(x)), int(round(y))) for x, y in ball_trail[-80:] if np.isfinite(x) and np.isfinite(y)]
    for index in range(1, len(points)):
        cv2.line(annotated, points[index - 1], points[index], (255, 80, 40), 2, cv2.LINE_AA)
    if ball_record is not None:
        x = int(round(ball_record["ball_center_x"]))
        y = int(round(ball_record["ball_center_y"]))
        cv2.circle(annotated, (x, y), 7, (255, 80, 40), -1, cv2.LINE_AA)
        cv2.putText(annotated, "Ball", (x + 9, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 40), 2, cv2.LINE_AA)
    return annotated


def generate_team_ball_court_map(player_df: pd.DataFrame, ball_df: pd.DataFrame, team_track_ids: list[int], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    team_ids = {str(track_id) for track_id in team_track_ids}
    if not player_df.empty and "roster_id" in player_df and player_df["roster_id"].astype(str).str.len().any():
        team = player_df[player_df["roster_id"].astype(str).isin(team_ids)].dropna(subset=["court_x_m", "court_y_m"])
    else:
        team = player_df[player_df["track_id"].astype(str).isin(team_ids)].dropna(subset=["court_x_m", "court_y_m"]) if not player_df.empty else pd.DataFrame()
    ball = ball_df.dropna(subset=["court_x_m", "court_y_m"]) if ball_df is not None and not ball_df.empty else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(5.8, 9.4))
    ax.set_facecolor("#15463d")
    fig.patch.set_facecolor("#ffffff")
    court = plt.Rectangle((0, 0), COURT_WIDTH_M, COURT_LENGTH_M, facecolor="#d8a85f", edgecolor="white", linewidth=2.2)
    ax.add_patch(court)
    ax.plot([0, COURT_WIDTH_M], [CENTER_LINE_Y_M, CENTER_LINE_Y_M], color="white", linewidth=2.2)
    for y_value in ATTACK_LINE_Y_M:
        ax.plot([0, COURT_WIDTH_M], [y_value, y_value], color="white", linewidth=1.5, linestyle="--")

    if not team.empty:
        group_column = "roster_id" if "roster_id" in team and team["roster_id"].astype(str).str.len().any() else "track_id"
        for track_id, group in team.groupby(group_column):
            ax.plot(group["court_x_m"], group["court_y_m"], linewidth=1.2, alpha=0.65, label=str(track_id))
            ax.scatter(group["court_x_m"].iloc[-1], group["court_y_m"].iloc[-1], s=42, edgecolor="white")

    if not ball.empty:
        projected = ball[["court_x_m", "court_y_m"]].copy()
        projected.loc[ball.timestamp_seconds.diff() > 0.25] = np.nan
        ax.plot(projected["court_x_m"], projected["court_y_m"], color="#1f5eff", linewidth=1.6, alpha=0.8, label="Ball image projection (not 3D)")
        ax.scatter(ball["court_x_m"], ball["court_y_m"], c=ball["frame_number"], cmap="cool", s=20, edgecolor="white", linewidth=0.2)

    ax.text(COURT_WIDTH_M / 2, 0.35, "Camera-side team", color="white", ha="center", va="bottom", fontsize=10)
    ax.set_xlim(-0.7, COURT_WIDTH_M + 0.7)
    ax.set_ylim(-0.7, COURT_LENGTH_M + 0.7)
    ax.set_aspect("equal")
    ax.set_xlabel("Court width (m)")
    ax.set_ylabel("Court length (m)")
    ax.set_title("Team movement + ball image projection")
    if len(ax.get_legend_handles_labels()[0]) <= 10:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
