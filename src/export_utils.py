from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import pandas as pd

from .analysis import cumulative_distance_by_frame, dataframe_for_csv
from .calibration import draw_court_lines
from .config import OUTPUT_DIR
from .visualisation import draw_ball_overlay, draw_player_boxes, draw_selected_trail, draw_stats_overlay


def ensure_output_folders() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_tracking_csv(df: pd.DataFrame, output_path: Path | None = None) -> Path:
    ensure_output_folders()
    path = output_path or (OUTPUT_DIR / "player_tracking.csv")
    dataframe_for_csv(df).to_csv(path, index=False)
    return path


def _open_writer(output_path: Path, fps: float, frame_size: tuple[int, int]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if writer.isOpened():
        return writer, output_path

    writer.release()
    fallback_path = output_path.with_suffix(".avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(fallback_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("Could not create output video writer with mp4v or MJPG codecs.")
    return writer, fallback_path


def write_annotated_video(
    input_video_path: str | Path,
    tracking_df: pd.DataFrame,
    selected_track_id: int,
    court_to_pixel_matrix,
    fps: float,
    jump_events: list[dict],
    output_path: Path | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    ball_df: pd.DataFrame | None = None,
    team_track_ids: list[int] | None = None,
) -> Path:
    ensure_output_folders()
    output_path = output_path or (OUTPUT_DIR / "annotated_video.mp4")

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Input video has invalid dimensions and cannot be annotated.")
    writer, final_path = _open_writer(output_path, fps, (width, height))

    if team_track_ids:
        team_ids = {int(track_id) for track_id in team_track_ids}
        tracking_df = tracking_df.copy()
        tracking_df["team_player"] = tracking_df["track_id"].astype(int).isin(team_ids)
    grouped = {int(frame): group.copy() for frame, group in tracking_df.groupby("frame_number")}
    selected_by_frame = {
        int(row.frame_number): row
        for row in tracking_df[tracking_df["track_id"] == selected_track_id].sort_values("frame_number").itertuples(index=False)
    }
    distance_by_frame = cumulative_distance_by_frame(tracking_df, selected_track_id, fps)
    jump_frames = sorted(int(event["apex_frame"]) for event in jump_events)
    ball_by_frame = {}
    if ball_df is not None and not ball_df.empty:
        ball_by_frame = {int(row.frame_number): row._asdict() for row in ball_df.sort_values("frame_number").itertuples(index=False)}

    trail_points: list[tuple[float, float]] = []
    ball_trail: list[tuple[float, float]] = []
    last_distance = 0.0
    jump_count = 0
    frame_number = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            annotated = draw_court_lines(frame, court_to_pixel_matrix)

            selected_row = selected_by_frame.get(frame_number)
            if selected_row is not None:
                trail_points.append((float(selected_row.foot_pixel_x), float(selected_row.foot_pixel_y)))

            if frame_number in distance_by_frame:
                last_distance = distance_by_frame[frame_number]
            while jump_count < len(jump_frames) and jump_frames[jump_count] <= frame_number:
                jump_count += 1

            ball_record = ball_by_frame.get(frame_number)
            if ball_record is not None:
                ball_trail.append((float(ball_record["ball_center_x"]), float(ball_record["ball_center_y"])))

            annotated = draw_selected_trail(annotated, trail_points)
            annotated = draw_ball_overlay(annotated, ball_record, ball_trail)
            annotated = draw_player_boxes(annotated, grouped.get(frame_number), selected_track_id=selected_track_id)
            annotated = draw_stats_overlay(
                annotated,
                {
                    "selected_track_id": selected_track_id,
                    "frame_number": frame_number,
                    "movement_distance_m": last_distance,
                    "jump_count": jump_count,
                },
            )

            writer.write(annotated)
            frame_number += 1
            if progress_callback and total_frames > 0:
                progress_callback(min(frame_number / total_frames, 1.0), f"Writing annotated video frame {frame_number} of {total_frames}")
    finally:
        cap.release()
        writer.release()

    return final_path
