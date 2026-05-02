from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .config import (
    CSV_COLUMNS,
    ESTIMATED_PLAYER_HEIGHT_M,
    MAX_JUMP_HEIGHT_M,
    MAX_REASONABLE_SPEED_MPS,
    MIN_JUMP_HEIGHT_M,
)

try:
    from scipy.signal import find_peaks as scipy_find_peaks
except Exception:  # pragma: no cover - scipy is listed in requirements.
    scipy_find_peaks = None


def records_to_dataframe(records: Iterable[dict], selected_track_id: int | None = None) -> pd.DataFrame:
    df = pd.DataFrame(list(records))
    if df.empty:
        return pd.DataFrame(columns=CSV_COLUMNS)

    df = df.sort_values(["frame_number", "track_id"]).reset_index(drop=True)
    df["selected_player"] = df["track_id"].eq(selected_track_id) if selected_track_id is not None else False
    df["jump_height_estimate_m"] = np.nan
    return df


def _valid_position_rows(df: pd.DataFrame, selected_track_id: int) -> pd.DataFrame:
    selected = df[df["track_id"] == selected_track_id].copy()
    selected = selected.dropna(subset=["court_x_m", "court_y_m", "timestamp_seconds"])
    selected = selected[np.isfinite(selected["court_x_m"]) & np.isfinite(selected["court_y_m"])]
    return selected.sort_values("frame_number")


def calculate_movement_metrics(df: pd.DataFrame, selected_track_id: int, fps: float) -> dict:
    selected = _valid_position_rows(df, selected_track_id)
    all_selected = df[df["track_id"] == selected_track_id]

    if selected.empty:
        return {
            "frames_tracked": int(len(all_selected)),
            "valid_position_count": 0,
            "time_tracked_seconds": 0.0,
            "total_distance_m": 0.0,
            "average_speed_mps": 0.0,
            "note": "Selected player was not mapped to court coordinates.",
        }

    total_distance = 0.0
    accepted_segments = 0
    rejected_segments = 0
    previous = None
    for row in selected.itertuples(index=False):
        current = (float(row.court_x_m), float(row.court_y_m), float(row.timestamp_seconds))
        if previous is not None:
            dx = current[0] - previous[0]
            dy = current[1] - previous[1]
            dt = max(current[2] - previous[2], 1.0 / max(fps, 1.0))
            distance = math.hypot(dx, dy)
            speed = distance / dt
            if speed <= MAX_REASONABLE_SPEED_MPS:
                total_distance += distance
                accepted_segments += 1
            else:
                rejected_segments += 1
        previous = current

    frames_tracked = int(len(all_selected))
    if frames_tracked > 0:
        time_tracked = frames_tracked / max(fps, 1.0)
    else:
        time_tracked = 0.0
    average_speed = total_distance / time_tracked if time_tracked > 0 else 0.0

    note = "Movement estimated from homography-mapped foot positions."
    if rejected_segments:
        note += f" Ignored {rejected_segments} unrealistic displacement segment(s)."

    return {
        "frames_tracked": frames_tracked,
        "valid_position_count": int(len(selected)),
        "time_tracked_seconds": round(float(time_tracked), 3),
        "total_distance_m": round(float(total_distance), 3),
        "average_speed_mps": round(float(average_speed), 3),
        "accepted_segments": accepted_segments,
        "rejected_segments": rejected_segments,
        "note": note,
    }


def _find_peak_indices(signal: np.ndarray, prominence: float, distance: int) -> np.ndarray:
    if scipy_find_peaks is not None:
        peaks, _ = scipy_find_peaks(signal, prominence=prominence, distance=distance)
        return peaks

    peaks = []
    for index in range(1, len(signal) - 1):
        if signal[index] > signal[index - 1] and signal[index] > signal[index + 1]:
            left = max(0, index - distance)
            right = min(len(signal), index + distance + 1)
            if signal[index] - np.nanmin(signal[left:right]) >= prominence:
                peaks.append(index)
    return np.asarray(peaks, dtype=int)


def detect_jumps(df: pd.DataFrame, selected_track_id: int, fps: float) -> tuple[list[dict], str]:
    selected = df[df["track_id"] == selected_track_id].dropna(subset=["vertical_signal_y"]).copy()
    selected = selected.sort_values("frame_number")
    if len(selected) < max(8, int(fps * 0.4)):
        return [], "jump not confidently detected: insufficient selected-player vertical signal"

    frame_min = int(selected["frame_number"].min())
    frame_max = int(selected["frame_number"].max())
    frame_index = pd.RangeIndex(frame_min, frame_max + 1)
    signal = pd.Series(selected["vertical_signal_y"].to_numpy(dtype=float), index=selected["frame_number"].astype(int))
    signal = signal.groupby(level=0).mean().reindex(frame_index)
    signal = signal.interpolate(limit=max(2, int(fps * 0.25)), limit_direction="both")
    if signal.notna().sum() < max(8, int(fps * 0.4)):
        return [], "jump not confidently detected: vertical signal has too many gaps"

    smooth_window = max(3, int(round(fps * 0.12)))
    if smooth_window % 2 == 0:
        smooth_window += 1
    smoothed = signal.rolling(window=smooth_window, min_periods=1, center=True).median()
    peak_signal = -smoothed.to_numpy(dtype=float)

    bbox_heights = (selected["bbox_y2"] - selected["bbox_y1"]).replace([np.inf, -np.inf], np.nan).dropna()
    median_bbox_height = float(bbox_heights.median()) if not bbox_heights.empty else 180.0
    if median_bbox_height <= 0:
        median_bbox_height = 180.0

    prominence = max(6.0, median_bbox_height * 0.035)
    min_peak_distance = max(3, int(fps * 0.35))
    peak_indices = _find_peak_indices(peak_signal, prominence=prominence, distance=min_peak_distance)

    source_counts = selected["vertical_signal_source"].value_counts().to_dict() if "vertical_signal_source" in selected else {}
    signal_source = "pose_com" if source_counts.get("pose_com", 0) >= source_counts.get("bbox_center", 0) else "bbox_center"
    source_note = "pose model" if signal_source == "pose_com" else "bounding-box fallback"

    events: list[dict] = []
    search_radius = max(3, int(fps * 0.45))
    for peak_index in peak_indices:
        apex_frame = int(frame_index[peak_index])
        apex_y = float(smoothed.iloc[peak_index])
        start = max(0, peak_index - search_radius)
        end = min(len(smoothed) - 1, peak_index + search_radius)
        takeoff_y = float(smoothed.iloc[start : peak_index + 1].max())
        landing_y = float(smoothed.iloc[peak_index : end + 1].max())
        baseline_y = max(takeoff_y, landing_y)
        pixel_jump = baseline_y - apex_y
        estimated_height_m = (pixel_jump / median_bbox_height) * ESTIMATED_PLAYER_HEIGHT_M

        if MIN_JUMP_HEIGHT_M <= estimated_height_m <= MAX_JUMP_HEIGHT_M:
            events.append(
                {
                    "apex_frame": apex_frame,
                    "time_seconds": round(apex_frame / max(fps, 1.0), 3),
                    "jump_height_estimate_m": round(float(estimated_height_m), 3),
                    "signal_source": signal_source,
                    "notes": f"Approximate jump estimate from {source_note}.",
                }
            )

    if not events:
        return [], f"jump not confidently detected using {source_note}"

    return events, f"Approximate jump estimate from {source_note}."


def add_jump_annotations(df: pd.DataFrame, selected_track_id: int, jump_events: list[dict], jump_note: str) -> pd.DataFrame:
    output = df.copy()
    if output.empty:
        return output

    if "selected_player" not in output:
        output["selected_player"] = output["track_id"].eq(selected_track_id)
    if "jump_height_estimate_m" not in output:
        output["jump_height_estimate_m"] = np.nan

    for event in jump_events:
        mask = (output["track_id"] == selected_track_id) & (output["frame_number"] == int(event["apex_frame"]))
        output.loc[mask, "jump_height_estimate_m"] = event["jump_height_estimate_m"]
        output.loc[mask, "notes"] = output.loc[mask, "notes"].fillna("").astype(str).str.strip()
        output.loc[mask, "notes"] = output.loc[mask, "notes"].apply(lambda value: f"{value}; jump apex".strip("; "))

    selected_mask = output["track_id"] == selected_track_id
    if jump_note:
        output.loc[selected_mask, "notes"] = output.loc[selected_mask, "notes"].fillna("").astype(str).apply(
            lambda value: f"{value}; {jump_note}".strip("; ")
        )

    return output


def summarise_jumps(jump_events: list[dict], jump_note: str) -> dict:
    if not jump_events:
        return {
            "estimated_jumps": 0,
            "max_jump_height_m": None,
            "average_jump_height_m": None,
            "note": jump_note,
        }

    heights = [float(event["jump_height_estimate_m"]) for event in jump_events]
    return {
        "estimated_jumps": len(jump_events),
        "max_jump_height_m": round(max(heights), 3),
        "average_jump_height_m": round(float(np.mean(heights)), 3),
        "note": jump_note,
    }


def cumulative_distance_by_frame(df: pd.DataFrame, selected_track_id: int, fps: float) -> dict[int, float]:
    selected = _valid_position_rows(df, selected_track_id)
    cumulative: dict[int, float] = {}
    distance_total = 0.0
    previous = None

    for row in selected.itertuples(index=False):
        current = (float(row.court_x_m), float(row.court_y_m), float(row.timestamp_seconds))
        if previous is not None:
            segment_distance = math.hypot(current[0] - previous[0], current[1] - previous[1])
            dt = max(current[2] - previous[2], 1.0 / max(fps, 1.0))
            if segment_distance / dt <= MAX_REASONABLE_SPEED_MPS:
                distance_total += segment_distance
        cumulative[int(row.frame_number)] = round(float(distance_total), 3)
        previous = current

    return cumulative


def dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for column in CSV_COLUMNS:
        if column not in output:
            output[column] = np.nan
    return output[CSV_COLUMNS]

