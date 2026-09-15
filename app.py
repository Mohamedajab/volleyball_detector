from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.ball_tracking import detect_ball_candidates_classical, BallTracker
from src.roster import team_records, roster_ids_from_dataframe
from src.stat_sheet import EVENTS, build_stat_sheet
from src.pose_enrichment import enrich_hands_near_ball
from src.analysis import (
    add_jump_annotations,
    calculate_movement_metrics,
    detect_jumps,
    records_to_dataframe,
    summarise_jumps,
)
from src.calibration import compute_homography, draw_calibration_points, draw_court_lines, validate_corner_points
from src.camera_3d import build_projection_matrix, draw_net_calibration
from src.config import (
    APP_DESCRIPTION,
    APP_TITLE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MAX_MISSING_FRAMES,
    DEFAULT_MAX_TRACK_DISTANCE_PX,
    DEFAULT_PREVIEW_SECONDS,
    DEFAULT_TRACKER_BACKEND,
    OUTPUT_DIR,
    SUPPORTED_VIDEO_TYPES,
    TRACKER_BACKENDS,
)
from src import detection as detection_module
from src.detection import load_detection_model, load_pose_model
from src.export_utils import write_annotated_video, write_tracking_csv
from src.tracking import run_tracking_on_video
from src.video_utils import (
    bgr_to_rgb,
    ensure_workspace_directories,
    extract_frame,
    is_supported_video,
    read_video_metadata,
    resize_for_display,
    save_uploaded_video,
)
from src import visualisation as visualisation_module
from src.visualisation import draw_player_boxes, generate_top_down_court
from src.volleyball_metrics import (
    ball_records_to_dataframe,
    estimate_ball_record,
    estimate_contacts,
    estimate_player_table,
    estimate_team_box_score,
    select_near_side_team,
    tag_team_players,
    write_volleyball_outputs,
)

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except Exception:  # pragma: no cover - dependency is in requirements, manual fallback remains available.
    streamlit_image_coordinates = None


st.set_page_config(page_title=APP_TITLE, layout="wide")


STATE_DEFAULTS = {
    "video_path": None,
    "uploaded_signature": None,
    "metadata": None,
    "calibration_frame_number": 0,
    "corner_points": [],
    "last_click_signature": None,
    "pixel_to_court_matrix": None,
    "court_to_pixel_matrix": None,
    "net_top_points": [],
    "projection_matrix_3d": None,
    "net_height_m": 2.43,
    "preview_records": None,
    "preview_track_ids": [],
    "preview_frame_number": None,
    "selected_track_id": None,
    "results": None,
    "roster_seeds": None,
}


def cached_detection_model():
    return load_detection_model(st.session_state.get("model_choice", "yolo26s.pt"))


@st.cache_resource(show_spinner=False)
def cached_pose_model():
    return load_pose_model()




@st.cache_resource(show_spinner=False)
def cached_ball_model():
    loader = getattr(detection_module, "load_ball_model", None)
    if loader is None:
        return None, None
    return loader()


@st.cache_data(show_spinner=False)
def cached_extract_frame(video_path: str, frame_number: int):
    return extract_frame(video_path, frame_number)


def initialise_state() -> None:
    for key, value in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, list) else value


def reset_for_new_video(video_path: Path, signature: str) -> None:
    st.session_state.roster_seeds = None
    st.session_state.video_path = str(video_path)
    st.session_state.uploaded_signature = signature
    st.session_state.metadata = read_video_metadata(video_path)
    st.session_state.calibration_frame_number = 0
    st.session_state.corner_points = []
    st.session_state.last_click_signature = None
    st.session_state.pixel_to_court_matrix = None
    st.session_state.court_to_pixel_matrix = None
    st.session_state.net_top_points = []
    st.session_state.projection_matrix_3d = None
    st.session_state.preview_records = None
    st.session_state.preview_track_ids = []
    st.session_state.preview_frame_number = None
    st.session_state.selected_track_id = None
    st.session_state.results = None
    cached_extract_frame.clear()




def run_ball_tracking_pass(video_path: str, ball_model, pixel_to_court_matrix, fps: float, frame_limit: int | None, progress_callback=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames_to_process = min(total_frames, frame_limit) if frame_limit else total_frames
    records = []
    frame_number = 0
    previous_frame = None
    ball_tracker = BallTracker()
    try:
        while True:
            if frame_limit is not None and frame_number >= frame_limit:
                break
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_number / fps if fps > 0 else 0.0
            detector = getattr(detection_module, "detect_balls", None)
            yolo_balls = detector(ball_model, frame, confidence_threshold=0.12) if detector is not None and ball_model is not None else []
            classical_balls = detect_ball_candidates_classical(frame, previous_frame)
            ball = ball_tracker.update(yolo_balls, classical_balls, timestamp, frame.shape)
            if ball is not None:
                records.append(estimate_ball_record(ball, frame_number, timestamp, pixel_to_court_matrix))
            previous_frame = frame.copy()
            frame_number += 1
            if progress_callback and frames_to_process:
                progress_callback(min(frame_number / frames_to_process, 1.0), f"Tracking ball frame {frame_number} of {frames_to_process}")
    finally:
        cap.release()
    return records


def render_sidebar() -> dict:
    st.sidebar.header("Workflow")
    steps = [
        ("Step 1", "Upload video", st.session_state.video_path is not None),
        ("Step 2", "Select calibration frame", st.session_state.video_path is not None),
        ("Step 3", "Mark court corners", st.session_state.pixel_to_court_matrix is not None),
        ("Step 4", "Run detection/tracking", bool(st.session_state.preview_track_ids)),
        ("Step 5", "Confirm analysis track", st.session_state.selected_track_id is not None),
        ("Step 6", "Generate outputs", st.session_state.results is not None),
        ("Step 7", "Download results", st.session_state.results is not None),
    ]
    for step, label, complete in steps:
        st.sidebar.write(f"{'[x]' if complete else '[ ]'} {step}: {label}")

    st.sidebar.divider()
    st.sidebar.header("Settings")
    st.sidebar.selectbox("Player detector", ["yolo26s.pt", "yolo26m.pt", "yolo11s.pt", "custom/local"], key="model_choice")
    confidence = st.sidebar.slider("Detection confidence", 0.10, 0.90, DEFAULT_CONFIDENCE_THRESHOLD, 0.05)
    tracker_backend = st.sidebar.selectbox(
        "Tracking backend",
        options=list(TRACKER_BACKENDS),
        index=list(TRACKER_BACKENDS).index(DEFAULT_TRACKER_BACKEND),
        format_func=lambda value: "BoT-SORT + appearance matching" if value == DEFAULT_TRACKER_BACKEND else value.replace(".yaml", ""),
        help="Appearance matching helps retain identity during overlaps. Compare backends on your footage.",
    )
    max_distance = st.sidebar.slider("Fallback tracker match distance (px)", 30.0, 220.0, DEFAULT_MAX_TRACK_DISTANCE_PX, 5.0)
    max_missing = st.sidebar.slider("Fallback keep-lost frames", 1, 60, DEFAULT_MAX_MISSING_FRAMES, 1)
    preview_seconds = st.sidebar.slider("Preview duration (seconds)", 1.0, 15.0, DEFAULT_PREVIEW_SECONDS, 1.0)
    full_frame_limit = st.sidebar.number_input(
        "Full processing frame limit (0 = full video)",
        min_value=0,
        max_value=100000,
        value=0,
        step=50,
        help="Useful for quick testing on long videos.",
    )
    use_pose = st.sidebar.checkbox("Use hand pose around ball contacts", value=True)

    return {
        "confidence": float(confidence),
        "tracker_backend": str(tracker_backend),
        "max_distance": float(max_distance),
        "max_missing": int(max_missing),
        "preview_seconds": float(preview_seconds),
        "full_frame_limit": int(full_frame_limit) or None,
        "use_pose": bool(use_pose),
    }


def render_upload_step() -> None:
    st.subheader("Step 1: Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a volleyball video",
        type=list(SUPPORTED_VIDEO_TYPES),
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Upload an MP4, MOV, AVI, or MKV file to begin.")
        return

    if not is_supported_video(uploaded_file.name):
        st.error("Unsupported video type. Please upload MP4, MOV, AVI, or MKV.")
        return

    signature = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.uploaded_signature != signature:
        try:
            video_path = save_uploaded_video(uploaded_file)
            reset_for_new_video(video_path, signature)
            st.success(f"Video saved to {video_path.name}")
        except Exception as exc:
            st.error(f"Could not save uploaded video: {exc}")


def render_video_summary() -> None:
    metadata = st.session_state.metadata
    if not metadata:
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Frames", metadata["frame_count"])
    col2.metric("FPS", f"{metadata['fps']:.2f}")
    col3.metric("Duration", f"{metadata['duration_seconds']:.1f}s")
    col4.metric("Resolution", f"{metadata['width']}x{metadata['height']}")
    st.video(st.session_state.video_path)


def _default_corner_points(frame: np.ndarray) -> list[tuple[int, int]]:
    height, width = frame.shape[:2]
    return [
        (int(width * 0.18), int(height * 0.86)),
        (int(width * 0.82), int(height * 0.86)),
        (int(width * 0.72), int(height * 0.34)),
        (int(width * 0.28), int(height * 0.34)),
    ]


def render_clickable_calibration_frame(frame: np.ndarray) -> None:
    annotated = draw_calibration_points(frame, st.session_state.corner_points)
    display_bgr, scale = resize_for_display(annotated, max_width=900)
    display_rgb = bgr_to_rgb(display_bgr)

    if streamlit_image_coordinates is None:
        st.image(display_rgb, caption="Calibration frame")
        st.warning("Click selection is unavailable because streamlit-image-coordinates is not installed. Use the manual coordinate fallback below.")
        return

    clicked = streamlit_image_coordinates(
        Image.fromarray(display_rgb),
        key=f"court_click_{st.session_state.uploaded_signature}_{st.session_state.calibration_frame_number}_{len(st.session_state.corner_points)}",
    )
    if clicked and len(st.session_state.corner_points) < 4:
        click_signature = (
            int(clicked["x"]),
            int(clicked["y"]),
            st.session_state.calibration_frame_number,
            len(st.session_state.corner_points),
        )
        if st.session_state.last_click_signature != click_signature:
            st.session_state.last_click_signature = click_signature
            original_x = int(round(clicked["x"] / scale))
            original_y = int(round(clicked["y"] / scale))
            st.session_state.corner_points.append((original_x, original_y))
            st.rerun()


def render_manual_corner_fallback(frame: np.ndarray) -> None:
    labels = ["near left", "near right", "far right", "far left"]
    defaults = _default_corner_points(frame)
    height, width = frame.shape[:2]

    with st.expander("Manual coordinate fallback"):
        st.caption("Enter pixel coordinates in the same order: near left, near right, far right, far left.")
        with st.form("manual_court_points"):
            typed_points: list[tuple[int, int]] = []
            for index, label in enumerate(labels):
                existing = st.session_state.corner_points[index] if index < len(st.session_state.corner_points) else defaults[index]
                col_x, col_y = st.columns(2)
                x_value = col_x.number_input(
                    f"{index + 1}. {label} x",
                    min_value=0,
                    max_value=max(width - 1, 0),
                    value=int(existing[0]),
                    step=1,
                    key=f"manual_x_{index}",
                )
                y_value = col_y.number_input(
                    f"{index + 1}. {label} y",
                    min_value=0,
                    max_value=max(height - 1, 0),
                    value=int(existing[1]),
                    step=1,
                    key=f"manual_y_{index}",
                )
                typed_points.append((int(x_value), int(y_value)))

            submitted = st.form_submit_button("Use typed points")
            if submitted:
                st.session_state.corner_points = typed_points
                st.session_state.last_click_signature = None
                st.rerun()


def render_vertical_calibration(frame: np.ndarray) -> None:
    st.markdown("#### Height calibration")
    st.caption("For contact height, click the top of the net tape at the left sideline, then at the right sideline.")
    net_height = st.radio(
        "Net height",
        options=[2.43, 2.24],
        format_func=lambda value: "Men / coed: 2.43 m" if value == 2.43 else "Women: 2.24 m",
        horizontal=True,
    )
    if float(net_height) != float(st.session_state.net_height_m):
        st.session_state.projection_matrix_3d = None
    st.session_state.net_height_m = float(net_height)
    preview = draw_court_lines(frame, st.session_state.court_to_pixel_matrix)
    preview = draw_net_calibration(preview, st.session_state.net_top_points)
    display_bgr, scale = resize_for_display(preview, max_width=900)
    display_rgb = bgr_to_rgb(display_bgr)
    if streamlit_image_coordinates is not None:
        clicked = streamlit_image_coordinates(
            Image.fromarray(display_rgb),
            key=f"net_click_{st.session_state.uploaded_signature}_{len(st.session_state.net_top_points)}",
        )
        if clicked and len(st.session_state.net_top_points) < 2:
            st.session_state.net_top_points.append(
                (int(round(clicked["x"] / scale)), int(round(clicked["y"] / scale)))
            )
            st.session_state.projection_matrix_3d = None
            st.rerun()
    else:
        st.image(display_rgb)

    height, width = frame.shape[:2]
    with st.expander("Enter net points manually"):
        with st.form("manual_net_points"):
            defaults = [(int(width * .35), int(height * .47)), (int(width * .65), int(height * .47))]
            values = []
            for index in range(2):
                current = st.session_state.net_top_points[index] if index < len(st.session_state.net_top_points) else defaults[index]
                col_x, col_y = st.columns(2)
                x = col_x.number_input(f"Net top {index + 1} x", 0, width - 1, int(current[0]))
                y = col_y.number_input(f"Net top {index + 1} y", 0, height - 1, int(current[1]))
                values.append((x, y))
            if st.form_submit_button("Use these net points"):
                st.session_state.net_top_points = values
                st.session_state.projection_matrix_3d = None
                st.rerun()

    reset_col, confirm_col = st.columns(2)
    if reset_col.button("Reset net points"):
        st.session_state.net_top_points = []
        st.session_state.projection_matrix_3d = None
        st.rerun()
    if confirm_col.button("Confirm height calibration", disabled=len(st.session_state.net_top_points) != 2):
        try:
            projection, error = build_projection_matrix(
                st.session_state.court_to_pixel_matrix,
                st.session_state.net_top_points,
                st.session_state.net_height_m,
            )
            if error > 4:
                st.error(f"Net calibration does not fit ({error:.1f} px error). Re-select both tape endpoints.")
            else:
                st.session_state.projection_matrix_3d = projection
                st.success(f"Height calibration ready ({error:.1f} px fit error).")
        except Exception as exc:
            st.error(f"Height calibration failed: {exc}")
    if st.session_state.projection_matrix_3d is not None:
        st.success("Metric height calibration is active.")


def render_calibration_step() -> None:
    st.subheader("Step 2 and 3: Select Frame and Mark Court Corners")
    metadata = st.session_state.metadata
    if not metadata:
        st.warning("Upload a video before calibration.")
        return

    frame_count = max(metadata["frame_count"], 1)
    frame_number = st.slider(
        "Calibration frame",
        min_value=0,
        max_value=frame_count - 1,
        value=min(st.session_state.calibration_frame_number, frame_count - 1),
        step=1,
        help="Choose a clear frame where the full court boundary is visible.",
    )
    if frame_number != st.session_state.calibration_frame_number:
        st.session_state.calibration_frame_number = frame_number
        st.session_state.corner_points = []
        st.session_state.pixel_to_court_matrix = None
        st.session_state.court_to_pixel_matrix = None
        st.session_state.net_top_points = []
        st.session_state.projection_matrix_3d = None
        st.session_state.preview_records = None
        st.session_state.preview_track_ids = []
        st.session_state.selected_track_id = None
        st.session_state.results = None
        st.rerun()

    frame = cached_extract_frame(st.session_state.video_path, st.session_state.calibration_frame_number)
    if frame is None:
        st.error("Could not extract the selected calibration frame.")
        return

    st.info("Click 4 court corners in this order: near left, near right, far right, far left.")
    render_clickable_calibration_frame(frame)
    render_manual_corner_fallback(frame)

    col_a, col_b = st.columns([1, 1])
    col_a.write(f"{len(st.session_state.corner_points)} of 4 points selected")
    if col_b.button("Reset corners"):
        st.session_state.corner_points = []
        st.session_state.last_click_signature = None
        st.session_state.pixel_to_court_matrix = None
        st.session_state.court_to_pixel_matrix = None
        st.session_state.net_top_points = []
        st.session_state.projection_matrix_3d = None
        st.session_state.preview_records = None
        st.session_state.preview_track_ids = []
        st.session_state.selected_track_id = None
        st.session_state.results = None
        st.rerun()

    if len(st.session_state.corner_points) == 4:
        ok, message = validate_corner_points(st.session_state.corner_points, frame.shape)
        if not ok:
            st.error(message)
            return

        pixel_to_court, court_to_pixel = compute_homography(st.session_state.corner_points)
        preview = draw_court_lines(draw_calibration_points(frame, st.session_state.corner_points), court_to_pixel)
        st.image(bgr_to_rgb(preview), caption="Court line overlay preview", use_container_width=True)

        if st.button("Confirm court calibration", type="primary"):
            st.session_state.pixel_to_court_matrix = pixel_to_court
            st.session_state.court_to_pixel_matrix = court_to_pixel
            st.session_state.preview_records = None
            st.session_state.preview_track_ids = []
            st.session_state.selected_track_id = None
            st.session_state.results = None
            st.session_state.net_top_points = []
            st.session_state.projection_matrix_3d = None
            st.success("Court calibration confirmed.")
            st.rerun()

    if st.session_state.court_to_pixel_matrix is not None:
        render_vertical_calibration(frame)


def _load_detector_with_ui():
    try:
        model, source = cached_detection_model()
        st.caption(f"Detection model: {source}")
        return model
    except Exception as exc:
        st.error(f"Model could not be loaded: {exc}")
        return None


def render_tracking_preview(settings: dict) -> None:
    st.subheader("Step 4: Find Your Team")
    metadata = st.session_state.metadata
    if st.button("Find players", type="primary"):
        model = _load_detector_with_ui()
        if model is None:
            return
        progress = st.progress(0.0)
        try:
            records = run_tracking_on_video(
                st.session_state.video_path, model, st.session_state.pixel_to_court_matrix,
                settings["confidence"], settings["max_distance"], settings["max_missing"],
                frame_limit=max(1, int(metadata["fps"] * settings["preview_seconds"])),
                tracker_backend=settings["tracker_backend"],
                progress_callback=lambda value, message: progress.progress(value, text=message),
            )
            if not records:
                st.error("No players detected. Try a clearer clip or a different detector.")
                return
            st.session_state.preview_records = records
            st.session_state.roster_seeds = None
            st.session_state.results = None
        except Exception as exc:
            st.error(str(exc))
        finally:
            progress.empty()
    if not st.session_state.preview_records:
        return
    raw = records_to_dataframe(st.session_state.preview_records)
    eligible = raw[raw.court_x_m.between(-0.75, 9.75) & raw.court_y_m.between(-1.5, 9.0)]
    if eligible.empty:
        st.warning("No players inside the near half. Check your court corners.")
        return
    frames = sorted(eligible.frame_number.unique().tolist())
    best = int(eligible.groupby("frame_number").size().idxmax())
    frame_number = st.select_slider("Team selection frame", options=frames, value=best)
    candidates = eligible[eligible.frame_number == frame_number].sort_values("court_x_m")
    frame = cached_extract_frame(st.session_state.video_path, int(frame_number))
    if frame is not None:
        preview = draw_court_lines(frame, st.session_state.court_to_pixel_matrix)
        st.image(bgr_to_rgb(draw_player_boxes(preview, candidates)), use_container_width=True)
    choices = candidates.track_id.astype(int).tolist()
    with st.form("lock_roster"):
        seeds = st.multiselect("Your six players in this frame", choices, default=choices[:6], max_selections=6)
        confirmed = st.form_submit_button("Lock these players", type="primary")
    if confirmed:
        if len(seeds) != 6:
            st.warning("Choose all six teammates. Try another frame if someone is hidden.")
        else:
            st.session_state.roster_seeds = seeds
            st.session_state.selected_track_id = 1
            st.session_state.results = None
    if st.session_state.roster_seeds:
        team = team_records(raw, st.session_state.roster_seeds)
        st.session_state.preview_track_ids = sorted(team.track_id.unique().tolist())
        st.success("Six-player roster locked.")
        st.dataframe(pd.DataFrame({
            "Player": [f"P{i+1}" for i in range(6)],
            "Selection label": st.session_state.roster_seeds,
        }), hide_index=True)


def render_player_selection() -> None:
    st.subheader("Step 5: Select Player")
    if not st.session_state.roster_seeds:
        st.info("Lock your six players above before generating a report.")
        return
    selected = st.selectbox("Player to highlight", list(range(1, 7)), format_func=lambda value: f"P{value}")
    st.session_state.selected_track_id = selected


def render_output_generation(settings: dict) -> None:
    st.subheader("Step 6: Generate Outputs")
    if not st.session_state.roster_seeds:
        st.warning("Lock your six-player roster before generating outputs.")
        return

    if not st.button("Process full video and create outputs", type="primary"):
        return

    model = _load_detector_with_ui()
    if model is None:
        return

    pose_model = None
    pose_source = None
    if settings["use_pose"]:
        pose_model, pose_source = cached_pose_model()
        if pose_model is None:
            st.info("Pose model not found. Continuing with bounding-box jump analysis.")
        else:
            st.caption(f"Pose model: {pose_source}")

    metadata = st.session_state.metadata
    selected_id = int(st.session_state.selected_track_id)

    tracking_progress = st.progress(0, text="Starting full-video tracking...")

    def tracking_update(value: float, text: str) -> None:
        tracking_progress.progress(value, text=text)

    try:
        records = run_tracking_on_video(
            video_path=st.session_state.video_path,
            detector_model=model,
            pixel_to_court_matrix=st.session_state.pixel_to_court_matrix,
            confidence_threshold=settings["confidence"],
            max_distance_px=settings["max_distance"],
            max_missing_frames=settings["max_missing"],
            frame_limit=settings["full_frame_limit"],
            progress_callback=tracking_update,
            pose_model=None,
            selected_track_id=selected_id,
            tracker_backend=settings["tracker_backend"],
        )
    except Exception as exc:
        tracking_progress.empty()
        st.error(f"Full-video tracking failed: {exc}")
        return

    tracking_progress.empty()
    df = team_records(records_to_dataframe(records), st.session_state.roster_seeds)
    df["selected_player"] = df.track_id == selected_id
    if df.empty:
        st.error("No players were detected, so outputs could not be generated.")
        return

    if selected_id not in set(df["track_id"].astype(int).unique()):
        st.warning("The selected player ID was not found in the full pass. Outputs will still include all tracked players.")

    movement_metrics = calculate_movement_metrics(df, selected_id, metadata["fps"])
    jump_events, jump_note = detect_jumps(df, selected_id, metadata["fps"])
    df = add_jump_annotations(df, selected_id, jump_events, jump_note)
    jump_summary = summarise_jumps(jump_events, jump_note)

    roster_ids = roster_ids_from_dataframe(df)
    if len(roster_ids) < 6:
        st.warning(f"Built {len(roster_ids)} near-side roster slot(s). For best results use a centred back-view clip where all six players are visible.")
    player_table = estimate_player_table(df, roster_ids, metadata["fps"], id_column="roster_id")

    ball_model = None
    ball_source = None
    try:
        ball_model, ball_source = cached_ball_model()
    except Exception:
        ball_model = None
    if ball_model is not None:
        st.caption(f"Ball model: {ball_source}")
    else:
        st.info("Ball model unavailable. Ball results will remain empty; player movement can still be exported.")

    ball_progress = st.progress(0, text="Tracking ball...")

    def ball_update(value: float, text: str) -> None:
        ball_progress.progress(value, text=text)

    ball_records = run_ball_tracking_pass(
        st.session_state.video_path,
        ball_model,
        st.session_state.pixel_to_court_matrix,
        metadata["fps"],
        settings["full_frame_limit"],
        progress_callback=ball_update,
    )
    ball_progress.empty()
    ball_df = ball_records_to_dataframe(ball_records)
    if pose_model is not None and not ball_df.empty:
        pose_progress = st.progress(0, text="Checking hands near the tracked ball...")
        df = enrich_hands_near_ball(
            st.session_state.video_path, df, ball_df, pose_model,
        )
        pose_progress.empty()
    contacts_df = estimate_contacts(
        df, ball_df, roster_ids, metadata["fps"],
        projection_matrix_3d=st.session_state.projection_matrix_3d,
    )
    box_score_df = estimate_team_box_score(player_table, contacts_df)

    try:
        team_map_generator = getattr(visualisation_module, "generate_team_ball_court_map", None)
        if team_map_generator is None:
            top_down_path = generate_top_down_court(df, selected_id, OUTPUT_DIR / "team_ball_court_map.png")
        else:
            top_down_path = team_map_generator(df, ball_df, roster_ids, OUTPUT_DIR / "team_ball_court_map.png")
        selected_top_down_path = generate_top_down_court(df, selected_id, OUTPUT_DIR / "top_down_court.png")
        csv_path = write_tracking_csv(df, OUTPUT_DIR / "player_tracking.csv")
        volleyball_paths = write_volleyball_outputs(player_table, ball_df, contacts_df, box_score_df)
    except Exception as exc:
        st.error(f"Could not create CSV or court-map outputs: {exc}")
        return

    video_progress = st.progress(0, text="Writing annotated video...")

    def video_update(value: float, text: str) -> None:
        video_progress.progress(value, text=text)

    try:
        annotated_video_path = write_annotated_video(
            input_video_path=st.session_state.video_path,
            tracking_df=df,
            selected_track_id=selected_id,
            court_to_pixel_matrix=st.session_state.court_to_pixel_matrix,
            fps=metadata["fps"],
            jump_events=jump_events,
            output_path=OUTPUT_DIR / "annotated_video.mp4",
            progress_callback=video_update,
            ball_df=ball_df,
            team_track_ids=roster_ids,
            contacts_df=contacts_df,
        )
    except Exception as exc:
        video_progress.empty()
        st.error(f"Output video could not be generated: {exc}")
        return

    video_progress.empty()
    st.session_state.results = {
        "tracking_df": df,
        "movement_metrics": movement_metrics,
        "jump_summary": jump_summary,
        "jump_events": jump_events,
        "team_track_ids": roster_ids,
        "player_table": player_table,
        "ball_df": ball_df,
        "contacts_df": contacts_df,
        "box_score_df": box_score_df,
        "csv_path": str(csv_path),
        "top_down_path": str(top_down_path),
        "selected_top_down_path": str(selected_top_down_path),
        "annotated_video_path": str(annotated_video_path),
        "volleyball_paths": {key: str(value) for key, value in volleyball_paths.items()},
    }
    st.success("Outputs generated successfully.")


def _download_button(path: str, label: str, mime: str) -> None:
    if not path:
        st.warning("Output file was not generated.")
        return
    file_path = Path(path)
    if not file_path.exists() or file_path.is_dir():
        st.warning(f"Missing output file: {file_path}")
        return
    st.download_button(
        label=label,
        data=file_path.read_bytes(),
        file_name=file_path.name,
        mime=mime,
    )


def render_results() -> None:
    results = st.session_state.results
    if not results:
        return

    st.subheader("Step 7: Download Results")
    metrics = results["movement_metrics"]
    jumps = results["jump_summary"]
    team_ids = results.get("team_track_ids", [])
    ball_df = results.get("ball_df")
    contacts_df = results.get("contacts_df")
    box_score_df = results.get("box_score_df")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Roster slots", len(team_ids))
    col2.metric("Ball detections", 0 if ball_df is None else len(ball_df))
    col3.metric("Contact guesses", 0 if contacts_df is None else len(contacts_df))
    col4.metric("Selected distance", f"{metrics.get('total_distance_m', 0):.2f} m")
    max_jump = jumps.get("max_jump_height_m")
    col5.metric("Selected max jump", "N/A" if max_jump is None else f"{max_jump:.2f} m")

    st.caption("P1-P6 are locked player identities. Missing observations remain gaps. Contact events need review before use as official statistics.")
    st.caption(metrics.get("note", ""))
    st.caption(jumps.get("note", ""))

    if box_score_df is not None and not box_score_df.empty:
        st.subheader("Automatic Contact Estimates (Unreviewed)")
        st.dataframe(box_score_df, use_container_width=True, hide_index=True)

    st.subheader("Reviewed Volleyball Stat Sheet")
    st.caption("Enter one row per action. A kill or attack error already counts as an attack attempt. Only checked rows count.")
    events = st.data_editor(
        pd.DataFrame(columns=["Time (s)", "Player", "Event", "Reviewed"]),
        key="reviewed_events_" + str(st.session_state.uploaded_signature),
        num_rows="dynamic",
        column_config={
            "Time (s)": st.column_config.NumberColumn(min_value=0.0),
            "Player": st.column_config.SelectboxColumn(options=[f"P{i}" for i in range(1, 7)], required=True),
            "Event": st.column_config.SelectboxColumn(options=EVENTS, required=True),
            "Reviewed": st.column_config.CheckboxColumn(default=False),
        },
        hide_index=True,
    )
    sheet = build_stat_sheet(events, [f"P{i}" for i in range(1, 7)])
    st.dataframe(sheet, hide_index=True)
    st.download_button("Download reviewed stat sheet", sheet.to_csv(index=False), "reviewed_stat_sheet.csv", "text/csv")
    st.download_button("Download reviewed events", events.to_csv(index=False), "reviewed_events.csv", "text/csv")

    if results.get("player_table") is not None and not results["player_table"].empty:
        st.subheader("Team Player Movement")
        st.dataframe(results["player_table"], use_container_width=True, hide_index=True)

    if contacts_df is not None and not contacts_df.empty:
        st.subheader("Ball Contact Guesses")
        st.dataframe(contacts_df, use_container_width=True, hide_index=True)

    top_down_path = results["top_down_path"]
    if Path(top_down_path).exists():
        st.image(top_down_path, caption="Near-side team and ball trajectory map", use_container_width=False)

    annotated_path = results["annotated_video_path"]
    if Path(annotated_path).exists() and Path(annotated_path).suffix.lower() == ".mp4":
        st.video(annotated_path)
    elif Path(annotated_path).exists():
        st.info(f"Annotated video was generated as {Path(annotated_path).name}. Download it below.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        _download_button(results["annotated_video_path"], "Download annotated video", "video/mp4")
    with col_b:
        _download_button(results["csv_path"], "Download all-player tracking CSV", "text/csv")
    with col_c:
        _download_button(results["top_down_path"], "Download team/ball court map", "image/png")

    volleyball_paths = results.get("volleyball_paths", {})
    if volleyball_paths:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            _download_button(volleyball_paths.get("team_box_score", ""), "Team box score CSV", "text/csv")
        with col2:
            _download_button(volleyball_paths.get("team_players", ""), "Team players CSV", "text/csv")
        with col3:
            _download_button(volleyball_paths.get("ball_contacts", ""), "Ball contacts CSV", "text/csv")
        with col4:
            _download_button(volleyball_paths.get("ball_tracking", ""), "Ball tracking CSV", "text/csv")

    preview_df = results["tracking_df"].head(200)
    st.subheader("Tracking Preview Rows")
    st.dataframe(preview_df, use_container_width=True, hide_index=True)


def main() -> None:
    ensure_workspace_directories()
    initialise_state()
    if st.session_state.get("build_version") != "foundation-3":
        for key, value in STATE_DEFAULTS.items():
            st.session_state[key] = value.copy() if isinstance(value, list) else value
        st.session_state.build_version = "foundation-3"

    st.title("AI Volleyball Back-View Team Analyzer")
    st.caption("Analysis foundation 3.0 | Fixed camera behind the baseline")
    st.info("Lock your six teammates, then review their movement and candidate ball contacts. Estimates need review before use as official statistics.")
    mode_cols = st.columns(4)
    mode_cols[0].metric("Primary mode", "Back-view")
    mode_cols[1].metric("Team focus", "6 slots")
    mode_cols[2].metric("Player identities", "P1-P6")
    mode_cols[3].metric("Ball path", "YOLO + CV")

    settings = render_sidebar()
    signature = (st.session_state.get("model_choice"), tuple(settings.items()))
    if st.session_state.get("pipeline_signature") != signature:
        st.session_state.preview_records = None
        st.session_state.preview_track_ids = []
        st.session_state.roster_seeds = None
        st.session_state.results = None
        st.session_state.pipeline_signature = signature

    render_upload_step()
    if st.session_state.video_path is None:
        return

    st.divider()
    render_video_summary()
    st.divider()
    render_calibration_step()

    if st.session_state.pixel_to_court_matrix is None:
        return

    st.divider()
    render_tracking_preview(settings)
    st.divider()
    render_player_selection()
    st.divider()
    render_output_generation(settings)
    render_results()


if __name__ == "__main__":
    main()
