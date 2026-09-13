from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.analysis import (
    add_jump_annotations,
    calculate_movement_metrics,
    detect_jumps,
    records_to_dataframe,
    summarise_jumps,
)
from src.calibration import compute_homography, draw_calibration_points, draw_court_lines, validate_corner_points
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
from src.detection import detect_balls, load_ball_model, load_detection_model, load_pose_model
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
from src.visualisation import draw_player_boxes, generate_team_ball_court_map, generate_top_down_court
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
    "preview_records": None,
    "preview_track_ids": [],
    "preview_frame_number": None,
    "selected_track_id": None,
    "results": None,
}


@st.cache_resource(show_spinner=False)
def cached_detection_model():
    return load_detection_model()


@st.cache_resource(show_spinner=False)
def cached_pose_model():
    return load_pose_model()




@st.cache_resource(show_spinner=False)
def cached_ball_model():
    return load_ball_model()


@st.cache_data(show_spinner=False)
def cached_extract_frame(video_path: str, frame_number: int):
    return extract_frame(video_path, frame_number)


def initialise_state() -> None:
    for key, value in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, list) else value


def reset_for_new_video(video_path: Path, signature: str) -> None:
    st.session_state.video_path = str(video_path)
    st.session_state.uploaded_signature = signature
    st.session_state.metadata = read_video_metadata(video_path)
    st.session_state.calibration_frame_number = 0
    st.session_state.corner_points = []
    st.session_state.last_click_signature = None
    st.session_state.pixel_to_court_matrix = None
    st.session_state.court_to_pixel_matrix = None
    st.session_state.preview_records = None
    st.session_state.preview_track_ids = []
    st.session_state.preview_frame_number = None
    st.session_state.selected_track_id = None
    st.session_state.results = None
    cached_extract_frame.clear()




def run_ball_tracking_pass(video_path: str, ball_model, pixel_to_court_matrix, fps: float, frame_limit: int | None, progress_callback=None):
    if ball_model is None:
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames_to_process = min(total_frames, frame_limit) if frame_limit else total_frames
    records = []
    frame_number = 0
    try:
        while True:
            if frame_limit is not None and frame_number >= frame_limit:
                break
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_number / fps if fps > 0 else 0.0
            balls = detect_balls(ball_model, frame, confidence_threshold=0.12)
            if balls:
                records.append(estimate_ball_record(balls[0], frame_number, timestamp, pixel_to_court_matrix))
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
    confidence = st.sidebar.slider("Detection confidence", 0.10, 0.90, DEFAULT_CONFIDENCE_THRESHOLD, 0.05)
    tracker_backend = st.sidebar.selectbox(
        "Tracking backend",
        options=list(TRACKER_BACKENDS),
        index=list(TRACKER_BACKENDS).index(DEFAULT_TRACKER_BACKEND),
        help="ByteTrack is the default. Try BoT-SORT when player IDs swap during overlaps. Use centroid fallback only if YOLO tracking fails.",
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
    use_pose = st.sidebar.checkbox("Use pose model for jump signal when available", value=True)

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
            st.success("Court calibration confirmed.")
            st.rerun()


def _load_detector_with_ui():
    try:
        model, source = cached_detection_model()
        st.caption(f"Detection model: {source}")
        return model
    except Exception as exc:
        st.error(f"Model could not be loaded: {exc}")
        return None


def render_tracking_preview(settings: dict) -> None:
    st.subheader("Step 4: Run Detection and Tracking Preview")
    if st.session_state.pixel_to_court_matrix is None:
        st.warning("Confirm court calibration before running tracking.")
        return

    metadata = st.session_state.metadata
    frame_limit = max(1, int(metadata["fps"] * settings["preview_seconds"]))
    frame_limit = min(frame_limit, metadata["frame_count"])
    st.write(f"Preview will process the first {frame_limit} frame(s).")

    if st.button("Run tracking preview", type="primary"):
        model = _load_detector_with_ui()
        if model is None:
            return

        progress = st.progress(0, text="Starting preview tracking...")

        def update_progress(value: float, text: str) -> None:
            progress.progress(value, text=text)

        try:
            records = run_tracking_on_video(
                video_path=st.session_state.video_path,
                detector_model=model,
                pixel_to_court_matrix=st.session_state.pixel_to_court_matrix,
                confidence_threshold=settings["confidence"],
                max_distance_px=settings["max_distance"],
                max_missing_frames=settings["max_missing"],
                frame_limit=frame_limit,
                progress_callback=update_progress,
                tracker_backend=settings["tracker_backend"],
            )
        except Exception as exc:
            progress.empty()
            st.error(f"Tracking failed: {exc}")
            return

        progress.empty()
        df = records_to_dataframe(records)
        if df.empty:
            st.session_state.preview_records = None
            st.session_state.preview_track_ids = []
            st.error("No players detected in the preview. Try a lower confidence threshold or a clearer video.")
            return

        track_counts = df.groupby("track_id").size().sort_values(ascending=False)
        track_ids = [int(track_id) for track_id in track_counts.index.tolist()]
        st.session_state.preview_records = records
        st.session_state.preview_track_ids = track_ids
        st.session_state.selected_track_id = track_ids[0] if track_ids else None
        st.session_state.preview_frame_number = int(df.groupby("frame_number").size().sort_values(ascending=False).index[0])
        st.success(f"Detected {len(track_ids)} track ID(s) in the preview. Defaulted to the longest-lived track.")
        st.rerun()

    if st.session_state.preview_records:
        df = records_to_dataframe(st.session_state.preview_records)
        preview_frame_number = st.session_state.preview_frame_number or int(df["frame_number"].max())
        frame = cached_extract_frame(st.session_state.video_path, preview_frame_number)
        if frame is not None:
            frame_records = df[df["frame_number"] == preview_frame_number]
            preview = draw_court_lines(frame, st.session_state.court_to_pixel_matrix)
            preview = draw_player_boxes(preview, frame_records, selected_track_id=st.session_state.selected_track_id)
            st.image(bgr_to_rgb(preview), caption=f"Preview frame {preview_frame_number} with track IDs", use_container_width=True)

        counts = df.groupby("track_id").size().reset_index(name="preview_frames_detected")
        counts = counts.sort_values("preview_frames_detected", ascending=False)
        st.dataframe(counts, use_container_width=True, hide_index=True)


def render_player_selection() -> None:
    st.subheader("Step 5: Confirm Analysis Track")
    track_ids = st.session_state.preview_track_ids
    if not track_ids:
        st.info("Run the tracking preview to populate player IDs. The full output will focus on the near-side team of up to six players.")
        return

    current = st.session_state.selected_track_id if st.session_state.selected_track_id in track_ids else track_ids[0]
    selected = st.selectbox("Player track ID", options=track_ids, index=track_ids.index(current))
    st.session_state.selected_track_id = int(selected)
    st.success(f"Selected Player ID: {selected}")


def render_output_generation(settings: dict) -> None:
    st.subheader("Step 6: Generate Outputs")
    if st.session_state.selected_track_id is None:
        st.warning("Select a player before generating outputs.")
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
            pose_model=pose_model,
            selected_track_id=selected_id,
            tracker_backend=settings["tracker_backend"],
        )
    except Exception as exc:
        tracking_progress.empty()
        st.error(f"Full-video tracking failed: {exc}")
        return

    tracking_progress.empty()
    df = records_to_dataframe(records, selected_track_id=selected_id)
    if df.empty:
        st.error("No players were detected, so outputs could not be generated.")
        return

    if selected_id not in set(df["track_id"].astype(int).unique()):
        st.warning("The selected player ID was not found in the full pass. Outputs will still include all tracked players.")

    movement_metrics = calculate_movement_metrics(df, selected_id, metadata["fps"])
    jump_events, jump_note = detect_jumps(df, selected_id, metadata["fps"])
    df = add_jump_annotations(df, selected_id, jump_events, jump_note)
    jump_summary = summarise_jumps(jump_events, jump_note)

    team_track_ids = select_near_side_team(df, max_players=6)
    if len(team_track_ids) < 6:
        st.warning(f"Detected {len(team_track_ids)} likely near-side team player(s). For best results use a centred back-view clip where all six players are visible.")
    df = tag_team_players(df, team_track_ids)
    player_table = estimate_player_table(df, team_track_ids, metadata["fps"])

    ball_model = None
    ball_source = None
    try:
        ball_model, ball_source = cached_ball_model()
    except Exception:
        ball_model = None
    if ball_model is not None:
        st.caption(f"Ball model: {ball_source}")
    else:
        st.info("Ball model unavailable. Team movement stats will still be generated, but ball trajectory/contact outputs will be empty.")

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
    contacts_df = estimate_contacts(df, ball_df, team_track_ids, metadata["fps"])
    box_score_df = estimate_team_box_score(player_table, contacts_df)

    try:
        top_down_path = generate_team_ball_court_map(df, ball_df, team_track_ids, OUTPUT_DIR / "team_ball_court_map.png")
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
            team_track_ids=team_track_ids,
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
        "team_track_ids": team_track_ids,
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
    col1.metric("Near-side players", len(team_ids))
    col2.metric("Ball detections", 0 if ball_df is None else len(ball_df))
    col3.metric("Contact guesses", 0 if contacts_df is None else len(contacts_df))
    col4.metric("Selected distance", f"{metrics.get('total_distance_m', 0):.2f} m")
    max_jump = jumps.get("max_jump_height_m")
    col5.metric("Selected max jump", "N/A" if max_jump is None else f"{max_jump:.2f} m")

    st.caption("Near-side team mode assumes a centred back-view recording from behind your team. Ball/contact stats are heuristic unless you provide a volleyball-trained ball model.")
    st.caption(metrics.get("note", ""))
    st.caption(jumps.get("note", ""))

    if box_score_df is not None and not box_score_df.empty:
        st.subheader("Near-Side Team Box Score")
        st.dataframe(box_score_df, use_container_width=True, hide_index=True)

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

    st.title(APP_TITLE)
    st.write(APP_DESCRIPTION)

    settings = render_sidebar()

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
