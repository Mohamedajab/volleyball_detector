# AI Volleyball Performance Analyzer

## Overview

AI Volleyball Performance Analyzer is a Streamlit web application for analysing volleyball clips. It lets a user upload a video, manually calibrate the court, detect and track players, select one player, estimate movement and jump metrics, and export an annotated video plus tracking CSV.

The project is inspired by professional sports analytics platforms such as Balltime, but it is an academic and portfolio-focused implementation rather than a production-grade biomechanics system.

## Key Features

- Streamlit web interface with a guided step-by-step workflow
- Manual volleyball court calibration using four selected court corners
- Homography mapping from camera pixels to real-world court coordinates in metres
- Volleyball court line overlay on the calibration preview and output video
- YOLOv8 player detection with configurable confidence threshold
- Centroid-based player tracking fallback that works without external tracker configs
- Player ID selection after a short preview pass
- Selected-player movement trail and stats overlay in the exported video
- Top-down court movement map
- Movement distance, average speed, tracked time, and approximate jump estimates
- Downloadable `CSV`, annotated video, and court map outputs

## Demo Workflow

1. Start the app with `streamlit run app.py`.
2. Upload a volleyball video.
3. Choose a clear calibration frame.
4. Select the four court corners in this order:
   near left, near right, far right, far left.
5. Confirm the court overlay preview.
6. Run a short detection/tracking preview.
7. Select the player track ID to analyse.
8. Process the full video.
9. Download:
   `output/annotated_video.mp4`, `output/player_tracking.csv`, and `output/top_down_court.png`.

## Tech Stack

- Python
- Streamlit
- OpenCV
- Ultralytics YOLOv8
- NumPy
- Pandas
- SciPy
- Matplotlib
- streamlit-image-coordinates

## System Architecture

```text
app.py
src/
  config.py          App constants, model paths, court dimensions
  video_utils.py     Upload handling, frame extraction, video metadata
  detection.py       YOLO model loading and player detection
  tracking.py        Centroid tracker and tracking record generation
  calibration.py     Homography, court mapping, court line overlay
  analysis.py        Movement metrics and approximate jump detection
  visualisation.py   Bounding boxes, trails, overlays, top-down map
  export_utils.py    CSV and annotated video export
```

## How It Works

### Video Upload

The app saves uploaded clips into `uploads/` and reads metadata such as FPS, duration, frame count, and resolution with OpenCV.

### Court Calibration

The user selects four court corners on a clear frame. Manual calibration is the default because it is more reliable than automatic court detection for varied camera angles and university-project footage.

### Homography Mapping

OpenCV computes a perspective transform from selected image pixels to a real volleyball court:

- Width: 9 metres
- Length: 18 metres
- Centre line: 9 metres
- Attack lines: 6 metres and 12 metres from one baseline

Detected player foot positions use the bottom centre of each bounding box and are mapped into court metres.

### Player Detection

The detector loads models in this order:

1. `models/best.pt`
2. `models/yolov8n.pt`
3. Ultralytics `yolov8n.pt` fallback

For jump signal estimation, the optional pose model is loaded from `models/yolov8n-pose.pt` when available.

### Player Tracking

The current MVP uses a centroid tracker that matches detections frame-to-frame by nearest bottom-centre point. It keeps tracks alive briefly when detections disappear and removes stale tracks after a configurable number of missing frames.

### Movement and Jump Analysis

Movement distance and speed are calculated from homography-mapped court positions. Unrealistic displacement segments are ignored to reduce tracking error impact.

Jump estimates use the pose centre-of-mass vertical signal when the pose model is available. Otherwise, the app falls back to bounding-box centre movement. These estimates are approximate and should not be treated as precise biomechanical measurements.

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Running the App

```bash
streamlit run app.py
```

The application will open in your browser. If it does not open automatically, use the localhost URL printed by Streamlit.

## Output Files

Generated files are written to `output/`:

- `annotated_video.mp4`: Original video with court overlay, player boxes, selected-player trail, and stats
- `player_tracking.csv`: Per-frame tracking data for all tracked players
- `top_down_court.png`: Top-down movement path for the selected player

CSV columns:

```text
frame_number, timestamp_seconds, track_id, selected_player,
bbox_x1, bbox_y1, bbox_x2, bbox_y2,
bbox_center_x, bbox_center_y,
foot_pixel_x, foot_pixel_y,
court_x_m, court_y_m,
detection_confidence, jump_height_estimate_m, notes
```

## Limitations

- Court calibration depends on the user selecting accurate court corners.
- Homography maps ground-plane positions only; it does not solve full 3D player motion.
- Jump height estimates are approximate and depend on camera angle, detection quality, pose quality, and calibration quality.
- The centroid tracker can switch IDs when players overlap heavily or move very quickly.
- YOLO detection quality depends on the model weights available in `models/`.
- The app is designed as a working MVP for portfolio demonstration, not a certified sports science tool.

## Future Improvements

- Integrate ByteTrack or BoT-SORT when tracker dependencies/configs are available
- Add automatic court line detection as an optional helper
- Improve player re-identification after occlusion
- Add team/side segmentation
- Add richer event detection for jumps, attacks, serves, and blocks
- Add a small sample video and screenshots for the GitHub demo
- Package the app for deployment on Streamlit Community Cloud

## Author

Mohamed Ajab

