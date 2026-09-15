# AI Volleyball Performance Analyzer

## Overview

AI Volleyball Performance Analyzer is a Streamlit web application for analysing volleyball clips. It lets a user upload a video, manually calibrate the court, detect and track players, select one player, estimate movement and jump metrics, and export an annotated video plus tracking CSV.

The project is inspired by professional sports analytics platforms such as Balltime, but it is an academic and portfolio-focused implementation rather than a production-grade biomechanics system.

## Key Features

- Streamlit web interface with a guided step-by-step workflow
- Manual volleyball court calibration using four selected court corners
- Homography mapping from camera pixels to real-world court coordinates in metres
- Volleyball court line overlay on the calibration preview and output video
- YOLO26 player detection by default, with larger and local custom-weight choices
- Back-view near-side team mode for the six players facing the net away from camera
- Appearance-assisted BoT-SORT tracking, with ByteTrack, OC-SORT and centroid alternatives
- Explicit six-player roster lock with stable P1-P6 identities independent of court rotation
- Temporal ball tracking, best used with volleyball-trained models/ball.pt
- Two-point net-tape calibration for metric contact-height estimates
- Wrist pose analysis focused around likely ball contacts
- Team box-score style CSVs for movement, contacts, attack-touch guesses, set-touch guesses, and reception/pass-touch guesses
- Selected-player movement trail and stats overlay in the exported video
- Top-down court maps for team/player movement and ball trajectory
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
7. Pick a frame showing all six teammates, lock those six detections, and choose P1-P6 for the highlighted trail.
8. Process the full video.
9. Download:
   `output/annotated_video.mp4`, `output/player_tracking.csv`, and `output/top_down_court.png`.

## Tech Stack

- Python
- Streamlit
- OpenCV
- Ultralytics YOLO26
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
  tracking.py        Multi-object tracking and frame observations
  roster.py          Locked six-player identity association
  calibration.py     Homography, court mapping, court line overlay
  camera_3d.py       Net-tape vertical calibration and height fitting
  pose_enrichment.py Wrist pose around tracked-ball frames
  analysis.py        Movement metrics and approximate jump detection
  visualisation.py   Bounding boxes, trails, overlays, top-down map
  ball_tracking.py   Temporal ball association and motion fallback
  stat_sheet.py      Reviewed volleyball stat calculations
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

The default detector is YOLO26 small at 1280-pixel inference. The sidebar also offers YOLO26 medium, YOLO11 small, and local custom weights. Official weights download on first use. A custom detector belongs at models/best.pt and must expose a person, player, or athlete class.

Hand pose uses a local models/yolov8n-pose.pt when present and otherwise tries YOLO26 nano pose. Pose runs only for players nearest the ball on ball-visible frames.

### Player Tracking

The app is designed for a fixed, centred camera behind the analysed team. Appearance-assisted BoT-SORT is the default. The user selects six near-side detections in one frame. A second global assignment layer keeps P1-P6 using tracker continuity, time gaps, and plausible court movement. The roster never grows beyond six and players are not renamed when they rotate.

Long or ambiguous gaps remain missing instead of being assigned to another person. Substitutions require a new roster lock.

### Ball and Contact Analysis

Ball candidates are associated over time; isolated detections and implausible jumps are rejected. Classical motion candidates can extend an established track but cannot start one. Useful accuracy requires a volleyball-specific models/ball.pt.

Four court corners measure floor positions. Contact height additionally requires clicks on both net-tape endpoints and the correct official net height. A contact candidate requires wrist proximity plus a ball-path change. Height is fitted above the matched player's ground position and rejected when reprojection error is too large.

The top-down ball map is an image projection onto the floor, not the actual airborne 3D path. Defensible 3D trajectory needs a second synchronized camera or a validated monocular model trained with 3D ground truth.

### Movement and Jump Analysis

Movement distance and speed are calculated from homography-mapped court positions. Unrealistic displacement segments are ignored to reduce tracking error impact.

Jump estimates use bounding-box motion and remain approximate. Contact height uses the separate net-tape geometry and wrist/ball event pipeline. Automatic contacts are review candidates. Kills, errors, aces and assists are calculated only from events checked in the reviewed stat editor.

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

- `annotated_video.mp4`: Original video with court overlay, player boxes, near-side team highlight, selected-player trail, ball trail when detected, and stats
- `player_tracking.csv`: Per-frame tracking data for all tracked players
- `team_players.csv`: Near-side team player movement summary
- `team_box_score.csv`: Volleyball-style heuristic stat sheet for the near-side team
- `ball_tracking.csv`: Ball detections and approximate mapped court positions
- `ball_contacts.csv`: Heuristic ball contact/contact-height events
- `team_ball_court_map.png`: Top-down team and ball trajectory map
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

- The app assumes a fixed centred back-view recording. Cuts, zooms, pans and broadcast camera changes invalidate calibration.
- Court calibration depends on the user selecting accurate court corners.
- Floor homography does not recover airborne 3D ball position. Net calibration estimates height only at a matched player's horizontal position.
- Contact height depends on accurate corners, net endpoints, feet, wrists and ball centre. Poor fits are rejected, but accuracy has not yet been measured against ground truth.
- The general sports-ball model can miss small, blurred or occluded volleyballs. Train a volleyball-specific ball model for useful results.
- Jump height estimates are approximate and depend on camera angle, detection quality, pose quality, and calibration quality.
- Appearance tracking can still switch same-uniform players in heavy overlap. P1-P6 are internal identities, not recognized shirt numbers.
- YOLO detection quality depends on the model weights available in `models/`.
- The app is designed as a working MVP for portfolio demonstration, not a certified sports science tool and not a clone of Balltime proprietary AI.

## Future Improvements

- Extract and label volleyball frames with tools/extract_ball_frames.py, then train models/ball.pt with tools/train_ball_model.py
- Add automatic court line detection as an optional helper
- Evaluate ball precision/recall, identity switches, contact-frame error and height error on labelled clips
- Add jersey-number OCR with user correction
- Add team/side segmentation
- Train a temporal action model for set, attack, serve, pass, dig and block classification
- Add synchronized second-camera calibration for defensible 3D ball trajectories
- Add a small sample video and screenshots for the GitHub demo
- Package the app for deployment on Streamlit Community Cloud

## Author

Mohamed Ajab
