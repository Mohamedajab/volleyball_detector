from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"

APP_TITLE = "AI Volleyball Performance Analyzer"
APP_DESCRIPTION = (
    "Upload a volleyball clip, calibrate the court, track a player, "
    "and generate movement/jump analytics."
)

SUPPORTED_VIDEO_TYPES = ("mp4", "mov", "avi", "mkv")
DEFAULT_FPS = 30.0

COURT_WIDTH_M = 9.0
COURT_LENGTH_M = 18.0
CENTER_LINE_Y_M = 9.0
ATTACK_LINE_Y_M = (6.0, 12.0)
COURT_BOUNDARY_MARGIN_M = 1.0

COURT_CORNERS_M = (
    (0.0, 0.0),
    (COURT_WIDTH_M, 0.0),
    (COURT_WIDTH_M, COURT_LENGTH_M),
    (0.0, COURT_LENGTH_M),
)

MODEL_PRIORITY_FILENAMES = ("best.pt", "yolo26s.pt", "yolo11s.pt", "yolov8n.pt")
POSE_MODEL_FILENAME = "yolov8n-pose.pt"

DEFAULT_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_MAX_TRACK_DISTANCE_PX = 90.0
DEFAULT_MAX_MISSING_FRAMES = 18
DEFAULT_PREVIEW_SECONDS = 5.0
MAX_REASONABLE_SPEED_MPS = 10.0
DEFAULT_TRACKER_BACKEND = str(PROJECT_ROOT / "configs" / "botsort_volleyball.yaml")
TRACKER_BACKENDS = (DEFAULT_TRACKER_BACKEND, "botsort.yaml", "bytetrack.yaml", "ocsort.yaml", "centroid fallback")

ESTIMATED_PLAYER_HEIGHT_M = 1.85
MIN_JUMP_HEIGHT_M = 0.08
MAX_JUMP_HEIGHT_M = 1.20

CSV_COLUMNS = [
    "frame_number",
    "timestamp_seconds",
    "track_id",
    "selected_player",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_center_x",
    "bbox_center_y",
    "foot_pixel_x",
    "foot_pixel_y",
    "court_x_m",
    "court_y_m",
    "detection_confidence",
    "jump_height_estimate_m",
    "notes",
]
