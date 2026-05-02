# Model Files

Place YOLO model files in this folder.

Recommended filenames:

- `best.pt` for a custom volleyball/player detector
- `yolov8n.pt` for the YOLOv8 nano object detector
- `yolov8n-pose.pt` for optional pose-based jump signal estimation

The app loads models in this order:

1. `models/best.pt`
2. `models/yolov8n.pt`
3. Ultralytics `yolov8n.pt` fallback

If `models/yolov8n-pose.pt` is missing, the app still works and uses bounding-box jump analysis.

