# Model Files

Place YOLO model files in this folder.

Recommended filenames:

- `best.pt` for a custom player detector
- `ball.pt` for a custom volleyball detector
- `yolov8n-pose.pt` for local pose weights

YOLO26 small is the default player detector. Choose `custom/local` in the app to use `best.pt`. The ball pipeline prefers `ball.pt`, then falls back to the general sports-ball class. Pose falls back to YOLO26 nano pose when local weights are absent.

Large weights are ignored by Git. Train ball weights with `tools/train_ball_model.py` after preparing YOLO-format labels.
