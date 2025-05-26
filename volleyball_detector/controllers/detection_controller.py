from ultralytics import YOLO
from models.player_model import Player
import cv2

def process_video(video_path, frame_interval=2.3, conf_threshold=0.5):
    """Process video and detect players"""
    model = YOLO("models/best.pt")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * frame_interval)
    frame_num = 0
    players = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % interval == 0:
            results = model(frame, conf=conf_threshold)
            for box in results[0].boxes:
                cls = int(box.cls[0])
                label = model.names[cls].lower()
                if label in ["team_one", "team_two"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame[y1:y2, x1:x2]
                    players.append(Player(frame_num, label, float(box.conf[0]), (x1, y1, x2, y2), crop))

        frame_num += 1

    cap.release()
    return players

