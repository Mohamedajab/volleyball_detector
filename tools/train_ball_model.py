"""Train a volleyball detector from YOLO-format labels."""
from pathlib import Path
from ultralytics import YOLO


def main():
    data = Path("training/ball.yaml")
    if not data.exists():
        raise SystemExit("Create training/ball.yaml from ball.yaml.example after labelling frames.")
    result = YOLO("yolo26s.pt").train(
        data=str(data), epochs=100, imgsz=1280, batch=-1,
        project="runs/ball", name="volleyball", patience=20,
    )
    print("Training complete. Copy the validated best.pt to models/ball.pt.")
    print(result.save_dir)


if __name__ == "__main__":
    main()
