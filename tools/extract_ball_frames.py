"""Extract evenly spaced frames for manual YOLO ball annotation."""
from __future__ import annotations

import argparse
from pathlib import Path
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--every", type=int, default=10, help="Save every Nth frame")
    parser.add_argument("--output", type=Path, default=Path("sample_data/ball_dataset/images/unlabelled"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    frame = 0
    saved = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if frame % max(1, args.every) == 0:
            cv2.imwrite(str(args.output / f"{args.video.stem}_{frame:06d}.jpg"), image)
            saved += 1
        frame += 1
    cap.release()
    print(f"Saved {saved} frames to {args.output}")


if __name__ == "__main__":
    main()
