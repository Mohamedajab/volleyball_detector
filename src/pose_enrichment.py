"""Run pose only where a tracked ball could contact a player."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .detection import estimate_pose_features


def enrich_hands_near_ball(video_path, player_df, ball_df, pose_model, max_players_per_frame=2):
    output = player_df.copy()
    wrist_columns = ("left_wrist_x", "left_wrist_y", "right_wrist_x", "right_wrist_y")
    for column in wrist_columns:
        if column not in output:
            output[column] = np.nan
    if pose_model is None or output.empty or ball_df.empty:
        return output

    balls = {int(row.frame_number): row for row in ball_df.itertuples(index=False)}
    players = {int(frame): group.index.tolist() for frame, group in output.groupby("frame_number")}
    wanted_frames = sorted(set(balls) & set(players))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return output
    try:
        for frame_number in wanted_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = cap.read()
            if not ok:
                continue
            ball = balls[frame_number]
            ball_pixel = np.array([ball.ball_center_x, ball.ball_center_y])
            ranked = []
            for index in players[frame_number]:
                row = output.loc[index]
                height = max(1.0, row.bbox_y2 - row.bbox_y1)
                hand_region = np.array([row.bbox_center_x, row.bbox_y1 + .2 * height])
                ranked.append((float(np.linalg.norm(ball_pixel - hand_region)) / height, index))
            for _, index in sorted(ranked)[:max_players_per_frame]:
                row = output.loc[index]
                features = estimate_pose_features(
                    pose_model, frame,
                    (row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2),
                )
                for column in wrist_columns:
                    if column in features:
                        output.at[index, column] = features[column]
    finally:
        cap.release()
    return output
