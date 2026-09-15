"""Vertical metric calibration from the court plane and net tape."""
from __future__ import annotations

import cv2
import numpy as np


def build_projection_matrix(
    court_to_pixel: np.ndarray,
    net_top_pixels: list[tuple[float, float]],
    net_height_m: float,
) -> tuple[np.ndarray, float]:
    """Recover the vertical camera column from two known net-top points."""
    if len(net_top_pixels) != 2 or net_height_m <= 0:
        raise ValueError("Two net-top points and a positive net height are required.")
    h = np.asarray(court_to_pixel, dtype=float)
    if h.shape != (3, 3) or not np.isfinite(h).all():
        raise ValueError("Court calibration matrix is invalid.")

    world_xy = ((0.0, 9.0), (9.0, 9.0))
    a = np.zeros((6, 5), dtype=float)
    b = np.zeros(6, dtype=float)
    for i, ((x, y), (u, v)) in enumerate(zip(world_xy, net_top_pixels)):
        ground = h @ np.array([x, y, 1.0])
        image = np.array([u, v, 1.0])
        for k in range(3):
            row = i * 3 + k
            a[row, k] = -net_height_m
            a[row, 3 + i] = image[k]
            b[row] = ground[k]
    solution, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    vertical = solution[:3]
    projection = np.column_stack((h[:, 0], h[:, 1], vertical, h[:, 2]))

    errors = []
    for (x, y), image in zip(world_xy, net_top_pixels):
        projected = project_point(projection, (x, y, net_height_m))
        errors.append(float(np.linalg.norm(projected - np.asarray(image))))
    return projection, float(np.mean(errors))


def project_point(projection: np.ndarray, point_xyz) -> np.ndarray:
    point = np.asarray([*point_xyz, 1.0], dtype=float)
    image = np.asarray(projection, dtype=float) @ point
    if abs(image[2]) < 1e-9:
        raise ValueError("Point projects at infinity.")
    return image[:2] / image[2]


def estimate_height_above_ground(
    projection: np.ndarray,
    ground_xy: tuple[float, float],
    image_pixel: tuple[float, float],
    maximum_height_m: float = 4.5,
) -> tuple[float | None, float]:
    """Fit height on a vertical line above a known ground point."""
    ground = project_point(projection, (ground_xy[0], ground_xy[1], 0.0))
    top = project_point(projection, (ground_xy[0], ground_xy[1], maximum_height_m))
    direction = top - ground
    denominator = float(direction @ direction)
    if denominator < 4.0:
        return None, float("inf")
    pixel = np.asarray(image_pixel, dtype=float)
    fraction = float(np.clip(((pixel - ground) @ direction) / denominator, 0.0, 1.0))

    # Refine because perspective makes the vertical image line non-uniform in z.
    candidates = np.linspace(max(0.0, fraction * maximum_height_m - 0.8),
                             min(maximum_height_m, fraction * maximum_height_m + 0.8), 161)
    residuals = [np.linalg.norm(project_point(projection, (*ground_xy, z)) - pixel) for z in candidates]
    best = int(np.argmin(residuals))
    return float(candidates[best]), float(residuals[best])


def draw_net_calibration(frame, net_top_pixels):
    result = frame.copy()
    for index, point in enumerate(net_top_pixels):
        p = tuple(np.round(point).astype(int))
        cv2.circle(result, p, 7, (255, 80, 30), -1, cv2.LINE_AA)
        cv2.putText(result, f"net top {index + 1}", (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 30), 2, cv2.LINE_AA)
    if len(net_top_pixels) == 2:
        cv2.line(result, tuple(map(int, net_top_pixels[0])), tuple(map(int, net_top_pixels[1])),
                 (255, 80, 30), 2, cv2.LINE_AA)
    return result
