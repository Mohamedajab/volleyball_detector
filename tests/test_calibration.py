import numpy as np

from src.calibration import compute_homography, map_pixel_to_court


def test_homography_maps_corners_to_volleyball_court_dimensions():
    points = [(10, 10), (910, 10), (910, 1810), (10, 1810)]
    pixel_to_court, _ = compute_homography(points)

    mapped = map_pixel_to_court(points, pixel_to_court)

    assert np.allclose(mapped[0], [0, 0], atol=1e-4)
    assert np.allclose(mapped[1], [9, 0], atol=1e-4)
    assert np.allclose(mapped[2], [9, 18], atol=1e-4)
    assert np.allclose(mapped[3], [0, 18], atol=1e-4)

