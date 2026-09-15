import numpy as np
from src.camera_3d import build_projection_matrix, estimate_height_above_ground, project_point


def test_net_points_recover_contact_height():
    projection = np.array([[120, 4, 8, 500], [2, 80, -100, 700], [0, .015, .002, 1]], dtype=float)
    homography = projection[:, [0, 1, 3]]
    net = [project_point(projection, (0, 9, 2.43)), project_point(projection, (9, 9, 2.43))]
    recovered, error = build_projection_matrix(homography, net, 2.43)
    pixel = project_point(projection, (4, 5, 3.05))
    height, residual = estimate_height_above_ground(recovered, (4, 5), pixel)
    assert error < 1e-6
    assert abs(height - 3.05) < 0.03
    assert residual < 1
