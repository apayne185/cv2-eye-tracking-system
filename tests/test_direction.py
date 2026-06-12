from direction import GazeDirectionEstimator


def test_centered_iris_no_head_movement_is_origin():
    est = GazeDirectionEstimator()
    dh, dv = est.estimate(0.5, 0.5, 0.0, 0.0)
    assert abs(dh) < 1e-9
    assert abs(dv) < 1e-9


def test_left_iris_gives_negative_dir_h():
    est = GazeDirectionEstimator()
    dh, _ = est.estimate(0.2, 0.5, 0.0, 0.0)
    assert dh < 0


def test_right_iris_gives_positive_dir_h():
    est = GazeDirectionEstimator()
    dh, _ = est.estimate(0.8, 0.5, 0.0, 0.0)
    assert dh > 0


def test_head_yaw_right_adds_positive_dir_h():
    est = GazeDirectionEstimator()
    dh_no_yaw, _ = est.estimate(0.5, 0.5, 0.0, 0.0)
    dh_yaw, _    = est.estimate(0.5, 0.5, 30.0, 0.0)
    assert dh_yaw > dh_no_yaw


def test_output_clamped_to_unit_range():
    est = GazeDirectionEstimator()
    dh, dv = est.estimate(1.0, 1.0, 90.0, 90.0)
    assert -1.0 <= dh <= 1.0
    assert -1.0 <= dv <= 1.0


def test_to_screen_point_straight_ahead_is_center():
    est = GazeDirectionEstimator()
    x, y = est.to_screen_point(0.0, 0.0, 1920, 1080)
    assert x == 960
    assert y == 540


def test_to_screen_point_full_left_is_zero():
    est = GazeDirectionEstimator()
    x, _ = est.to_screen_point(-1.0, 0.0, 1920, 1080)
    assert x == 0


# --- 3D gaze ray tests ---

import numpy as np


def test_ray_direction_is_unit_vector():
    est = GazeDirectionEstimator()
    R = np.eye(3)
    t = np.zeros((3, 1))
    _, direction = est.gaze_ray_3d(0.5, 0.5, R, t)
    assert abs(np.linalg.norm(direction) - 1.0) < 1e-6


def test_centered_iris_identity_pose_gives_forward_ray():
    est = GazeDirectionEstimator()
    R = np.eye(3)
    t = np.zeros((3, 1))
    _, direction = est.gaze_ray_3d(0.5, 0.5, R, t)
    # Identity pose + centered iris → gaze straight forward (+Z in camera space)
    assert direction[2] > 0.99
    assert abs(direction[0]) < 0.05
    assert abs(direction[1]) < 0.05


def test_right_iris_deviates_ray_rightward():
    est = GazeDirectionEstimator()
    R = np.eye(3)
    t = np.zeros((3, 1))
    _, d_center = est.gaze_ray_3d(0.5, 0.5, R, t)
    _, d_right  = est.gaze_ray_3d(0.8, 0.5, R, t)
    assert d_right[0] > d_center[0]


def test_down_iris_deviates_ray_downward():
    # Face model uses Y-up convention, so looking down = more negative Y component.
    # solvePnP's rotation matrix flips this to camera Y-down in real usage.
    est = GazeDirectionEstimator()
    R = np.eye(3)
    t = np.zeros((3, 1))
    _, d_center = est.gaze_ray_3d(0.5, 0.5, R, t)
    _, d_down   = est.gaze_ray_3d(0.5, 0.8, R, t)
    assert d_down[1] < d_center[1]


def test_ray_origin_matches_eye_midpoint_at_identity():
    est = GazeDirectionEstimator()
    R = np.eye(3)
    t = np.zeros((3, 1))
    origin, _ = est.gaze_ray_3d(0.5, 0.5, R, t)
    # With identity pose and zero translation, origin = eye midpoint in model space
    expected = np.array([0.0, 170.0, -135.0])
    assert np.allclose(origin, expected, atol=1e-6)
