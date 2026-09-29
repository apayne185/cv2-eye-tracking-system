import numpy as np
import pytest

from eyetrack.direction import GazeDirectionEstimator
from eyetrack.head_pose import EYE_MIDPOINT_MODEL


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


def test_head_turned_toward_image_left_lowers_dir_h():
    # +yaw = face turned toward image left (HeadPoseEstimator convention)
    est = GazeDirectionEstimator()
    dh, _ = est.estimate(0.5, 0.5, 30.0, 0.0)
    assert dh < 0


def test_head_tilted_down_raises_dir_v():
    # +pitch = face tilted down; +dir_v = toward image bottom
    est = GazeDirectionEstimator()
    _, dv = est.estimate(0.5, 0.5, 0.0, 20.0)
    assert dv > 0


def _head_rotation(yaw, pitch):
    """solvePnP-style rotation for a head at (yaw, pitch) in camera coordinates."""
    y, p = np.radians(yaw), np.radians(pitch)
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return ry @ rx @ np.diag([1.0, -1.0, -1.0])


@pytest.mark.parametrize("ratio_h, ratio_v, yaw, pitch", [
    (0.5, 0.5, 30.0, 0.0),     # head only
    (0.5, 0.5, -30.0, 0.0),
    (0.5, 0.5, 0.0, 20.0),
    (0.5, 0.5, 0.0, -20.0),
    (0.8, 0.5, 0.0, 0.0),      # eyes only
    (0.2, 0.5, 0.0, 0.0),
    (0.5, 0.8, 0.0, 0.0),
    (0.5, 0.2, 0.0, 0.0),
    (0.75, 0.5, -25.0, 0.0),   # head and eyes the same way
    (0.25, 0.5, 25.0, 0.0),
    (0.5, 0.75, 0.0, 15.0),
])
def test_2d_direction_agrees_with_3d_ray(ratio_h, ratio_v, yaw, pitch):
    """The fused 2-D direction must point the same way as the 3-D gaze ray."""
    est = GazeDirectionEstimator()
    dh, dv = est.estimate(ratio_h, ratio_v, yaw, pitch)
    _, ray = est.gaze_ray_3d(ratio_h, ratio_v, _head_rotation(yaw, pitch), np.zeros(3))
    # the ray points back toward the camera (-z); x, y are image right / down
    for d2, d3 in ((dh, ray[0]), (dv, ray[1])):
        if abs(d3) > 1e-6:
            assert np.sign(d2) == np.sign(d3)
        else:
            assert abs(d2) < 1e-6


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
    assert np.allclose(origin, EYE_MIDPOINT_MODEL, atol=1e-6)
