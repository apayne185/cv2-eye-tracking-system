
import cv2
import numpy as np
import pytest

from conftest import project_face
from eyetrack.head_pose import HeadPoseEstimator

W, H = 640, 480


def _synthetic_landmarks(est, rvec, tvec):
    """Project the 3D face model with a known pose into normalised landmarks."""
    return project_face(rvec, tvec)


@pytest.fixture
def known_pose():
    rvec = np.array([[np.pi + 0.1], [0.2], [0.05]])
    tvec = np.array([[0.0], [0.0], [600.0]])
    return rvec, tvec


def test_intrinsics_from_frame_size():
    est = HeadPoseEstimator(W, H)
    assert est.K[0, 0] == W
    assert est.K[0, 2] == W / 2
    assert est.K[1, 2] == H / 2


def test_no_pose_before_estimate():
    est = HeadPoseEstimator(W, H)
    assert est.rotation_matrix is None
    assert est.translation_vector is None
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    est.draw_axes(frame)
    assert not frame.any()


def test_estimate_recovers_known_pose(known_pose):
    rvec, tvec = known_pose
    est = HeadPoseEstimator(W, H)
    angles = est.estimate(_synthetic_landmarks(est, rvec, tvec), (H, W))

    assert all(isinstance(a, float) for a in angles)
    expected_R, _ = cv2.Rodrigues(rvec)
    np.testing.assert_allclose(est.rotation_matrix, expected_R, atol=1e-3)
    np.testing.assert_allclose(est.translation_vector, tvec, atol=0.5)


def test_draw_axes_after_estimate(known_pose):
    est = HeadPoseEstimator(W, H)
    est.estimate(_synthetic_landmarks(est, *known_pose), (H, W))
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    est.draw_axes(frame)
    assert frame.any()


def test_gaze_ray_drawn_in_front_of_camera():
    est = HeadPoseEstimator(W, H)
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    est.draw_gaze_ray(frame, np.array([0.0, 0.0, 600.0]), np.array([0.0, 0.0, -1.0]))
    assert frame.any()


def test_gaze_ray_behind_camera_is_skipped():
    est = HeadPoseEstimator(W, H)
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    est.draw_gaze_ray(frame, np.array([0.0, 0.0, -10.0]), np.array([0.0, 0.0, 1.0]))
    assert not frame.any()


def _rx(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _ry(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


@pytest.mark.parametrize("yaw, pitch", [(0, 0), (20, 0), (-35, 0), (0, 15), (0, -10), (25, 10)])
def test_angles_match_ground_truth(yaw, pitch):
    """A head squarely facing the camera is (0, 0, 0); turns and tilts read directly."""
    head = _ry(yaw) @ _rx(pitch)                    # head rotation in camera coordinates
    rvec, _ = cv2.Rodrigues(head @ np.diag([1.0, -1.0, -1.0]))   # as solvePnP sees the y-up model
    tvec = np.array([[0.0], [0.0], [600.0]])
    est = HeadPoseEstimator(W, H)
    p, y, r = est.estimate(_synthetic_landmarks(est, rvec, tvec), (H, W))
    assert (p, y, r) == pytest.approx((pitch, yaw, 0.0), abs=0.5)
