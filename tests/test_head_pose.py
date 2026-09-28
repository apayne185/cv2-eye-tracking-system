from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from eyetrack.head_pose import _LM_IDS, _MODEL_3D, HeadPoseEstimator

W, H = 640, 480


def _synthetic_landmarks(est, rvec, tvec):
    """Project the 3D face model with a known pose into normalised landmarks."""
    pts, _ = cv2.projectPoints(_MODEL_3D, rvec, tvec, est.K, est.D)
    pts = pts.reshape(-1, 2)
    landmark = {i: SimpleNamespace(x=x / W, y=y / H) for i, (x, y) in zip(_LM_IDS, pts, strict=True)}
    return SimpleNamespace(landmark=landmark)


@pytest.fixture
def known_pose():
    rvec = np.array([[np.pi + 0.1], [0.2], [0.05]])
    tvec = np.array([[0.0], [0.0], [1500.0]])
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
