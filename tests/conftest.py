from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from eyetrack.eye_tracker import EyeTracker
from eyetrack.head_pose import _LM_IDS, _MODEL_3D, HeadPoseEstimator

W, H = 640, 480


def _pt(x, y):
    return SimpleNamespace(x=x / W, y=y / H, z=0.0)


def synthetic_face_landmarks():
    """
    478 FaceMesh-style landmarks for a frontal face: head-pose points are
    projected from the 3D model at a known pose, eyes are open with the
    iris centred, and every other landmark sits at the frame centre.
    """
    lm = [_pt(W / 2, H / 2) for _ in range(478)]

    est = HeadPoseEstimator(W, H)
    rvec = np.array([[np.pi], [0.0], [0.0]])
    tvec = np.array([[0.0], [0.0], [2500.0]])
    pts, _ = cv2.projectPoints(_MODEL_3D, rvec, tvec, est.K, est.D)
    for i, (x, y) in zip(_LM_IDS, pts.reshape(-1, 2), strict=True):
        lm[i] = _pt(x, y)

    # eye box: outer/inner corners, top/bottom lids, EAR points, iris centre
    for outer, inner, top, bot, ear_ids, iris, cx in (
        (33, 133, 159, 145, (160, 158, 153, 144), 468, 280),
        (263, 362, 386, 374, (387, 385, 380, 373), 473, 360),
    ):
        lm[outer], lm[inner] = _pt(cx - 20, 200), _pt(cx + 20, 200)
        lm[top], lm[bot] = _pt(cx, 190), _pt(cx, 210)
        up1, up2, lo2, lo1 = ear_ids
        lm[up1], lm[up2] = _pt(cx - 7, 190), _pt(cx + 7, 190)
        lm[lo2], lm[lo1] = _pt(cx + 7, 210), _pt(cx - 7, 210)
        lm[iris] = _pt(cx, 200)

    return SimpleNamespace(landmark=lm)


class FakeEyeTracker(EyeTracker):
    """EyeTracker with MediaPipe replaced by a fixed landmark set (or none)."""

    def __init__(self, landmarks=None):
        super().__init__()
        self.landmarks = landmarks

    def process(self, frame):
        faces = [self.landmarks] if self.landmarks is not None else None
        return SimpleNamespace(multi_face_landmarks=faces)


@pytest.fixture
def frame():
    return np.zeros((H, W, 3), dtype=np.uint8)


@pytest.fixture
def face_tracker():
    return FakeEyeTracker(synthetic_face_landmarks())


@pytest.fixture
def no_face_tracker():
    return FakeEyeTracker(None)
