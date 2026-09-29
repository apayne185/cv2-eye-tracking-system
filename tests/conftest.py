from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from eyetrack.eye_tracker import EyeTracker
from eyetrack.head_pose import _MODEL_3D, HeadPoseEstimator

W, H = 640, 480


def project_face(rvec, tvec, w=W, h=H):
    """
    478 FaceMesh-style landmarks from the canonical face model at a known
    pose: all 468 mesh points projected, irises centred in each eye.
    """
    est = HeadPoseEstimator(w, h)
    pts, _ = cv2.projectPoints(_MODEL_3D, rvec, tvec, est.K, est.D)
    pts = pts.reshape(-1, 2)
    lm = [SimpleNamespace(x=x / w, y=y / h, z=0.0) for x, y in pts]

    # iris centre = midpoint of the eye corners (x) and lids (y);
    # 469-472 / 474-477 are the iris rings, placed on the centre
    for corners, lids, first in (((33, 133), (159, 145), 468), ((263, 362), (386, 374), 473)):
        cx, cy = pts[list(corners), 0].mean(), pts[list(lids), 1].mean()
        lm.extend(SimpleNamespace(x=cx / w, y=cy / h, z=0.0) for _ in range(5))
        assert len(lm) == first + 5
    return SimpleNamespace(landmark=lm)


# rvec for a head squarely facing the camera (model is y-up, camera y-down)
FRONTAL = np.array([[np.pi], [0.0], [0.0]])


def synthetic_face_landmarks():
    """Frontal face 60 cm from the camera, eyes open, gaze straight ahead."""
    return project_face(FRONTAL, np.array([[0.0], [0.0], [600.0]]))


class FakeEyeTracker(EyeTracker):
    """EyeTracker with MediaPipe replaced by a fixed landmark set (or none)."""

    def __init__(self, landmarks=None):
        super().__init__()
        self.landmarks = landmarks

    def process(self, frame):
        faces = [self.landmarks] if self.landmarks is not None else None
        return SimpleNamespace(multi_face_landmarks=faces)


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """
    Run each test from an empty directory so default paths (models/,
    data/) never pick up a developer's local files; CI has none.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def frame():
    return np.zeros((H, W, 3), dtype=np.uint8)


@pytest.fixture
def face_tracker():
    return FakeEyeTracker(synthetic_face_landmarks())


@pytest.fixture
def no_face_tracker():
    return FakeEyeTracker(None)
