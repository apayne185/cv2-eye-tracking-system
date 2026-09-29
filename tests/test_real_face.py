"""
Pipeline behaviour on real footage (tests/fixtures/face_clip.mp4): a
public-domain NASA interview clip, face turned roughly 45° from camera.

These guard properties a synthetic face can't: MediaPipe detection on a
real face, temporally stable pose, and angles in a physically sensible
range (a level head once read pitch ≈ ±180°).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eyetrack.cli import main
from eyetrack.pipeline import FrameProcessor
from eyetrack.sources import VideoSource

CLIP = Path(__file__).parent / "fixtures" / "face_clip.mp4"
N_FRAMES = 120


@pytest.fixture(scope="module")
def results():
    with VideoSource(str(CLIP)) as src:
        proc = FrameProcessor(src.width, src.height)
        return [proc.process(frame, ts) for frame, ts in src.frames()]


@pytest.fixture(scope="module")
def df(results):
    return pd.DataFrame([r.to_row() for r in results])


def test_face_detected_on_nearly_every_frame(results):
    assert len(results) == N_FRAMES
    assert sum(r.face_detected for r in results) >= 0.95 * N_FRAMES


def test_head_pose_is_physically_plausible(df):
    for angle in ("pitch", "yaw", "roll"):
        assert df[angle].abs().max() < 60, f"{angle} out of range"


def test_turned_head_reads_as_large_stable_yaw(df):
    assert df["yaw"].abs().mean() > 25
    assert df["yaw"].std() < 5            # stable frame to frame


def test_gaze_direction_not_saturated(df):
    saturated = (df["dir_v"].abs() >= 1) | (df["dir_h"].abs() >= 1)
    assert saturated.mean() < 0.10


def test_gaze_rays_are_unit_vectors(results):
    norms = [np.linalg.norm(r.ray_direction) for r in results if r.ray_direction is not None]
    assert len(norms) >= 0.95 * N_FRAMES
    assert norms == pytest.approx([1.0] * len(norms))


def test_cli_run_on_clip(tmp_path):
    assert main(["run", "--source", str(CLIP), "--no-display", "--output-dir", str(tmp_path)]) == 0
    out = pd.read_csv(next(tmp_path.glob("gaze_*.csv")))
    assert len(out) == N_FRAMES
    assert out["gaze_x"].notna().mean() >= 0.95
    assert out["timestamp"].iloc[-1] == pytest.approx((N_FRAMES - 1) / 29.97, abs=0.01)
