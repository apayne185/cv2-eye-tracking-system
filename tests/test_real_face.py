"""
Pipeline behaviour on real footage (tests/fixtures/face_clip.mp4): a
public-domain NASA interview clip, face turned roughly 45° from camera.

These guard properties a synthetic face can't: MediaPipe detection on a
real face, temporally stable pose, and angles in a physically sensible
range (a level head once read pitch ≈ ±180°), and a 2-D gaze direction
that agrees with the 3-D gaze ray.
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


def test_fused_direction_agrees_with_gaze_ray(df):
    # head and eyes both point toward image right: the 2-D direction and the
    # 3-D ray must agree (a sign error once made them disagree on every frame)
    assert (df["dir_h"] > 0).mean() >= 0.95
    assert (np.sign(df["dir_h"]) == np.sign(df["ray_dx"])).mean() >= 0.95


def test_level_gaze_reads_level(df):
    # head tilted ~20° down, eyes raised: the net gaze is about level, as the
    # 3-D ray shows; dir_v was once pinned at +1, then read strongly upward
    assert abs(df["ray_dy"].mean()) < 0.25
    assert abs(df["dir_v"].mean()) < 0.25
    assert (df["dir_v"].abs() >= 1).mean() < 0.10


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
