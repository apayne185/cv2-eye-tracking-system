import json
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from calibration import CALIB_POINTS_NORM, GazeCalibrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perfect_samples():
    """
    Synthetic samples where the mapping is identity:
    ratio_h == target_nx, ratio_v == target_ny.
    A perfect linear fit should reproduce the targets exactly.
    """
    return [(nx, ny, nx, ny) for nx, ny in CALIB_POINTS_NORM]


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

def test_fit_raises_on_too_few_points():
    with pytest.raises(ValueError, match='3'):
        GazeCalibrator().fit([(0.5, 0.5, 0.5, 0.5), (0.1, 0.1, 0.1, 0.1)])


def test_fit_marks_as_fitted():
    clf = GazeCalibrator()
    assert not clf._fitted
    clf.fit(_perfect_samples())
    assert clf._fitted


def test_fit_perfect_identity():
    """With identity data the regressor should reproduce targets almost exactly."""
    clf = GazeCalibrator().fit(_perfect_samples())
    for nx, ny in CALIB_POINTS_NORM:
        px, py = clf.to_screen_point(nx, ny, 1000, 1000)
        assert abs(px - int(nx * 1000)) <= 2
        assert abs(py - int(ny * 1000)) <= 2


def test_fit_non_trivial_mapping():
    """Verify that a non-identity linear mapping is learnt correctly."""
    # ratio_h -> 1 - nx (mirror), ratio_v -> ny (identity)
    samples = [(nx, ny, 1.0 - nx, ny) for nx, ny in CALIB_POINTS_NORM]
    clf = GazeCalibrator().fit(samples)
    # Centre point: ratio (0.5, 0.5) -> screen (0.5, 0.5)
    px, py = clf.to_screen_point(0.5, 0.5, 100, 100)
    assert abs(px - 50) <= 2
    assert abs(py - 50) <= 2


# ---------------------------------------------------------------------------
# to_screen_point()
# ---------------------------------------------------------------------------

def test_to_screen_point_before_fit_raises():
    with pytest.raises(RuntimeError, match='fit'):
        GazeCalibrator().to_screen_point(0.5, 0.5, 640, 480)


def test_to_screen_point_within_bounds():
    clf = GazeCalibrator().fit(_perfect_samples())
    for rh, rv in [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5)]:
        px, py = clf.to_screen_point(rh, rv, 640, 480)
        assert 0 <= px <= 640
        assert 0 <= py <= 480


def test_to_screen_point_extreme_clipped():
    """Predictions outside [0,1] must be clipped to screen bounds."""
    clf = GazeCalibrator().fit(_perfect_samples())
    # Extrapolate well beyond training range — must not go negative
    px, py = clf.to_screen_point(-2.0, -2.0, 640, 480)
    assert px >= 0 and py >= 0


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

def test_save_creates_json(tmp_path):
    path = tmp_path / 'calib.json'
    GazeCalibrator().fit(_perfect_samples()).save(path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert 'coef_x' in data and 'coef_y' in data


def test_save_load_round_trip(tmp_path):
    path = tmp_path / 'calib.json'
    original = GazeCalibrator().fit(_perfect_samples())
    original.save(path)

    loaded = GazeCalibrator.load(path)
    assert loaded._fitted

    for rh, rv in [(0.5, 0.5), (0.15, 0.15), (0.85, 0.85)]:
        assert original.to_screen_point(rh, rv, 640, 480) == \
               loaded.to_screen_point(rh, rv, 640, 480)


def test_save_before_fit_raises(tmp_path):
    with pytest.raises(RuntimeError):
        GazeCalibrator().save(tmp_path / 'calib.json')
