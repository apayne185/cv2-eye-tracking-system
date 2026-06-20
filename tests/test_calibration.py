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


def test_fit_exactly_three_points():
    """Boundary: exactly 3 points must succeed (guard is < 3, not <= 3)."""
    samples = list(_perfect_samples())[:3]
    clf = GazeCalibrator().fit(samples)
    assert clf.is_fitted


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
    # Centre: ratio (0.5, 0.5) -> screen (0.5, 0.5)
    px, py = clf.to_screen_point(0.5, 0.5, 100, 100)
    assert abs(px - 50) <= 2 and abs(py - 50) <= 2
    # Top-left target (0.15, 0.15) maps to mirrored x: (0.85, 0.15) in norm space
    px2, py2 = clf.to_screen_point(0.15, 0.15, 100, 100)
    assert abs(px2 - 85) <= 2, f"expected ~85 got {px2}"
    assert abs(py2 - 15) <= 2, f"expected ~15 got {py2}"


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


def test_to_screen_point_extreme_clipped_low():
    """Predictions below 0 must be clipped to 0."""
    clf = GazeCalibrator().fit(_perfect_samples())
    px, py = clf.to_screen_point(-2.0, -2.0, 640, 480)
    assert px >= 0 and py >= 0


def test_to_screen_point_extreme_clipped_high():
    """Predictions above 1 must be clipped to screen bounds."""
    clf = GazeCalibrator().fit(_perfect_samples())
    px, py = clf.to_screen_point(3.0, 3.0, 640, 480)
    assert px <= 640 and py <= 480


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


def test_load_predict_works(tmp_path):
    """Loaded model must be able to call predict() without sklearn attribute errors."""
    path = tmp_path / 'calib.json'
    GazeCalibrator().fit(_perfect_samples()).save(path)
    loaded = GazeCalibrator.load(path)
    # This exercises LinearRegression.predict() which requires n_features_in_ on sklearn>=1.0
    px, py = loaded.to_screen_point(0.5, 0.5, 640, 480)
    assert isinstance(px, int) and isinstance(py, int)


def test_is_fitted_property():
    clf = GazeCalibrator()
    assert not clf.is_fitted
    clf.fit(_perfect_samples())
    assert clf.is_fitted
