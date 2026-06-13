import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gaze_classifier import (
    FEATURES,
    ZONES,
    GazeZoneClassifier,
    generate_training_data,
)


# ---------------------------------------------------------------------------
# generate_training_data
# ---------------------------------------------------------------------------

def test_generate_training_data_shape():
    X, y = generate_training_data(n_per_class=30)
    assert X.shape == (90, 5)
    assert y.shape == (90,)


def test_generate_training_data_balanced():
    n = 40
    _, y = generate_training_data(n_per_class=n)
    for zone in ZONES:
        assert (y == zone).sum() == n, f'expected {n} samples for zone {zone!r}'


def test_generate_training_data_dtype():
    X, _ = generate_training_data(n_per_class=10)
    assert X.dtype == np.float32


def test_generate_training_data_reproducible():
    X1, y1 = generate_training_data(seed=7)
    X2, y2 = generate_training_data(seed=7)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_on_screen_features_centred():
    """on_screen samples should have iris ratios close to 0.5 and low yaw."""
    X, y = generate_training_data(n_per_class=200, seed=0)
    mask = y == 'on_screen'
    on = X[mask]
    assert abs(on[:, 0].mean() - 0.5) < 0.05, 'gaze_ratio_h should be ~0.5'
    assert on[:, 2].std() < 20, 'yaw std should be modest for on_screen'


def test_away_features_extreme_yaw():
    """away samples should have high |yaw|."""
    X, y = generate_training_data(n_per_class=200, seed=0)
    mask = y == 'away'
    away = X[mask]
    assert np.abs(away[:, 2]).mean() > 40, 'away samples should have large mean |yaw|'


# ---------------------------------------------------------------------------
# GazeZoneClassifier
# ---------------------------------------------------------------------------

@pytest.fixture()
def trained_clf():
    X, y = generate_training_data(n_per_class=200, seed=42)
    return GazeZoneClassifier().train(X, y)


def test_untrained_raises():
    clf = GazeZoneClassifier()
    with pytest.raises(RuntimeError, match='train'):
        clf.predict(np.zeros(5))


def test_predict_returns_valid_zone(trained_clf):
    result = trained_clf.predict(np.array([0.5, 0.45, 0.0, 0.0, 0.0]))
    assert result in ZONES


def test_predict_dict_interface(trained_clf):
    features = {
        'gaze_ratio_h': 0.5, 'gaze_ratio_v': 0.45,
        'yaw': 0.0, 'dir_h': 0.0, 'dir_v': 0.0,
    }
    result = trained_clf.predict(features)
    assert result in ZONES


def test_predict_on_screen(trained_clf):
    """Centred gaze + zero head rotation → on_screen."""
    result = trained_clf.predict({'gaze_ratio_h': 0.50, 'gaze_ratio_v': 0.45,
                                  'yaw': 0.0, 'dir_h': 0.0, 'dir_v': 0.0})
    assert result == 'on_screen'


def test_predict_away(trained_clf):
    """Extreme lateral gaze + high yaw → away."""
    result = trained_clf.predict({'gaze_ratio_h': 0.05, 'gaze_ratio_v': 0.5,
                                  'yaw': 55.0, 'dir_h': -0.9, 'dir_v': 0.0})
    assert result == 'away'


def test_predict_proba_sums_to_one(trained_clf):
    proba = trained_clf.predict_proba(np.zeros(5))
    total = sum(proba.values())
    assert abs(total - 1.0) < 1e-5


def test_predict_proba_keys(trained_clf):
    proba = trained_clf.predict_proba(np.zeros(5))
    assert set(proba.keys()) == set(ZONES)


def test_accuracy_above_threshold(trained_clf):
    """Classifier should reach >90% accuracy on held-out synthetic data."""
    X_test, y_test = generate_training_data(n_per_class=150, seed=99)
    result = trained_clf.evaluate(X_test, y_test)
    acc = result['report']['accuracy']
    assert acc > 0.90, f'accuracy {acc:.3f} below 0.90 threshold'


def test_confusion_matrix_shape(trained_clf):
    X, y = generate_training_data(n_per_class=50, seed=1)
    result = trained_clf.evaluate(X, y)
    cm = result['confusion_matrix']
    assert cm.shape == (3, 3)


def test_feature_importances_sum(trained_clf):
    imps = trained_clf.feature_importances()
    assert abs(sum(imps.values()) - 1.0) < 1e-5


def test_feature_importances_keys(trained_clf):
    imps = trained_clf.feature_importances()
    assert set(imps.keys()) == set(FEATURES)


def test_save_and_load(tmp_path, trained_clf):
    path = tmp_path / 'model.joblib'
    saved = trained_clf.save(path)
    assert saved.exists()

    loaded = GazeZoneClassifier.load(path)
    # Loaded model should reproduce the same predictions
    x = np.array([0.5, 0.45, 0.0, 0.0, 0.0])
    assert loaded.predict(x) == trained_clf.predict(x)
