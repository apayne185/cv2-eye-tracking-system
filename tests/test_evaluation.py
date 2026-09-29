import numpy as np
import pandas as pd
import pytest

from eyetrack.evaluation import (
    evaluate,
    mirror,
    predict_fixed,
    predict_loso,
    render_report,
    score,
    synthetic_model,
    usable,
)
from eyetrack.gaze_classifier import FEATURES


def _frames(subject, label, center, n=20, face=True, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(center, 0.01, size=(n, len(FEATURES)))
    df = pd.DataFrame(X, columns=list(FEATURES))
    df["subject"], df["label"], df["face_detected"], df["t"] = subject, label, face, np.arange(n)
    return df


def test_score_known_values():
    s = score(["on_screen", "on_screen", "away", "away"], ["on_screen", "away", "away", "away"])
    assert s.accuracy == 0.75
    assert s.per_class.loc["away", "recall"] == 1.0
    assert s.per_class.loc["on_screen", "recall"] == 0.5
    assert s.confusion.loc["on_screen", "away"] == 1
    # macro-F1 over classes present only (peripheral absent)
    assert s.macro_f1 == pytest.approx(np.mean([2 / 3, 0.8]))


def test_unusable_rows_default_to_away():
    df = pd.concat([_frames("a", "on_screen", 0.5, n=3),
                    _frames("b", "away", 0.5, n=2, face=False)], ignore_index=True)
    df.loc[0, "yaw"] = np.nan
    assert usable(df).tolist() == [False, True, True, False, False]
    pred = predict_fixed(synthetic_model(n_per_class=50), df)
    assert (pred[~usable(df)] == "away").all()


def test_loso_never_trains_on_held_out_subject():
    # each subject has its own label and feature cluster: with leakage the
    # model would be perfect; without it, a held-out subject's label is unseen
    df = pd.concat([_frames("s1", "on_screen", 0.0), _frames("s2", "peripheral", 5.0),
                    _frames("s3", "away", 10.0)], ignore_index=True)
    pred = predict_loso(df)
    assert (pred != df["label"]).all()


def test_loso_generalises_across_subjects_with_shared_labels():
    df = pd.concat([_frames(f"on{i}", "on_screen", 0.0, seed=i) for i in range(3)]
                   + [_frames(f"away{i}", "away", 10.0, seed=10 + i) for i in range(3)],
                   ignore_index=True)
    assert (predict_loso(df) == df["label"]).all()


def test_report_renders_all_sections():
    df = pd.concat([_frames(f"on{i}", "on_screen", 0.0, seed=i) for i in range(2)]
                   + [_frames(f"away{i}", "away", 10.0, seed=5 + i) for i in range(2)],
                   ignore_index=True)
    results = [evaluate("A. test", df, predict_loso(df))]
    text = render_report(results, df)
    for heading in ("# Attention-zone evaluation", "## Data", "## Summary", "## A. test",
                    "confusion matrix", "per subject"):
        assert heading in text


def test_mirror_flips_horizontal_features_only():
    X = np.array([[0.7, 0.4, 20.0, 0.3, -0.1]], dtype=np.float32)   # FEATURES order
    np.testing.assert_allclose(mirror(X), [[0.3, 0.4, -20.0, -0.3, -0.1]], rtol=1e-6)
    np.testing.assert_allclose(mirror(mirror(X)), X)


def test_mirrored_loso_still_does_not_leak():
    df = pd.concat([_frames("s1", "on_screen", 0.0), _frames("s2", "peripheral", 5.0),
                    _frames("s3", "away", 10.0)], ignore_index=True)
    assert (predict_loso(df, with_mirror=True) != df["label"]).all()


def test_mirroring_lets_one_side_inform_the_other():
    # peripheral subjects in training all look right; the held-out one looks left
    def side(subject, sign, seed):
        d = _frames(subject, "peripheral", 0.0, seed=seed)
        d[list(FEATURES)] = [0.5 + 0.1 * sign, 0.5, -15.0 * sign, 0.35 * sign, 0.0]
        return d
    on = [_frames(f"on{i}", "on_screen", 0.0, seed=i) for i in range(3)]
    for d in on:
        d[list(FEATURES)] = [0.5, 0.5, 0.0, 0.0, 0.0]
    df = pd.concat(on + [side("r1", +1, 7), side("r2", +1, 8), side("left", -1, 9)],
                   ignore_index=True)
    left = df["subject"] == "left"
    assert (predict_loso(df)[left] != "peripheral").all()
    assert (predict_loso(df, with_mirror=True)[left] == "peripheral").all()
