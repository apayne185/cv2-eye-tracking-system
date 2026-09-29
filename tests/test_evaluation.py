import numpy as np
import pandas as pd
import pytest

from eyetrack.evaluation import (
    evaluate,
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
