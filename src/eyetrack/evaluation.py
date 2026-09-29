"""
Evaluate the attention-zone classifier on labelled real-video frames.

Splits are always by subject (leave-one-subject-out): frames from one
person are nearly identical, so a frame-level split would leak.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .gaze_classifier import FEATURES, ZONES, GazeZoneClassifier, generate_training_data

NO_PREDICTION = "away"   # system view: no face (or no pose) means the user looked away

ModelFactory = Callable[[], GazeZoneClassifier]


@dataclass
class Scores:
    accuracy: float
    macro_f1: float
    per_class: pd.DataFrame      # precision / recall / f1 / support, indexed by zone
    confusion: pd.DataFrame      # rows = true, columns = predicted
    n: int


def score(y_true, y_pred) -> Scores:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(ZONES), zero_division=0)
    per_class = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s},
                             index=list(ZONES))
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=list(ZONES)),
                      index=list(ZONES), columns=list(ZONES))
    present = [z for z in ZONES if (y_true == z).any()]
    return Scores(
        accuracy=float((y_true == y_pred).mean()),
        macro_f1=float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        per_class=per_class, confusion=cm, n=len(y_true),
    )


def usable(df: pd.DataFrame) -> pd.Series:
    """Rows the classifier can run on: face detected and every feature present."""
    return df["face_detected"].astype(bool) & df[list(FEATURES)].notna().all(axis=1)


def _predict(model: GazeZoneClassifier, df: pd.DataFrame) -> np.ndarray:
    X = df[list(FEATURES)].to_numpy(dtype=np.float32)
    return np.array([model.predict(x) for x in X]) if len(X) else np.array([], dtype=str)


def synthetic_model(n_per_class: int = 600, seed: int = 42) -> GazeZoneClassifier:
    return GazeZoneClassifier(n_estimators=100, random_state=seed).train(
        *generate_training_data(n_per_class=n_per_class, seed=seed))


def predict_fixed(model: GazeZoneClassifier, df: pd.DataFrame) -> pd.Series:
    """Predictions from one model for every row; unusable rows get NO_PREDICTION."""
    out = pd.Series(NO_PREDICTION, index=df.index, dtype=object)
    ok = usable(df)
    out[ok] = _predict(model, df[ok])
    return out


def mirror(X: np.ndarray) -> np.ndarray:
    """
    Left-right mirror of feature rows (FEATURES order). Zones are symmetric
    (20° left of the screen is as peripheral as 20° right), so mirrored
    copies let a subject looking one way inform one looking the other.
    """
    m = X.copy()
    col = {f: i for i, f in enumerate(FEATURES)}
    m[:, col["gaze_ratio_h"]] = 1.0 - m[:, col["gaze_ratio_h"]]
    m[:, col["yaw"]] = -m[:, col["yaw"]]
    m[:, col["dir_h"]] = -m[:, col["dir_h"]]
    return m


def predict_loso(df: pd.DataFrame, with_synthetic: bool = False,
                 with_mirror: bool = False, seed: int = 42) -> pd.Series:
    """
    Leave-one-subject-out: each subject is predicted by a model trained on
    every other subject's usable frames, optionally plus their left-right
    mirror images and/or synthetic data. Test frames are never augmented.
    """
    out = pd.Series(NO_PREDICTION, index=df.index, dtype=object)
    ok = usable(df)
    X_syn, y_syn = generate_training_data(n_per_class=600, seed=seed) if with_synthetic else (None, None)
    for subject in df["subject"].unique():
        test = ok & (df["subject"] == subject)
        train = ok & (df["subject"] != subject)
        if not test.any():
            continue
        X = df.loc[train, list(FEATURES)].to_numpy(dtype=np.float32)
        y = df.loc[train, "label"].to_numpy()
        if with_mirror:
            X, y = np.vstack([X, mirror(X)]), np.concatenate([y, y])
        if with_synthetic:
            X, y = np.vstack([X, X_syn]), np.concatenate([y, y_syn])
        model = GazeZoneClassifier(n_estimators=100, random_state=seed).train(X, y)
        out[test] = _predict(model, df[test])
    return out


def train_final(df: pd.DataFrame, with_mirror: bool = True, seed: int = 42) -> GazeZoneClassifier:
    """Model for deployment: every subject's usable frames (+ mirror images)."""
    ok = usable(df)
    X = df.loc[ok, list(FEATURES)].to_numpy(dtype=np.float32)
    y = df.loc[ok, "label"].to_numpy()
    if with_mirror:
        X, y = np.vstack([X, mirror(X)]), np.concatenate([y, y])
    return GazeZoneClassifier(n_estimators=100, random_state=seed).train(X, y)


@dataclass
class Evaluation:
    name: str
    classifier: Scores     # frames with a usable face only
    system: Scores         # every frame; no face -> away
    per_subject: pd.Series  # system accuracy per subject


def evaluate(name: str, df: pd.DataFrame, predictions: pd.Series) -> Evaluation:
    ok = usable(df)
    correct = predictions == df["label"]
    return Evaluation(
        name=name,
        classifier=score(df.loc[ok, "label"], predictions[ok]),
        system=score(df["label"], predictions),
        per_subject=correct.groupby(df["subject"]).mean(),
    )


def run_all(df: pd.DataFrame) -> list[Evaluation]:
    return [
        evaluate("A. synthetic only", df, predict_fixed(synthetic_model(), df)),
        evaluate("B. real (LOSO)", df, predict_loso(df)),
        evaluate("C. real + mirrored (LOSO)", df, predict_loso(df, with_mirror=True)),
        evaluate("D. synthetic + real + mirrored (LOSO)", df,
                 predict_loso(df, with_synthetic=True, with_mirror=True)),
    ]


def _table(frame: pd.DataFrame, fmt: str = "{:.2f}") -> str:
    cols = [""] + [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in frame.iterrows():
        cells = [fmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx), *cells]) + " |")
    return "\n".join(lines)


def render_report(results: list[Evaluation], df: pd.DataFrame) -> str:
    ok = usable(df)
    by_label = df.groupby("label")
    comp = pd.DataFrame({
        "frames": by_label.size().map("{:d}".format),
        "subjects": by_label["subject"].nunique().map("{:d}".format),
        "face detected": by_label["face_detected"].mean().map("{:.0%}".format),
    }).reindex(list(ZONES))

    summary = pd.DataFrame({
        "system macro-F1": [r.system.macro_f1 for r in results],
        "system accuracy": [r.system.accuracy for r in results],
        "classifier macro-F1": [r.classifier.macro_f1 for r in results],
        **{f"{z} recall": [r.system.per_class.loc[z, "recall"] for r in results] for z in ZONES},
    }, index=[r.name for r in results])

    out = [
        "# Attention-zone evaluation on real video",
        "",
        f"{len(df)} frames from {df['subject'].nunique()} subjects "
        f"({ok.sum()} with a usable face). Generated by `scripts/evaluate_zones.py` "
        "from `datasets/real_clips/manifest.csv`; see that folder's README for "
        "label definitions and limitations.",
        "",
        "**System** scores every frame, treating frames with no detected face as `away`. "
        "**Classifier** scores only frames with a usable face. Real-data models use "
        "leave-one-subject-out: each person is predicted by a model that never saw them. "
        "Macro-F1 averages the per-class F1 over classes present, so the large "
        "`on_screen` class can't hide a weak one.",
        "",
        "## Data",
        "",
        _table(comp),
        "",
        "## Summary",
        "",
        _table(summary),
        "",
    ]
    for r in results:
        out += [f"## {r.name}", "",
                "System confusion matrix (rows = true, columns = predicted):", "",
                _table(r.system.confusion.astype(float), "{:.0f}"), "",
                "System accuracy per subject:", "",
                _table(r.per_subject.to_frame("accuracy")), ""]
    return "\n".join(out)
