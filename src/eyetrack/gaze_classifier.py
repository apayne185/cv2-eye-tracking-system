"""
ML-based gaze attention zone classifier.

Classifies each frame into one of three zones:
  on_screen  – gaze directed at the display in front of the subject
  peripheral – gaze near the screen edge or slightly off-centre
  away       – gaze directed clearly away from the screen

Features: gaze_ratio_h, gaze_ratio_v, yaw, dir_h, dir_v
Model: Random Forest with StandardScaler preprocessing
"""

from pathlib import Path

import numpy as np

from .direction import fuse_direction

ZONES    = ('on_screen', 'peripheral', 'away')
FEATURES = ('gaze_ratio_h', 'gaze_ratio_v', 'yaw', 'dir_h', 'dir_v')

DEFAULT_MODEL_PATH = Path('models') / 'gaze_zone_classifier.joblib'


def generate_training_data(n_per_class: int = 600,
                           seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates balanced synthetic training data for all three zones.

    Feature distributions are grounded in gaze-research norms:
      - on_screen  : iris near centre (ratio ~0.5), small head rotation
      - peripheral : iris displaced to one side OR moderate head turn
      - away       : large iris deviation AND/OR high yaw magnitude

    Returns X (N, 5) float32 and y (N,) string label array.
    """
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[str] = []

    n = n_per_class

    # --- on_screen ---
    rh    = rng.normal(0.50, 0.06, n).clip(0.32, 0.68)
    rv    = rng.normal(0.44, 0.06, n).clip(0.28, 0.62)
    yaw   = rng.normal(0,  8, n).clip(-22,  22)
    pitch = rng.normal(5,  5, n).clip(-12,  18)
    dh, dv = fuse_direction(rh, rv, yaw, pitch)
    rows.append(np.stack([rh, rv, yaw, dh, dv], axis=1))
    labels.extend(['on_screen'] * n)

    # --- peripheral ---
    # Half the samples have the iris displaced (any head angle), half have
    # the iris near centre with a moderate head turn; without the second
    # group a turned head with centred eyes fell outside every class.
    # Explicit full-length halves avoid the n//2+n//2 = n-1 trap on odd n
    half = n // 2
    n_eyes = n - half
    eyes_off = np.concatenate([rng.uniform(0.25, 0.38, n_eyes // 2),
                               rng.uniform(0.62, 0.75, n_eyes - n_eyes // 2)])
    rh = np.concatenate([eyes_off, rng.normal(0.50, 0.06, half).clip(0.32, 0.68)])
    head_turn = rng.uniform(25, 45, half) * rng.choice([-1, 1], half)
    yaw = np.concatenate([rng.uniform(-45, 45, n_eyes), head_turn])
    order = rng.permutation(n)
    rh, yaw = rh[order], yaw[order]
    rv    = rng.normal(0.46, 0.09, n).clip(0.20, 0.76)
    pitch = rng.normal(0, 10, n).clip(-25, 25)
    dh, dv = fuse_direction(rh, rv, yaw, pitch)
    rows.append(np.stack([rh, rv, yaw, dh, dv], axis=1))
    labels.extend(['peripheral'] * n)

    # --- away ---
    rh = np.concatenate([rng.uniform(0.05, 0.24, half),
                         rng.uniform(0.76, 0.95, n - half)])
    rng.shuffle(rh)
    rv    = rng.uniform(0.05, 0.95, n)
    yaw_m = rng.uniform(42, 62, n)
    yaw   = yaw_m * rng.choice([-1, 1], n)
    pitch = rng.uniform(-30, 30, n)
    dh, dv = fuse_direction(rh, rv, yaw, pitch)
    rows.append(np.stack([rh, rv, yaw, dh, dv], axis=1))
    labels.extend(['away'] * n)

    X = np.vstack(rows).astype(np.float32)
    y = np.array(labels)
    # Shuffle so classes are not contiguous
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


class GazeZoneClassifier:
    """
    Random Forest classifier for gaze attention zones.

    Wraps a sklearn Pipeline (StandardScaler + RandomForestClassifier)
    with convenience methods for training, inference, evaluation, and
    joblib-based persistence.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self._model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                class_weight='balanced',
            )),
        ])
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray) -> 'GazeZoneClassifier':
        self._model.fit(X, y)
        self._trained = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: 'dict | np.ndarray') -> str:
        """
        Predict zone from a feature dict or a (5,) array.
        Feature order / keys: gaze_ratio_h, gaze_ratio_v, yaw, dir_h, dir_v.
        Missing keys default to 0.0 (safe neutral value).
        """
        self._check_trained()
        row = self._to_row(features)
        return self._model.predict(row)[0]

    def predict_proba(self, features: 'dict | np.ndarray') -> 'dict[str, float]':
        """Returns class → probability mapping for a single sample."""
        self._check_trained()
        row = self._to_row(features)
        probs = self._model.predict_proba(row)[0]
        return dict(zip(self._model.classes_, probs.tolist(), strict=True))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Returns classification report dict and confusion matrix."""
        from sklearn.metrics import classification_report, confusion_matrix
        self._check_trained()
        y_pred = self._model.predict(X)
        return {
            'report':           classification_report(y, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y, y_pred, labels=list(ZONES)),
            'classes':          list(ZONES),
        }

    def feature_importances(self) -> 'dict[str, float]':
        self._check_trained()
        imps = self._model.named_steps['clf'].feature_importances_
        return dict(zip(FEATURES, imps.tolist(), strict=True))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: 'str | Path' = DEFAULT_MODEL_PATH) -> Path:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)
        return path

    @classmethod
    def load(cls, path: 'str | Path' = DEFAULT_MODEL_PATH) -> 'GazeZoneClassifier':
        import joblib
        obj = cls.__new__(cls)
        obj._model   = joblib.load(path)
        obj._trained = True
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_trained(self) -> None:
        if not self._trained:
            raise RuntimeError('Call train() before using the classifier.')

    def _to_row(self, features: 'dict | np.ndarray') -> np.ndarray:
        if isinstance(features, dict):
            row = np.array([features.get(k, 0.0) for k in FEATURES],
                           dtype=np.float32)
        else:
            row = np.asarray(features, dtype=np.float32)
        return row.reshape(1, -1)
