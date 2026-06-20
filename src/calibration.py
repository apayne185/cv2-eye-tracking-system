"""
5-point gaze calibration.

Displays fixation targets at known screen positions, collects iris ratio
samples while the user fixates each one, then fits a LinearRegression
mapping (ratio_h, ratio_v) → normalised screen coordinates.

The fitted model replaces the heuristic screen-point formula in direction.py
with a person-specific mapping, improving absolute gaze accuracy.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

DEFAULT_CALIB_PATH = Path(__file__).parent.parent / 'models' / 'calibration.json'

# Normalised (x, y) positions of the 5 fixation targets
CALIB_POINTS_NORM: list[tuple[float, float]] = [
    (0.50, 0.50),   # centre
    (0.15, 0.15),   # top-left
    (0.85, 0.15),   # top-right
    (0.15, 0.85),   # bottom-left
    (0.85, 0.85),   # bottom-right
]

_WAIT_S     = 1.5   # seconds to look before collection starts
_N_COLLECT  = 40    # frames to average per point
_DOT_RADIUS = 18
_DOT_RING   = 28


def _draw_target(frame_w: int, frame_h: int,
                 px: int, py: int,
                 phase: str, progress: int = 0, total: int = _N_COLLECT) -> np.ndarray:
    canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    # Faint crosshair guides
    cv2.line(canvas, (px, 0), (px, frame_h), (40, 40, 40), 1)
    cv2.line(canvas, (0, py), (frame_w, py), (40, 40, 40), 1)

    # Outer ring + filled dot
    cv2.circle(canvas, (px, py), _DOT_RING,  (180, 180, 180), 1)
    color = (0, 255, 120) if phase == 'collecting' else (255, 255, 255)
    cv2.circle(canvas, (px, py), _DOT_RADIUS, color, -1)

    # Progress arc during collection
    if phase == 'collecting' and total > 0:
        angle = int(360 * progress / total)
        cv2.ellipse(canvas, (px, py), (_DOT_RING, _DOT_RING),
                    -90, 0, angle, (0, 200, 255), 2)

    # Status text
    if phase == 'look':
        msg = "Look at the dot and hold still..."
    else:
        msg = f"Collecting  {progress}/{total}"

    (tw, _th), _baseline = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    cv2.putText(canvas, msg,
                (frame_w // 2 - tw // 2, frame_h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    return canvas


class GazeCalibrator:
    """
    Fits a linear mapping from iris ratios to screen coordinates.

    Usage (interactive):
        calib = GazeCalibrator()
        calib.run(cap, tracker, frame_w, frame_h)
        calib.save()

    Usage (inference):
        calib = GazeCalibrator.load()
        sx, sy = calib.to_screen_point(ratio_h, ratio_v, screen_w, screen_h)
    """

    def __init__(self) -> None:
        self._reg_x: LinearRegression | None = None
        self._reg_y: LinearRegression | None = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ------------------------------------------------------------------
    # Interactive collection
    # ------------------------------------------------------------------

    def run(self, cap, tracker, frame_w: int, frame_h: int,
            wait_s: float = _WAIT_S, n_collect: int = _N_COLLECT) -> 'GazeCalibrator':
        """
        Display 5 fixation targets in sequence, collect iris ratios,
        fit the linear model. Returns self for chaining.
        """
        collected: list[tuple[float, float, float, float]] = []

        cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

        for point_idx, (nx, ny) in enumerate(CALIB_POINTS_NORM):
            px = int(nx * frame_w)
            py = int(ny * frame_h)

            # ── countdown phase ──────────────────────────────────────
            t_start = time.time()
            while time.time() - t_start < wait_s:
                ret, _ = cap.read()
                if not ret:
                    break
                canvas = _draw_target(frame_w, frame_h, px, py, 'look')
                cv2.imshow('Calibration', canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyWindow('Calibration')
                    return self

            # ── collection phase ─────────────────────────────────────
            ratios_h: list[float] = []
            ratios_v: list[float] = []
            t_collect_start = time.time()
            _COLLECT_TIMEOUT = max(10.0, n_collect / 5.0)  # give up after N secs

            while len(ratios_h) < n_collect:
                if time.time() - t_collect_start > _COLLECT_TIMEOUT:
                    print(f"  Point {point_idx + 1}: timed out waiting for face — skipping.")
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                results = tracker.process(frame)
                if results.multi_face_landmarks:
                    lms = results.multi_face_landmarks[0]
                    _, _, rh, rv = tracker.get_iris_gaze(lms, frame.shape)
                    ratios_h.append(rh)
                    ratios_v.append(rv)

                canvas = _draw_target(frame_w, frame_h, px, py,
                                      'collecting', len(ratios_h), n_collect)
                cv2.imshow('Calibration', canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyWindow('Calibration')
                    return self

            if len(ratios_h) >= 5:
                collected.append((
                    float(np.median(ratios_h)),
                    float(np.median(ratios_v)),
                    nx, ny,
                ))
                print(f"  Point {point_idx + 1}/5  "
                      f"rh={collected[-1][0]:.3f}  rv={collected[-1][1]:.3f}  "
                      f"target=({nx:.2f},{ny:.2f})")

        cv2.destroyWindow('Calibration')

        if len(collected) >= 3:
            self.fit(collected)
            print(f"Calibration complete — {len(collected)} points fitted.")
        else:
            print("Calibration failed: fewer than 3 points collected.")

        return self

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, samples: list[tuple[float, float, float, float]]) -> 'GazeCalibrator':
        """
        Fit two independent linear regressors from collected samples.
        samples: list of (ratio_h, ratio_v, target_nx, target_ny)
        """
        if len(samples) < 3:
            raise ValueError(f'Need at least 3 calibration points, got {len(samples)}.')

        arr = np.array(samples, dtype=np.float64)
        X  = arr[:, :2]   # (ratio_h, ratio_v)
        nx = arr[:, 2]    # target normalised x
        ny = arr[:, 3]    # target normalised y

        self._reg_x = LinearRegression().fit(X, nx)
        self._reg_y = LinearRegression().fit(X, ny)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def to_screen_point(self, ratio_h: float, ratio_v: float,
                        screen_w: int, screen_h: int) -> tuple[int, int]:
        """Map iris ratios to pixel screen coordinates using the fitted model."""
        self._check_fitted()
        feat = np.array([[ratio_h, ratio_v]], dtype=np.float64)
        nx = float(np.clip(self._reg_x.predict(feat)[0], 0.0, 1.0))
        ny = float(np.clip(self._reg_y.predict(feat)[0], 0.0, 1.0))
        return int(nx * screen_w), int(ny * screen_h)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: 'str | Path' = DEFAULT_CALIB_PATH) -> Path:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'coef_x':      self._reg_x.coef_.tolist(),
            'intercept_x': float(self._reg_x.intercept_),
            'coef_y':      self._reg_y.coef_.tolist(),
            'intercept_y': float(self._reg_y.intercept_),
        }
        path.write_text(json.dumps(data, indent=2))
        return path

    @classmethod
    def load(cls, path: 'str | Path' = DEFAULT_CALIB_PATH) -> 'GazeCalibrator':
        path = Path(path)
        data = json.loads(path.read_text())
        obj = cls()
        for attr, coef_key, intercept_key in [
            ('_reg_x', 'coef_x', 'intercept_x'),
            ('_reg_y', 'coef_y', 'intercept_y'),
        ]:
            reg = LinearRegression()
            reg.coef_          = np.array(data[coef_key], dtype=np.float64)
            reg.intercept_     = np.float64(data[intercept_key])
            reg.n_features_in_ = reg.coef_.shape[0]  # required by sklearn >= 1.0
            setattr(obj, attr, reg)
        obj._fitted = True
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError('Run calibration or call fit() first.')
