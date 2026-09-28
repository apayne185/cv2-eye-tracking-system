"""
Per-frame processing, independent of any display or output.

FrameProcessor turns one BGR frame into a FrameResult. It keeps the
stateful estimators (fixation tracking, head pose, AOI dwell) across
frames but never draws, prints, or writes files, so the same pipeline
backs the desktop CLI, headless runs, and diagnostics.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .aoi import AOITracker
from .direction import GazeDirectionEstimator
from .eye_tracker import EyeTracker
from .head_pose import HeadPoseEstimator

# CSV column order; matches the schema documented in the README.
CSV_COLUMNS = (
    "frame", "timestamp",
    "gaze_x", "gaze_y", "gaze_ratio_h", "gaze_ratio_v",
    "pitch", "yaw", "roll",
    "dir_h", "dir_v",
    "ray_ox", "ray_oy", "ray_oz",
    "ray_dx", "ray_dy", "ray_dz",
    "left_ear", "right_ear",
    "is_blink", "is_fixation", "active_aoi",
    "predicted_zone",
)


@dataclass
class FrameResult:
    frame: int
    timestamp: float
    face_detected: bool = False

    gaze_x: int | None = None
    gaze_y: int | None = None
    gaze_ratio_h: float | None = None
    gaze_ratio_v: float | None = None

    pitch: float | None = None
    yaw: float | None = None
    roll: float | None = None

    dir_h: float | None = None
    dir_v: float | None = None

    ray_origin: np.ndarray | None = field(default=None, repr=False)
    ray_direction: np.ndarray | None = field(default=None, repr=False)

    left_ear: float | None = None
    right_ear: float | None = None
    is_blink: bool = False
    is_fixation: bool = False
    active_aoi: str | None = None
    predicted_zone: str | None = None

    screen_point: tuple[int, int] | None = None   # calibrated, in frame pixels
    landmarks: Any = field(default=None, repr=False)

    def to_row(self) -> dict:
        """Flat, rounded record for CSV export and telemetry."""
        def r(v, nd):
            return None if v is None else round(float(v), nd)

        o = self.ray_origin
        d = self.ray_direction
        row = dict(
            frame=self.frame, timestamp=round(self.timestamp, 4),
            gaze_x=self.gaze_x, gaze_y=self.gaze_y,
            gaze_ratio_h=r(self.gaze_ratio_h, 3), gaze_ratio_v=r(self.gaze_ratio_v, 3),
            pitch=r(self.pitch, 1), yaw=r(self.yaw, 1), roll=r(self.roll, 1),
            dir_h=r(self.dir_h, 3), dir_v=r(self.dir_v, 3),
            ray_ox=None if o is None else r(o[0], 1),
            ray_oy=None if o is None else r(o[1], 1),
            ray_oz=None if o is None else r(o[2], 1),
            ray_dx=None if d is None else r(d[0], 3),
            ray_dy=None if d is None else r(d[1], 3),
            ray_dz=None if d is None else r(d[2], 3),
            left_ear=r(self.left_ear, 3), right_ear=r(self.right_ear, 3),
            is_blink=self.is_blink, is_fixation=self.is_fixation,
            active_aoi=self.active_aoi, predicted_zone=self.predicted_zone,
        )
        return {k: row[k] for k in CSV_COLUMNS}


class FrameProcessor:
    def __init__(self, frame_w: int, frame_h: int, *,
                 calibrator=None, zone_classifier=None, aois=None,
                 tracker: EyeTracker | None = None):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.tracker    = tracker or EyeTracker()
        self.pose_est   = HeadPoseEstimator(frame_w, frame_h)
        self.dir_est    = GazeDirectionEstimator()
        self.aoi        = AOITracker(aois)
        self.calibrator = calibrator
        self.zone_clf   = zone_classifier
        self._frame_idx = 0

    def process(self, frame: np.ndarray, ts: float) -> FrameResult:
        res = FrameResult(frame=self._frame_idx, timestamp=ts)
        self._frame_idx += 1

        mesh = self.tracker.process(frame)
        if not mesh.multi_face_landmarks:
            return res
        self._process_face(res, mesh.multi_face_landmarks[0], frame.shape, ts)
        return res

    def _process_face(self, res: FrameResult, lms, shape, ts: float) -> None:
        res.face_detected = True
        res.landmarks = lms

        gx, gy, rh, rv = self.tracker.get_iris_gaze(lms, shape)
        res.gaze_x, res.gaze_y = gx, gy
        res.gaze_ratio_h, res.gaze_ratio_v = rh, rv

        res.is_fixation = self.tracker.update_fixation((gx, gy), ts)
        res.is_blink, res.left_ear, res.right_ear = self.tracker.detect_blink(lms, shape)
        res.active_aoi = self.aoi.update((gx, gy), ts)

        pitch, yaw, roll = self.pose_est.estimate(lms, shape)
        if pitch is None:
            return
        res.pitch, res.yaw, res.roll = pitch, yaw, roll

        dh, dv = self.dir_est.estimate(rh, rv, yaw, pitch)
        res.dir_h, res.dir_v = dh, dv

        if self.calibrator is not None:
            res.screen_point = self.calibrator.to_screen_point(
                rh, rv, self.frame_w, self.frame_h)

        if self.zone_clf is not None:
            res.predicted_zone = self.zone_clf.predict({
                'gaze_ratio_h': rh, 'gaze_ratio_v': rv,
                'yaw': yaw, 'dir_h': dh, 'dir_v': dv,
            })

        R = self.pose_est.rotation_matrix
        if R is not None:
            res.ray_origin, res.ray_direction = self.dir_est.gaze_ray_3d(
                rh, rv, R, self.pose_est.translation_vector)
