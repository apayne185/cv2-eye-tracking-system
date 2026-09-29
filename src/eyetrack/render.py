"""Debug overlays for a processed frame (display only; never affects results)."""

import cv2
import numpy as np

from .direction import GazeDirectionEstimator
from .pipeline import FrameProcessor, FrameResult

# Colour coding for attention zone overlay
_ZONE_COLORS = {
    'on_screen':  (0, 200,   0),   # green
    'peripheral': (0, 165, 255),   # orange
    'away':       (0,   0, 255),   # red
}


def gaze_label(ratio_h: float, ratio_v: float) -> str:
    h = "LEFT" if ratio_h < 0.40 else ("RIGHT" if ratio_h > 0.60 else "CENTER")
    v = "UP"   if ratio_v < 0.35 else ("DOWN"  if ratio_v > 0.65 else "")
    return f"{v} {h}".strip() if v else h


def draw_result(frame: np.ndarray, res: FrameResult, proc: FrameProcessor) -> None:
    """Draws eye outlines, head-pose axes, gaze ray, AOIs, and labels in place."""
    if not res.face_detected:
        return

    if res.dir_h is not None:
        GazeDirectionEstimator.draw_direction_marker(frame, res.dir_h, res.dir_v)
    if res.screen_point is not None:
        cv2.drawMarker(frame, res.screen_point, (255, 100, 0), cv2.MARKER_CROSS, 20, 2)
    if res.ray_origin is not None:
        proc.pose_est.draw_gaze_ray(frame, res.ray_origin, res.ray_direction)

    proc.aoi.draw(frame, res.active_aoi)
    proc.tracker.draw_overlays(frame, res.landmarks)
    proc.pose_est.draw_axes(frame)

    label = "BLINK" if res.is_blink else gaze_label(res.gaze_ratio_h, res.gaze_ratio_v)
    color = (0, 0, 255) if res.is_blink else (0, 255, 0)
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    if res.pitch is not None:
        cv2.putText(
            frame, f"P:{res.pitch:.1f}  Y:{res.yaw:.1f}  R:{res.roll:.1f}",
            (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1,
        )
    if res.predicted_zone is not None:
        cv2.putText(
            frame, f"Zone: {res.predicted_zone}",
            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            _ZONE_COLORS.get(res.predicted_zone, (255, 255, 255)), 2,
        )
