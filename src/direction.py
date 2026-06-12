import cv2
import numpy as np


class GazeDirectionEstimator:
    """
    Fuses iris gaze ratios with head-pose angles to estimate gaze direction.

    Iris ratios alone track where the eye points within the socket; head pose
    captures where the head is oriented. Combining both gives a direction that
    remains stable when the subject turns their head while keeping their eyes
    still, and correctly reflects eye movement independent of head rotation.

    Output (dir_h, dir_v) is in [-1, 1]:
      -1 = full left  / full up
       0 = straight ahead
      +1 = full right / full down
    """

    _EYE_SCALE  = 1.4    # iris deviation from centre (±0.5) → direction
    _HEAD_SCALE = 0.014  # degrees of yaw/pitch → direction

    def estimate(self, ratio_h: float, ratio_v: float,
                 yaw: float, pitch: float) -> tuple[float, float]:
        """
        ratio_h, ratio_v : iris position in [0, 1], centre = 0.5
        yaw              : head yaw in degrees  (+ve = turned right)
        pitch            : head pitch in degrees (+ve = tilted down in OpenCV convention)
        Returns (dir_h, dir_v) clamped to [-1, 1].
        """
        iris_h = (ratio_h - 0.5) * self._EYE_SCALE
        iris_v = (ratio_v - 0.5) * self._EYE_SCALE
        dir_h  = float(np.clip(iris_h + yaw   * self._HEAD_SCALE, -1.0, 1.0))
        dir_v  = float(np.clip(iris_v - pitch * self._HEAD_SCALE, -1.0, 1.0))
        return dir_h, dir_v

    def to_screen_point(self, dir_h: float, dir_v: float,
                        screen_w: int, screen_h: int) -> tuple[int, int]:
        """Maps direction [-1, 1] to pixel coordinates on a notional screen."""
        x = int(np.clip((dir_h + 1) / 2 * screen_w, 0, screen_w - 1))
        y = int(np.clip((dir_v + 1) / 2 * screen_h, 0, screen_h - 1))
        return x, y

    @staticmethod
    def draw_direction_marker(frame: np.ndarray,
                              dir_h: float, dir_v: float,
                              size: int = 80, margin: int = 10) -> None:
        """
        Draws a miniature gaze-direction indicator in the top-right corner.
        The orange dot's position within the box represents gaze direction;
        the centre crosshair marks straight-ahead.
        """
        h, w = frame.shape[:2]
        x0, y0 = w - size - margin, margin
        x1, y1 = x0 + size, y0 + size

        cv2.rectangle(frame, (x0, y0), (x1, y1), (30, 30, 30), -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (90, 90, 90), 1)

        mid_x, mid_y = (x0 + x1) // 2, (y0 + y1) // 2
        cv2.line(frame, (mid_x - 6, mid_y), (mid_x + 6, mid_y), (70, 70, 70), 1)
        cv2.line(frame, (mid_x, mid_y - 6), (mid_x, mid_y + 6), (70, 70, 70), 1)

        cx = int(np.clip(x0 + (dir_h + 1) / 2 * size, x0, x1))
        cy = int(np.clip(y0 + (dir_v + 1) / 2 * size, y0, y1))
        cv2.circle(frame, (cx, cy), 6, (0, 165, 255), -1)

        cv2.putText(frame, "DIR", (x0, y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)
