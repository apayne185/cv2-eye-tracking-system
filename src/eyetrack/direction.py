import cv2
import numpy as np

# Average of the two eye-corner landmarks in the 6-point face model (mm).
# Used as the 3D origin of the gaze ray.
_EYE_MODEL_MIDPOINT = np.array([0.0, 170.0, -135.0])


class GazeDirectionEstimator:
    """
    Fuses iris gaze ratios with head-pose angles to estimate gaze direction,
    and computes a fully 3D gaze ray in camera coordinates.

    2-D estimate (dir_h, dir_v):
      Iris ratios alone track where the eye points within the socket; head pose
      captures where the head is oriented. Combining both gives a direction that
      is stable under head rotation.  Output is in [-1, 1].

    3-D gaze ray:
      Uses the rotation matrix from solvePnP to transform the iris-derived
      gaze direction from head/model space into camera space, producing a
      physical ray (origin + unit direction) expressed in mm.
    """

    _EYE_SCALE  = 1.4    # iris deviation from centre (±0.5) → 2-D direction
    _HEAD_SCALE = 0.014  # degrees of yaw/pitch → 2-D direction

    # Approximate angular range of voluntary iris movement.
    # Full ratio range [0, 1] maps to ±half of these values.
    _EYE_FOV_H = 60.0   # degrees horizontal
    _EYE_FOV_V = 40.0   # degrees vertical

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

    def gaze_ray_3d(
        self,
        ratio_h: float,
        ratio_v: float,
        rotation_matrix: np.ndarray,
        translation_vector: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes a 3D gaze ray in camera coordinates.

        Converts iris ratios to azimuth/elevation angles in head space, rotates
        the resulting direction vector into camera space via the head-pose
        rotation matrix, and translates the ray origin (eye midpoint) similarly.

        Args:
            ratio_h, ratio_v   : iris position in [0, 1], centre = 0.5
            rotation_matrix    : 3×3 rotation matrix R from HeadPoseEstimator
            translation_vector : (3,1) translation t from HeadPoseEstimator

        Returns:
            origin    : np.ndarray (3,) — eye midpoint in camera coords (mm)
            direction : np.ndarray (3,) — unit gaze vector in camera coords
        """
        R = rotation_matrix
        t = translation_vector.ravel()

        theta_h = np.radians((ratio_h - 0.5) * self._EYE_FOV_H)
        theta_v = np.radians((ratio_v - 0.5) * self._EYE_FOV_V)

        ch, sh = np.cos(theta_h), np.sin(theta_h)
        cv_, sv = np.cos(theta_v), np.sin(theta_v)

        # Rotate forward vector [0,0,1] by iris angles in head space
        Ry = np.array([[ch, 0, sh], [0, 1, 0], [-sh, 0, ch]])
        Rx = np.array([[1, 0, 0],  [0, cv_, -sv], [0, sv, cv_]])
        d_head = Ry @ Rx @ np.array([0.0, 0.0, 1.0])

        origin    = R @ _EYE_MODEL_MIDPOINT + t
        direction = R @ d_head
        direction = direction / (np.linalg.norm(direction) + 1e-9)

        return origin, direction

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
