import cv2
import numpy as np

# Generic 6-point 3D face model (mm).
# Chosen points cover the face geometry needed for stable PnP.
_MODEL_3D = np.array([
    (0.0,    0.0,    0.0),     # nose tip          → landmark 1
    (0.0,  -330.0,  -65.0),   # chin              → landmark 152
    (-225.0, 170.0, -135.0),  # left eye corner   → landmark 33
    (225.0,  170.0, -135.0),  # right eye corner  → landmark 263
    (-150.0,-150.0, -125.0),  # left mouth corner → landmark 61
    (150.0, -150.0, -125.0),  # right mouth corner→ landmark 291
], dtype=np.float64)

_LM_IDS = [1, 152, 33, 263, 61, 291]


class HeadPoseEstimator:
    def __init__(self, frame_w, frame_h):
        # Approximate intrinsics: focal length = frame width, principal point = center
        f  = float(frame_w)
        cx = frame_w / 2.0
        cy = frame_h / 2.0
        self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        self.D = np.zeros((4, 1))

    def _image_points(self, lms, shape):
        h, w = shape[:2]
        return np.array(
            [(lms.landmark[i].x * w, lms.landmark[i].y * h) for i in _LM_IDS],
            dtype=np.float64,
        )

    def estimate(self, lms, shape):
        """Returns (pitch, yaw, roll) in degrees, or (None, None, None) on failure."""
        pts2d = self._image_points(lms, shape)
        ok, rvec, _ = cv2.solvePnP(_MODEL_3D, pts2d, self.K, self.D,
                                    flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None, None

        rmat, _ = cv2.Rodrigues(rvec)
        angles  = cv2.RQDecomp3x3(rmat)[0]       # (pitch, yaw, roll) in degrees
        return float(angles[0]), float(angles[1]), float(angles[2])

    def draw_axes(self, frame, lms):
        """Draws X/Y/Z axes at the nose tip for visual confirmation."""
        pts2d = self._image_points(lms, frame.shape)
        ok, rvec, tvec = cv2.solvePnP(_MODEL_3D, pts2d, self.K, self.D,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return

        axis = np.float32([[80, 0, 0], [0, 80, 0], [0, 0, 80]])
        projected, _ = cv2.projectPoints(axis, rvec, tvec, self.K, self.D)
        projected = projected.astype(int)
        nose = pts2d[0].astype(int)

        cv2.line(frame, tuple(nose), tuple(projected[0].ravel()), (0,   0, 255), 2)  # X red
        cv2.line(frame, tuple(nose), tuple(projected[1].ravel()), (0, 255,   0), 2)  # Y green
        cv2.line(frame, tuple(nose), tuple(projected[2].ravel()), (255,  0,   0), 2)  # Z blue
