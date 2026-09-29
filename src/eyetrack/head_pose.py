from importlib.resources import files

import cv2
import numpy as np


# MediaPipe's canonical face model: one 3D vertex per FaceMesh landmark
# (468), in cm, y-up, face toward +z. Solving PnP against all of them is
# far better conditioned than a handful of nearly coplanar points, which
# let yaw flip sign on near-frontal faces. Stored in mm to match the gaze
# ray. Source and license: src/eyetrack/assets/README.md.
def _load_canonical_model() -> np.ndarray:
    text = files("eyetrack").joinpath("assets/canonical_face_model.obj").read_text()
    verts = [line.split()[1:4] for line in text.splitlines() if line.startswith("v ")]
    return 10.0 * np.array(verts, dtype=np.float64)


_MODEL_3D = _load_canonical_model()
_N_LANDMARKS = len(_MODEL_3D)          # 468; iris landmarks 468-477 aren't in the model
_NOSE_TIP = 1

# Midpoint between the two eye centres (corner midpoints), in model space;
# origin of the 3D gaze ray.
EYE_MIDPOINT_MODEL = _MODEL_3D[[33, 133, 263, 362]].mean(axis=0)

# The model is y-up with the face toward +z, while camera coordinates are
# y-down with z pointing away from the camera, so solvePnP's rotation for a head squarely facing
# the camera is a 180° turn about x. Undoing that before decomposing makes
# a frontal pose read (0, 0, 0) instead of pitch ≈ ±180°, and keeps
# RQDecomp3x3 on a single branch so yaw can't jump to 180° - yaw.
_MODEL_TO_CAMERA = np.diag([1.0, -1.0, -1.0])


class HeadPoseEstimator:
    def __init__(self, frame_w, frame_h):
        # Approximate intrinsics: focal length = frame width, principal point = center
        f  = float(frame_w)
        cx = frame_w / 2.0
        cy = frame_h / 2.0
        self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        self.D = np.zeros((4, 1))
        self._rvec = None
        self._tvec = None
        self._nose = None

    def _image_points(self, lms, shape):
        h, w = shape[:2]
        return np.array(
            [(lms.landmark[i].x * w, lms.landmark[i].y * h) for i in range(_N_LANDMARKS)],
            dtype=np.float64,
        )

    def estimate(self, lms, shape):
        """Returns (pitch, yaw, roll) in degrees, or (None, None, None) on failure."""
        pts2d = self._image_points(lms, shape)
        ok, rvec, tvec = cv2.solvePnP(_MODEL_3D, pts2d, self.K, self.D,
                                       flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            self._rvec = self._tvec = self._nose = None
            return None, None, None

        self._rvec = rvec
        self._tvec = tvec
        self._nose = pts2d[_NOSE_TIP].astype(int)

        rmat, _ = cv2.Rodrigues(rvec)
        angles  = cv2.RQDecomp3x3(rmat @ _MODEL_TO_CAMERA)[0]   # (pitch, yaw, roll) in degrees
        return float(angles[0]), float(angles[1]), float(angles[2])

    def draw_axes(self, frame, lms=None):
        """Draws X/Y/Z axes using the cached result from the last estimate() call."""
        if self._rvec is None:
            return

        axis = np.float32([[80, 0, 0], [0, 80, 0], [0, 0, 80]])
        projected, _ = cv2.projectPoints(axis, self._rvec, self._tvec, self.K, self.D)
        projected = projected.astype(int)

        cv2.line(frame, tuple(self._nose), tuple(projected[0].ravel()), (0,   0, 255), 2)  # X red
        cv2.line(frame, tuple(self._nose), tuple(projected[1].ravel()), (0, 255,   0), 2)  # Y green
        cv2.line(frame, tuple(self._nose), tuple(projected[2].ravel()), (255,  0,   0), 2)  # Z blue

    @property
    def rotation_matrix(self) -> np.ndarray | None:
        """3×3 rotation matrix from the last estimate() call, or None."""
        if self._rvec is None:
            return None
        R, _ = cv2.Rodrigues(self._rvec)
        return R

    @property
    def translation_vector(self) -> np.ndarray | None:
        """(3,1) translation vector from the last estimate() call, or None."""
        return self._tvec

    def draw_gaze_ray(self, frame: np.ndarray,
                      origin: np.ndarray, direction: np.ndarray,
                      length: float = 150.0) -> None:
        """
        Projects a 3D gaze ray (origin + direction, in camera coordinates)
        onto the frame and draws it as an orange arrow.
        length is in the same units as the face model (mm).
        """
        def _project(p: np.ndarray) -> tuple[int, int] | None:
            if p[2] <= 0:
                return None
            x = int(self.K[0, 0] * p[0] / p[2] + self.K[0, 2])
            y = int(self.K[1, 1] * p[1] / p[2] + self.K[1, 2])
            return x, y

        p1 = _project(origin)
        p2 = _project(origin + direction * length)
        if p1 is None or p2 is None:
            return

        h, w = frame.shape[:2]
        if not (0 <= p1[0] < w and 0 <= p1[1] < h):
            return

        p2_clipped = (int(np.clip(p2[0], -w, 2 * w)),
                      int(np.clip(p2[1], -h, 2 * h)))
        cv2.arrowedLine(frame, p1, p2_clipped, (0, 165, 255), 2, tipLength=0.25)
