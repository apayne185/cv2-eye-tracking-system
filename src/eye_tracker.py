import cv2
import mediapipe as mp
import numpy as np

# MediaPipe FaceMesh landmark indices
_LEFT_EYE_OUTLINE  = [33, 133, 160, 158, 159, 144, 145, 153]
_RIGHT_EYE_OUTLINE = [362, 263, 387, 385, 386, 374, 380, 373]

# EAR: [outer, upper1, upper2, inner, lower2, lower1]
_LEFT_EAR_IDS  = [33, 160, 158, 133, 153, 144]
_RIGHT_EAR_IDS = [362, 387, 385, 263, 380, 373]

# Iris centers (requires refine_landmarks=True)
_LEFT_IRIS  = 468
_RIGHT_IRIS = 473

# Eye corner references for gaze ratio
_LEFT_OUTER  = 33;  _LEFT_INNER  = 133
_RIGHT_INNER = 362; _RIGHT_OUTER = 263
_LEFT_TOP    = 159; _LEFT_BOT    = 145
_RIGHT_TOP   = 386; _RIGHT_BOT   = 374

EAR_BLINK_THRESHOLD      = 0.20
FIXATION_VEL_PX_PER_SEC  = 25.0   # pixels/sec below which gaze is a fixation
MIN_FIXATION_SECS        = 0.10


class EyeTracker:
    def __init__(self):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,          # unlocks iris landmarks 468-477
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self._prev_gaze = None
        self._prev_ts   = None
        self._fix_start = None
        self._fixating  = False
        self.fixations  = []               # list of completed fixation dicts

    def process(self, frame):
        return self._mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def get_iris_gaze(self, lms, shape):
        """
        Returns (gaze_x, gaze_y, ratio_h, ratio_v).
        ratio_h: 0=left edge of eye, 1=right edge — center ~0.5
        ratio_v: 0=top edge,  1=bottom edge — center ~0.5
        """
        h, w = shape[:2]
        lm = lms.landmark

        def px(i): return lm[i].x * w
        def py(i): return lm[i].y * h

        lh = (px(_LEFT_IRIS)  - min(px(_LEFT_OUTER),  px(_LEFT_INNER)))  / (abs(px(_LEFT_OUTER)  - px(_LEFT_INNER))  + 1e-6)
        rh = (px(_RIGHT_IRIS) - min(px(_RIGHT_OUTER), px(_RIGHT_INNER))) / (abs(px(_RIGHT_OUTER) - px(_RIGHT_INNER)) + 1e-6)

        lv = (py(_LEFT_IRIS)  - min(py(_LEFT_TOP),  py(_LEFT_BOT)))  / (abs(py(_LEFT_TOP)  - py(_LEFT_BOT))  + 1e-6)
        rv = (py(_RIGHT_IRIS) - min(py(_RIGHT_TOP), py(_RIGHT_BOT))) / (abs(py(_RIGHT_TOP) - py(_RIGHT_BOT)) + 1e-6)

        ratio_h = (lh + rh) / 2
        ratio_v = (lv + rv) / 2
        gaze_x  = int((px(_LEFT_IRIS) + px(_RIGHT_IRIS)) / 2)
        gaze_y  = int((py(_LEFT_IRIS) + py(_RIGHT_IRIS)) / 2)

        return gaze_x, gaze_y, ratio_h, ratio_v

    def detect_blink(self, lms, shape):
        """Returns (is_blink, left_ear, right_ear)."""
        h, w = shape[:2]

        def pts(ids):
            return np.array([(lms.landmark[i].x * w, lms.landmark[i].y * h) for i in ids])

        def ear(p):
            # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
            return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (
                2 * np.linalg.norm(p[0] - p[3]) + 1e-6
            )

        l = ear(pts(_LEFT_EAR_IDS))
        r = ear(pts(_RIGHT_EAR_IDS))
        return (l + r) / 2 < EAR_BLINK_THRESHOLD, float(l), float(r)

    def update_fixation(self, gaze, ts):
        """
        Velocity-based fixation detection.
        Returns True if the current frame is part of a fixation.
        Completed fixations are appended to self.fixations.
        """
        is_fix = False
        if self._prev_gaze is not None and self._prev_ts is not None:
            dt = ts - self._prev_ts
            if dt > 0:
                vel = np.linalg.norm(np.subtract(gaze, self._prev_gaze)) / dt
                if vel < FIXATION_VEL_PX_PER_SEC:
                    if not self._fixating:
                        self._fix_start = ts
                        self._fixating  = True
                    is_fix = True
                else:
                    if self._fixating and self._fix_start is not None:
                        dur = ts - self._fix_start
                        if dur >= MIN_FIXATION_SECS:
                            self.fixations.append({
                                "x": gaze[0], "y": gaze[1],
                                "duration": dur, "end_time": ts,
                            })
                    self._fixating = False

        self._prev_gaze, self._prev_ts = gaze, ts
        return is_fix

    def draw_overlays(self, frame, lms):
        h, w = frame.shape[:2]
        lm = lms.landmark

        for ids, color in [(_LEFT_EYE_OUTLINE, (0, 255, 0)), (_RIGHT_EYE_OUTLINE, (255, 100, 0))]:
            pts = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in ids], np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)

        for idx in (_LEFT_IRIS, _RIGHT_IRIS):
            cv2.circle(frame, (int(lm[idx].x * w), int(lm[idx].y * h)), 3, (0, 255, 255), -1)
