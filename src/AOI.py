import cv2
import time
from collections import defaultdict

# Default AOI layout — override by passing a dict to AOITracker.__init__
DEFAULT_AOIS = {
    "Left":   (50,  100, 300, 400),
    "Center": (320, 100, 600, 400),
    "Right":  (610, 100, 900, 400),
}


class AOITracker:
    def __init__(self, aois=None):
        self.aois = aois or DEFAULT_AOIS
        self.time_spent = defaultdict(float)
        self._last_ts   = None

    def track(self, frame, gaze_point, ts=None):
        """
        Draws all AOIs on frame and accumulates dwell time for the active one.
        Returns the name of the active AOI, or None.
        Pass ts to use a frame-synchronised timestamp instead of wall clock.
        """
        now    = ts if ts is not None else time.time()
        active = None

        for name, (x1, y1, x2, y2) in self.aois.items():
            gx, gy = gaze_point
            inside = x1 <= gx <= x2 and y1 <= gy <= y2

            color = (0, 255, 0) if inside else (100, 100, 100)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if inside else 1)
            cv2.putText(frame, name, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

            if inside:
                active = name
                if self._last_ts is not None:
                    self.time_spent[name] += now - self._last_ts

        self._last_ts = now
        return active

    def print_summary(self):
        if not self.time_spent:
            return
        print("\n--- AOI Dwell Time ---")
        for name, secs in sorted(self.time_spent.items(), key=lambda x: -x[1]):
            print(f"  {name}: {secs:.2f}s")
