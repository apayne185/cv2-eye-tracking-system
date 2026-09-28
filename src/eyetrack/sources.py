"""Video input: webcams, network streams, and recorded files."""

import time
from collections.abc import Iterator

import cv2
import numpy as np

_DEFAULT_FPS = 30.0
_STREAM_SCHEMES = ("rtsp://", "rtmp://", "http://", "https://")


class SourceError(RuntimeError):
    pass


class VideoSource:
    """
    Wraps cv2.VideoCapture and timestamps each frame.

    Live sources (webcam index, rtsp/http URL) use wall-clock time.
    Files use media time (frame index / fps), so fixation and dwell
    timings reflect the recording rather than how fast it is processed.
    """

    def __init__(self, source: str | int):
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.source = source
        self.is_live = isinstance(source, int) or str(source).startswith(_STREAM_SCHEMES)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.cap.release()
            raise SourceError(f"cannot open video source {source!r}")
        self._frame_idx = 0

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else _DEFAULT_FPS

    def read(self) -> tuple[np.ndarray, float] | None:
        """Returns (frame, timestamp_seconds), or None at end of stream."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        ts = time.time() if self.is_live else self._frame_idx / self.fps
        self._frame_idx += 1
        return frame, ts

    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        while (item := self.read()) is not None:
            yield item

    def release(self) -> None:
        self.cap.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
