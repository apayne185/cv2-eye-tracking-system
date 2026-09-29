import cv2
import numpy as np
import pytest

from eyetrack.sources import SourceError, VideoSource


@pytest.fixture
def video_file(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (160, 120))
    for i in range(10):
        writer.write(np.full((120, 160, 3), i * 20, dtype=np.uint8))
    writer.release()
    return path


def test_file_frames_use_media_time(video_file):
    with VideoSource(str(video_file)) as src:
        assert not src.is_live
        assert (src.width, src.height) == (160, 120)
        timestamps = [ts for _, ts in src.frames()]
    assert len(timestamps) == 10
    assert timestamps == pytest.approx([i / 20.0 for i in range(10)])


def test_read_returns_none_at_end(video_file):
    with VideoSource(str(video_file)) as src:
        list(src.frames())
        assert src.read() is None


def test_missing_file_raises(tmp_path):
    with pytest.raises(SourceError):
        VideoSource(str(tmp_path / "missing.mp4"))


@pytest.mark.parametrize("uri", ["rtsp://cam.local/stream", "http://10.0.0.2/mjpg"])
def test_stream_urls_are_live(monkeypatch, uri):
    class FakeCapture:
        def __init__(self, _):
            pass

        def isOpened(self):
            return True

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    assert VideoSource(uri).is_live
