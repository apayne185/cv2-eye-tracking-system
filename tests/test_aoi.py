import numpy as np
import pytest

from eyetrack.aoi import AOITracker

AOIS = {"A": (0, 0, 100, 100), "B": (200, 0, 300, 100)}


def _frame():
    return np.zeros((200, 400, 3), dtype=np.uint8)


def test_returns_active_aoi_name():
    tracker = AOITracker(AOIS)
    assert tracker.track(_frame(), (50, 50), ts=0.0) == "A"
    assert tracker.track(_frame(), (250, 50), ts=0.1) == "B"


def test_gaze_outside_all_aois_returns_none():
    tracker = AOITracker(AOIS)
    assert tracker.track(_frame(), (150, 150), ts=0.0) is None


def test_boundaries_are_inclusive():
    tracker = AOITracker(AOIS)
    assert tracker.track(_frame(), (100, 100), ts=0.0) == "A"


def test_dwell_time_accumulates_between_frames():
    tracker = AOITracker(AOIS)
    for i in range(5):
        tracker.track(_frame(), (50, 50), ts=i * 0.5)
    # first frame only sets the reference timestamp
    assert tracker.time_spent["A"] == pytest.approx(2.0)
    assert "B" not in tracker.time_spent


def test_draws_aoi_rectangles_on_frame():
    frame = _frame()
    AOITracker(AOIS).track(frame, (50, 50), ts=0.0)
    assert frame.any()


def test_default_layout_used_when_none_given():
    tracker = AOITracker()
    assert set(tracker.aois) == {"Left", "Center", "Right"}
