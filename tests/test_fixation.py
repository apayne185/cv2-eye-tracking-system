from eye_tracker import EyeTracker, MIN_FIXATION_SECS


def test_stationary_gaze_is_fixation():
    tracker = EyeTracker()
    is_fix = False
    for i in range(5):
        is_fix = tracker.update_fixation((100, 100), i * 0.05)
    assert is_fix


def test_fast_saccade_is_not_fixation():
    tracker = EyeTracker()
    tracker.update_fixation((0, 0), 0.0)
    is_fix = tracker.update_fixation((1000, 0), 0.001)  # ~1,000,000 px/s
    assert not is_fix


def test_completed_fixation_is_logged():
    tracker = EyeTracker()
    for i in range(10):
        tracker.update_fixation((50, 50), i * 0.05)
    # End fixation with a fast jump
    tracker.update_fixation((1000, 1000), 10 * 0.05 + 0.001)
    assert len(tracker.fixations) == 1


def test_fixation_duration_meets_minimum():
    tracker = EyeTracker()
    for i in range(10):
        tracker.update_fixation((50, 50), i * 0.05)
    tracker.update_fixation((1000, 1000), 10 * 0.05 + 0.001)
    assert tracker.fixations[0]["duration"] >= MIN_FIXATION_SECS


def test_too_short_fixation_not_logged():
    tracker = EyeTracker()
    tracker.update_fixation((50, 50), 0.0)
    tracker.update_fixation((50, 50), 0.01)   # only 10 ms — below threshold
    tracker.update_fixation((1000, 1000), 0.02)
    assert len(tracker.fixations) == 0
