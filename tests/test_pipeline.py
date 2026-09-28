import numpy as np
import pytest

from conftest import H, W
from eyetrack.gaze_classifier import ZONES, GazeZoneClassifier, generate_training_data
from eyetrack.pipeline import CSV_COLUMNS, FrameProcessor


def test_no_face_gives_empty_result(frame, no_face_tracker):
    proc = FrameProcessor(W, H, tracker=no_face_tracker)
    res = proc.process(frame, 1.0)
    assert not res.face_detected
    assert res.gaze_x is None and res.pitch is None and res.ray_origin is None
    assert res.to_row()["is_blink"] is False


def test_frame_index_increments(frame, no_face_tracker):
    proc = FrameProcessor(W, H, tracker=no_face_tracker)
    assert [proc.process(frame, t).frame for t in (0.0, 0.1, 0.2)] == [0, 1, 2]


def test_face_populates_gaze_pose_and_ray(frame, face_tracker):
    res = FrameProcessor(W, H, tracker=face_tracker).process(frame, 0.0)

    assert res.face_detected
    assert res.gaze_ratio_h == pytest.approx(0.5, abs=0.01)
    assert res.gaze_ratio_v == pytest.approx(0.5, abs=0.01)
    assert (res.gaze_x, res.gaze_y) == (320, 200)
    assert not res.is_blink
    assert res.pitch is not None and res.yaw is not None
    assert res.dir_h is not None
    assert np.linalg.norm(res.ray_direction) == pytest.approx(1.0)
    assert res.active_aoi == "Center"


def test_optional_models_are_skipped_when_absent(frame, face_tracker):
    res = FrameProcessor(W, H, tracker=face_tracker).process(frame, 0.0)
    assert res.screen_point is None
    assert res.predicted_zone is None


def test_zone_classifier_prediction(frame, face_tracker):
    clf = GazeZoneClassifier().train(*generate_training_data(n_per_class=60))
    res = FrameProcessor(W, H, tracker=face_tracker, zone_classifier=clf).process(frame, 0.0)
    assert res.predicted_zone in ZONES


def test_process_does_not_draw_on_frame(frame, face_tracker):
    FrameProcessor(W, H, tracker=face_tracker).process(frame, 0.0)
    assert not frame.any()


def test_stationary_gaze_becomes_fixation(frame, face_tracker):
    proc = FrameProcessor(W, H, tracker=face_tracker)
    results = [proc.process(frame, i / 30) for i in range(10)]
    assert not results[0].is_fixation
    assert all(r.is_fixation for r in results[1:])


def test_to_row_matches_csv_schema_and_rounds(frame, face_tracker):
    row = FrameProcessor(W, H, tracker=face_tracker).process(frame, 12.345678).to_row()
    assert tuple(row) == CSV_COLUMNS
    assert row["timestamp"] == 12.3457
    assert row["ray_dx"] == round(row["ray_dx"], 3)
