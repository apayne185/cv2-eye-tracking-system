import pandas as pd

from conftest import H, W
from eyetrack.pipeline import CSV_COLUMNS, FrameProcessor
from eyetrack.render import draw_result, gaze_label
from eyetrack.session import SessionRecorder, build_summary


def _run(tracker, frame, n=5, export_ply=False):
    proc = FrameProcessor(W, H, tracker=tracker)
    rec = SessionRecorder(W, H, export_ply=export_ply)
    for i in range(n):
        rec.add(proc.process(frame, i / 30))
    return proc, rec


def test_save_writes_csv_summary_and_heatmap(tmp_path, frame, face_tracker):
    proc, rec = _run(face_tracker, frame)
    out = rec.save(tmp_path, face_tracker.fixations, proc.aoi.time_spent)

    df = pd.read_csv(out.csv)
    assert list(df.columns) == list(CSV_COLUMNS)
    assert len(df) == 5
    assert out.summary.read_text().startswith("--- Session Summary ---")
    assert out.heatmap.exists()
    assert out.face_mesh_ply is None


def test_export_ply_writes_point_clouds(tmp_path, frame, face_tracker):
    proc, rec = _run(face_tracker, frame, export_ply=True)
    out = rec.save(tmp_path, [], proc.aoi.time_spent)
    assert out.face_mesh_ply.exists()
    assert out.gaze_trajectory_ply.exists()


def test_no_face_session_still_saves(tmp_path, frame, no_face_tracker):
    proc, rec = _run(no_face_tracker, frame)
    out = rec.save(tmp_path, [], proc.aoi.time_spent)
    assert pd.read_csv(out.csv)["gaze_x"].isna().all()
    assert out.gaze_trajectory_ply is None


def test_empty_session_writes_nothing(tmp_path):
    out = SessionRecorder(W, H).save(tmp_path / "out", [], {})
    assert out.csv is None
    assert not (tmp_path / "out").exists()


def test_summary_reports_blinks_fixations_and_aoi():
    df = pd.DataFrame({
        "is_blink": [True, False, False, False],
        "is_fixation": [False, True, True, False],
        "active_aoi": ["Center", "Center", None, "Left"],
    })
    text = build_summary(df, [{"duration": 0.5}], {"Center": 1.25})
    assert "Blinks detected:  1" in text
    assert "Fixation frames:  2  (50.0%)" in text
    assert "Fixations:        1" in text
    assert "Center: 2  (50.0%)" in text
    assert "Center: 1.25s" in text


def test_draw_result_draws_only_with_face(frame, face_tracker, no_face_tracker):
    proc = FrameProcessor(W, H, tracker=no_face_tracker)
    draw_result(frame, proc.process(frame, 0.0), proc)
    assert not frame.any()

    proc = FrameProcessor(W, H, tracker=face_tracker)
    draw_result(frame, proc.process(frame, 0.0), proc)
    assert frame.any()


def test_gaze_label():
    assert gaze_label(0.5, 0.5) == "CENTER"
    assert gaze_label(0.2, 0.5) == "LEFT"
    assert gaze_label(0.8, 0.1) == "UP RIGHT"
