"""End-to-end CLI runs: real VideoSource, real MediaPipe, headless."""

import cv2
import numpy as np
import pandas as pd
import pytest

from eyetrack.cli import main
from eyetrack.pipeline import CSV_COLUMNS


@pytest.fixture
def blank_video(tmp_path):
    path = tmp_path / "blank.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (320, 240))
    for _ in range(12):
        writer.write(np.full((240, 320, 3), 90, dtype=np.uint8))
    writer.release()
    return path


def _run(*args):
    return main(["run", "--no-display", *args])


def test_headless_run_on_video_saves_session(tmp_path, blank_video, capsys):
    out = tmp_path / "out"
    assert _run("--source", str(blank_video), "--output-dir", str(out)) == 0

    csvs = list(out.glob("gaze_*.csv"))
    assert len(csvs) == 1
    df = pd.read_csv(csvs[0])
    assert list(df.columns) == list(CSV_COLUMNS)
    assert len(df) == 12
    assert df["gaze_x"].isna().all()                      # no face in a blank video
    assert df["timestamp"].tolist() == pytest.approx([i / 30 for i in range(12)], abs=1e-4)
    assert list(out.glob("summary_*.txt")) and list(out.glob("heatmap_*.jpg"))
    assert "Frames recorded:  12" in capsys.readouterr().out


def test_max_frames(tmp_path, blank_video):
    assert _run("--source", str(blank_video), "--output-dir", str(tmp_path),
                "--max-frames", "5") == 0
    assert len(pd.read_csv(next(tmp_path.glob("gaze_*.csv")))) == 5


def test_bare_flags_default_to_run(tmp_path, blank_video):
    assert main(["--source", str(blank_video), "--no-display",
                 "--output-dir", str(tmp_path)]) == 0


def test_unopenable_source_exits_1(tmp_path):
    assert _run("--source", str(tmp_path / "missing.mp4")) == 1


def test_bad_config_exits_2(tmp_path):
    cfg = tmp_path / "bad.toml"
    cfg.write_text("nonsense = true")
    assert _run("--config", str(cfg)) == 2


def test_calibration_requires_display(blank_video):
    assert _run("--source", str(blank_video), "--calibrate") == 2


def test_json_logs(tmp_path, blank_video, capsys):
    _run("--source", str(blank_video), "--output-dir", str(tmp_path), "--log-format", "json")
    err_lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("{")]
    assert any('"face_frames": 0' in ln for ln in err_lines)
