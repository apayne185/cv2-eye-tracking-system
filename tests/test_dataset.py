import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from eyetrack.dataset import (
    MANIFEST_COLUMNS,
    ManifestError,
    Segment,
    bytes_needed,
    cache_path,
    extract_features,
    fetch,
    load_manifest,
    segment_frames,
)

CLIP = Path(__file__).parent / "fixtures" / "face_clip.mp4"


def _row(**over):
    row = dict(segment_id="s1", subject="alice", label="on_screen", start_s="1", end_s="3",
               source_title="t", source_page="https://example.org/p", media_url=str(CLIP),
               media_duration_s="4", license="Public domain", notes="")
    row.update(over)
    return row


def _write(tmp_path, *rows):
    path = tmp_path / "manifest.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def test_load_manifest(tmp_path):
    segs = load_manifest(_write(tmp_path, _row(), _row(segment_id="s2", label="away")))
    assert [s.segment_id for s in segs] == ["s1", "s2"]
    assert segs[0].start_s == 1.0 and segs[1].label == "away"


@pytest.mark.parametrize("over, message", [
    ({"label": "sideways"}, "label"),
    ({"start_s": "3", "end_s": "1"}, "start_s < end_s"),
    ({"end_s": "9"}, "media_duration_s"),
    ({"license": "CC BY-SA 4.0"}, "license"),
    ({"start_s": "abc"}, "line 2"),
])
def test_invalid_rows_rejected(tmp_path, over, message):
    with pytest.raises(ManifestError, match=message):
        load_manifest(_write(tmp_path, _row(**over)))


def test_duplicate_ids_rejected(tmp_path):
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifest(_write(tmp_path, _row(), _row()))


def test_missing_columns_rejected(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("segment_id,label\ns1,away\n")
    with pytest.raises(ManifestError, match="missing columns"):
        load_manifest(path)


def test_bytes_needed_covers_segment_with_headroom():
    assert bytes_needed(100_000_000, 100.0, 50.0) == 57_500_000 + 2_000_000
    assert bytes_needed(10_000_000, 100.0, 99.0) == 10_000_000     # capped at file size


def test_cache_path_is_stable_and_keeps_extension(tmp_path):
    url = "https://upload.example.org/a/b/clip.webm.480p.vp9.webm"
    assert cache_path(url, tmp_path) == cache_path(url, tmp_path)
    assert cache_path(url, tmp_path).suffix == ".webm"


def test_fetch_uses_local_files_in_place(tmp_path):
    seg = Segment(**{**_row(), "start_s": 0.0, "end_s": 1.0, "media_duration_s": 4.0})
    assert fetch([seg], tmp_path / "cache") == {str(CLIP): CLIP}


@pytest.fixture
def ramp_video(tmp_path):
    path = tmp_path / "ramp.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48))
    for i in range(60):                                  # 3 s at 20 fps
        w.write(np.full((48, 64, 3), i * 4, dtype=np.uint8))
    w.release()
    return path


def test_segment_frames_samples_window(ramp_video):
    seg = Segment(**{**_row(), "start_s": 1.0, "end_s": 2.0, "media_duration_s": 3.0})
    times = [t for _, t in segment_frames(ramp_video, seg, sample_fps=5)]
    assert times == pytest.approx([1.0, 1.2, 1.4, 1.6, 1.8])


def test_extract_features_on_real_clip():
    seg = Segment(**{**_row(), "start_s": 0.0, "end_s": 2.0, "media_duration_s": 4.0})
    df = extract_features([seg], {str(CLIP): CLIP}, sample_fps=5)
    assert len(df) == 10
    assert df["face_detected"].all()
    assert set(df["label"]) == {"on_screen"} and set(df["subject"]) == {"alice"}
    assert df["yaw"].abs().mean() > 25                   # head turned in this clip
