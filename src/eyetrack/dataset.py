"""
Labelled real-video segments for evaluating (and training) the zone classifier.

The manifest (CSV) lists segments of public-domain videos with a gaze-zone
label per segment. Videos are not stored in the repo: fetch() downloads
just enough of each source to cover its segments into a local cache.
"""

import csv
import math
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .gaze_classifier import ZONES
from .pipeline import FrameProcessor

MANIFEST_COLUMNS = (
    "segment_id", "subject", "label", "start_s", "end_s",
    "source_title", "source_page", "media_url", "media_duration_s",
    "license", "notes",
)
ALLOWED_LICENSES = {"Public domain", "CC0"}
_USER_AGENT = "eyetrack-dataset/0.2 (https://github.com/apayne185/cv2-eye-tracking-system)"


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class Segment:
    segment_id: str
    subject: str
    label: str
    start_s: float
    end_s: float
    source_title: str
    source_page: str
    media_url: str
    media_duration_s: float
    license: str
    notes: str = ""


def load_manifest(path: str | Path) -> list[Segment]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ManifestError(f"manifest missing columns: {sorted(missing)}")
        segments = []
        for n, row in enumerate(reader, start=2):
            try:
                seg = Segment(**{k: row[k] for k in MANIFEST_COLUMNS
                                 if k not in ("start_s", "end_s", "media_duration_s")},
                              start_s=float(row["start_s"]), end_s=float(row["end_s"]),
                              media_duration_s=float(row["media_duration_s"]))
            except ValueError as e:
                raise ManifestError(f"line {n}: {e}") from e
            _validate(seg, n)
            segments.append(seg)
    ids = [s.segment_id for s in segments]
    if len(ids) != len(set(ids)):
        raise ManifestError("duplicate segment_id")
    return segments


def _validate(seg: Segment, line: int) -> None:
    if seg.label not in ZONES:
        raise ManifestError(f"line {line}: label {seg.label!r} not in {ZONES}")
    if not 0 <= seg.start_s < seg.end_s <= seg.media_duration_s:
        raise ManifestError(f"line {line}: need 0 <= start_s < end_s <= media_duration_s")
    if seg.license not in ALLOWED_LICENSES:
        raise ManifestError(f"line {line}: license {seg.license!r} not in {sorted(ALLOWED_LICENSES)}")


def cache_path(media_url: str, cache_dir: str | Path) -> Path:
    suffix = Path(media_url.split("?")[0]).suffix or ".bin"
    return Path(cache_dir) / f"{sha1(media_url.encode()).hexdigest()[:16]}{suffix}"


def bytes_needed(total_size: int, duration_s: float, end_s: float) -> int:
    """Bytes of a streamable file that cover [0, end_s], with headroom."""
    frac = min(1.0, end_s / duration_s * 1.15)
    return min(total_size, math.ceil(total_size * frac) + 2_000_000)


def _http(url: str, headers: dict, method: str = "GET"):
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, **headers}, method=method)
    for attempt in range(5):
        try:
            return urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 4:
                raise
            time.sleep(int(e.headers.get("Retry-After", 10 * (attempt + 1))))
    raise RuntimeError("unreachable")


def fetch(segments: list[Segment], cache_dir: str | Path) -> dict[str, Path]:
    """
    Downloads each distinct media_url once, only as far as its latest
    segment needs (WebM decodes fine when truncated). Local paths and
    file:// URLs are used in place. Returns media_url -> local path.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for url in sorted({s.media_url for s in segments}):
        if url.startswith("file://") or "://" not in url:
            paths[url] = Path(url.removeprefix("file://"))
            continue
        path = cache_path(url, cache_dir)
        if not path.exists():
            segs = [s for s in segments if s.media_url == url]
            size = int(_http(url, {}, "HEAD").headers["Content-Length"])
            n = bytes_needed(size, segs[0].media_duration_s, max(s.end_s for s in segs))
            data = _http(url, {"Range": f"bytes=0-{n - 1}"}).read()
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.rename(path)
            time.sleep(1)   # be polite to the media host
        paths[url] = path
    return paths


def segment_frames(path: str | Path, seg: Segment,
                   sample_fps: float = 5.0) -> Iterator[tuple[np.ndarray, float]]:
    """Yields (frame, seconds_into_source) for [start_s, end_s) at ~sample_fps."""
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(fps / sample_fps))
        first = math.ceil(seg.start_s * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        idx = first
        while idx / fps < seg.end_s:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - first) % step == 0:
                yield frame, idx / fps
            idx += 1
    finally:
        cap.release()


FEATURE_COLUMNS = ("gaze_ratio_h", "gaze_ratio_v", "yaw", "pitch", "dir_h", "dir_v")


def extract_features(segments: list[Segment], paths: dict[str, Path],
                     sample_fps: float = 5.0) -> pd.DataFrame:
    """One row per sampled frame: segment metadata, label, and pipeline outputs."""
    rows = []
    for seg in segments:
        proc = None
        for frame, t in segment_frames(paths[seg.media_url], seg, sample_fps):
            proc = proc or FrameProcessor(frame.shape[1], frame.shape[0])
            res = proc.process(frame, t)
            rows.append({
                "segment_id": seg.segment_id, "subject": seg.subject,
                "label": seg.label, "t": round(t, 3),
                "face_detected": res.face_detected,
                **{c: getattr(res, c) for c in FEATURE_COLUMNS},
            })
    return pd.DataFrame(rows)
