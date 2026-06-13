import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from face_mesh_3d import write_ply, export_session_face_mesh, export_gaze_trajectory


def _read_ply_vertex_count(path: Path) -> int:
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"end_header":
                break
    for line in header.decode("ascii").splitlines():
        if line.startswith("element vertex"):
            return int(line.split()[-1])
    return -1


def test_write_ply_creates_file():
    pts = np.random.rand(100, 3).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = Path(f.name)
    write_ply(path, pts)
    assert path.exists()
    assert path.stat().st_size > 0


def test_write_ply_vertex_count_in_header():
    pts = np.zeros((42, 3), dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = Path(f.name)
    write_ply(path, pts)
    assert _read_ply_vertex_count(path) == 42


def test_write_ply_with_colors():
    pts    = np.zeros((10, 3), dtype=np.float32)
    colors = np.full((10, 3), 128, dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = Path(f.name)
    write_ply(path, pts, colors)
    assert _read_ply_vertex_count(path) == 10


def test_export_session_face_mesh_empty_does_not_raise():
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = Path(f.name)
    export_session_face_mesh(path, [])
    # File untouched — no crash


def test_export_session_face_mesh_point_count():
    frames = [np.zeros((478, 3), dtype=np.float32) for _ in range(5)]
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = Path(f.name)
    export_session_face_mesh(path, frames)
    assert _read_ply_vertex_count(path) == 5 * 478


def test_export_gaze_trajectory_point_count():
    n = 100
    origins    = np.zeros((n, 3), dtype=np.float32)
    directions = np.tile([0.0, 0.0, 1.0], (n, 1)).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = Path(f.name)
    export_gaze_trajectory(path, origins, directions, depth=500.0)
    assert _read_ply_vertex_count(path) == n


def test_export_gaze_trajectory_depth_offset():
    origins    = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    directions = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = Path(f.name)
    export_gaze_trajectory(path, origins, directions, depth=300.0)
    # Endpoint should be at z=300
    with open(path, "rb") as f:
        content = f.read()
    end_header = content.index(b"end_header\n") + len(b"end_header\n")
    x, y, z = struct.unpack_from("<fff", content, end_header)
    assert abs(z - 300.0) < 0.01
