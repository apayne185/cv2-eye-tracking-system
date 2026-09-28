"""
3D face landmark extraction and PLY point cloud export.

MediaPipe provides per-landmark (x, y, z) where x/y are normalised to [0,1]
by frame dimensions and z is relative depth at the same scale as x.
Multiplying by frame dimensions gives pixel-space 3D coordinates.

Two point clouds are exported per session:
  - face_mesh_<ts>.ply   : sampled face landmarks coloured by time (blue→red)
  - gaze_trajectory_<ts>.ply : 3D gaze ray endpoints coloured by horizontal position
"""

import struct
from pathlib import Path

import numpy as np


def landmarks_to_numpy(lms, frame_w: int, frame_h: int) -> np.ndarray:
    """
    Converts a MediaPipe face landmark result to a (N, 3) float32 array.
    Units: pixels for x/y; relative depth (same scale) for z.
    """
    return np.array(
        [[lm.x * frame_w, lm.y * frame_h, lm.z * frame_w]
         for lm in lms.landmark],
        dtype=np.float32,
    )


def write_ply(path: str | Path, points: np.ndarray,
              colors: np.ndarray | None = None) -> None:
    """
    Writes a point cloud to a binary little-endian PLY file.

    points : (N, 3) float32  — XYZ coordinates
    colors : (N, 3) uint8    — RGB per-point color, or None for no color
    """
    n = len(points)
    path = Path(path)

    color_props = (
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
    ) if colors is not None else ""

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"{color_props}"
        "end_header\n"
    )

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        if colors is not None:
            for i in range(n):
                f.write(struct.pack("<fff", *points[i]))
                f.write(struct.pack("<BBB", *colors[i].astype(np.uint8)))
        else:
            f.write(points.astype(np.float32).tobytes())


def export_session_face_mesh(path: str | Path,
                             frame_arrays: list[np.ndarray]) -> None:
    """
    Exports sampled face landmark frames as a single point cloud.
    Points are coloured from blue (session start) to red (session end),
    so the temporal evolution of head pose is visible in a 3D viewer.

    frame_arrays : list of (N_landmarks, 3) arrays, one per sampled frame
    """
    if not frame_arrays:
        return

    n_frames   = len(frame_arrays)
    all_points = np.concatenate(frame_arrays, axis=0)
    colors     = np.zeros((len(all_points), 3), dtype=np.uint8)

    offset = 0
    for i, pts in enumerate(frame_arrays):
        t = i / max(n_frames - 1, 1)
        colors[offset : offset + len(pts)] = [int(t * 255), 0, int((1 - t) * 255)]
        offset += len(pts)

    write_ply(path, all_points, colors)


def export_gaze_trajectory(path: str | Path,
                           origins: np.ndarray,
                           directions: np.ndarray,
                           depth: float = 500.0) -> None:
    """
    Exports gaze ray endpoints as a 3D point cloud.

    Each point is origin + direction * depth — the location in camera space
    where the gaze ray intersects a virtual plane at the given depth (mm).
    Points are coloured by horizontal position: blue = left, red = right.

    origins    : (N, 3) float32  — ray origins from GazeDirectionEstimator
    directions : (N, 3) float32  — unit direction vectors
    depth      : virtual screen depth in mm (default 500 = ~arm's length)
    """
    if len(origins) == 0:
        return

    endpoints = (origins + directions * depth).astype(np.float32)

    x_min = endpoints[:, 0].min()
    x_max = endpoints[:, 0].max()
    x_range = max(x_max - x_min, 1e-6)

    colors = np.zeros((len(endpoints), 3), dtype=np.uint8)
    t = (endpoints[:, 0] - x_min) / x_range       # 0 = leftmost, 1 = rightmost
    colors[:, 0] = (t * 255).astype(np.uint8)      # red channel
    colors[:, 2] = ((1 - t) * 255).astype(np.uint8)  # blue channel

    write_ply(path, endpoints, colors)
