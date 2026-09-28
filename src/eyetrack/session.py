"""Accumulates per-frame results and writes session outputs on exit."""

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .face_mesh_3d import (
    export_gaze_trajectory,
    export_session_face_mesh,
    landmarks_to_numpy,
)
from .gaze_analysis import add_gaze_point, make_accumulator, render_heatmap
from .pipeline import CSV_COLUMNS, FrameResult

_PLY_SAMPLE_INTERVAL = 30  # save one face mesh snapshot per N frames


@dataclass
class SessionOutputs:
    csv: Path | None = None
    summary: Path | None = None
    summary_text: str = ""
    heatmap: Path | None = None
    heatmap_overlay: Path | None = None
    face_mesh_ply: Path | None = None
    gaze_trajectory_ply: Path | None = None


class SessionRecorder:
    def __init__(self, frame_w: int, frame_h: int, export_ply: bool = False):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.export_ply = export_ply
        self.heat_acc = make_accumulator(frame_h, frame_w)
        self.rows: list[dict] = []
        self._mesh_snapshots: list[np.ndarray] = []
        self._ray_origins: list[np.ndarray] = []
        self._ray_directions: list[np.ndarray] = []

    def add(self, res: FrameResult) -> None:
        self.rows.append(res.to_row())
        if not res.face_detected:
            return
        add_gaze_point(self.heat_acc, res.gaze_x, res.gaze_y)
        if res.ray_origin is not None:
            self._ray_origins.append(res.ray_origin)
            self._ray_directions.append(res.ray_direction)
        if self.export_ply and res.frame % _PLY_SAMPLE_INTERVAL == 0:
            self._mesh_snapshots.append(
                landmarks_to_numpy(res.landmarks, self.frame_w, self.frame_h))

    def heatmap_overlay(self, frame: np.ndarray) -> np.ndarray:
        return cv2.addWeighted(frame, 0.6, render_heatmap(self.heat_acc), 0.4, 0)

    def save(self, out_dir: Path, fixations: list, aoi_time_spent: dict,
             last_overlay: np.ndarray | None = None) -> SessionOutputs:
        outputs = SessionOutputs()
        if not self.rows:
            return outputs

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        df = pd.DataFrame(self.rows, columns=list(CSV_COLUMNS))

        outputs.csv = out_dir / f"gaze_{ts_str}.csv"
        df.to_csv(outputs.csv, index=False)

        outputs.summary_text = build_summary(df, fixations, aoi_time_spent)
        outputs.summary = out_dir / f"summary_{ts_str}.txt"
        outputs.summary.write_text(outputs.summary_text)

        outputs.heatmap = out_dir / f"heatmap_{ts_str}.jpg"
        cv2.imwrite(str(outputs.heatmap), render_heatmap(self.heat_acc))
        if last_overlay is not None:
            outputs.heatmap_overlay = out_dir / f"heatmap_overlay_{ts_str}.jpg"
            cv2.imwrite(str(outputs.heatmap_overlay), last_overlay)

        if self.export_ply and self._mesh_snapshots:
            outputs.face_mesh_ply = out_dir / f"face_mesh_{ts_str}.ply"
            export_session_face_mesh(outputs.face_mesh_ply, self._mesh_snapshots)
        if self.export_ply and self._ray_origins:
            outputs.gaze_trajectory_ply = out_dir / f"gaze_trajectory_{ts_str}.ply"
            export_gaze_trajectory(
                outputs.gaze_trajectory_ply,
                np.array(self._ray_origins, dtype=np.float32),
                np.array(self._ray_directions, dtype=np.float32),
            )
        return outputs


def build_summary(df: pd.DataFrame, fixations: list, aoi_time_spent: dict) -> str:
    n = len(df)
    blinks     = int(df["is_blink"].sum())
    fix_frames = int(df["is_fixation"].sum())

    lines = [
        "--- Session Summary ---",
        f"Frames recorded:  {n}",
        f"Blinks detected:  {blinks}",
        f"Fixation frames:  {fix_frames}  ({100 * fix_frames / n:.1f}%)",
    ]

    if fixations:
        durs = [f["duration"] for f in fixations]
        lines.append(
            f"Fixations:        {len(fixations)}"
            f"  avg={np.mean(durs):.2f}s"
            f"  max={np.max(durs):.2f}s"
        )

    aoi_col = df["active_aoi"].dropna()
    if not aoi_col.empty:
        lines.append("AOI dwell (frames):")
        for name, cnt in aoi_col.value_counts().items():
            lines.append(f"  {name}: {cnt}  ({100 * cnt / n:.1f}%)")

    if aoi_time_spent:
        lines.append("AOI dwell (seconds):")
        for name, secs in sorted(aoi_time_spent.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {secs:.2f}s")

    return "\n".join(lines)
