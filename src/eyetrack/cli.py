"""
Command-line entry point.

    eyetrack run        track a webcam, file, or stream and save session outputs
    eyetrack calibrate  fit a per-user gaze calibration and save it

Exit codes: 0 success, 1 runtime failure (e.g. source unavailable),
2 invalid arguments or config.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

from . import __version__
from .calibration import GazeCalibrator
from .config import Config, ConfigError, load_config
from .eye_tracker import EyeTracker
from .gaze_classifier import GazeZoneClassifier
from .logs import setup_logging
from .pipeline import FrameProcessor
from .render import draw_result
from .session import SessionRecorder
from .sources import SourceError, VideoSource

log = logging.getLogger("eyetrack")

COMMANDS = ("run", "calibrate")
_WINDOW = "Eye Tracker"
_HEATMAP_WARMUP_FRAMES = 10


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eyetrack", description="Real-time eye tracking")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, help="TOML config file (see eyetrack.example.toml)")
    common.add_argument("--source", help="webcam index, video file, or rtsp/http URL (default: 0)")
    common.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    common.add_argument("--log-format", choices=["text", "json"])

    run = sub.add_parser("run", parents=[common], help="track gaze and save session outputs")
    run.add_argument("--output-dir", type=Path, help="where session files go (default: data/)")
    run.add_argument("--export-ply", action="store_true", default=None,
                     help="also export face mesh and gaze trajectory point clouds")
    run.add_argument("--calibrate", action="store_true",
                     help="run 5-point calibration before the session")
    run.add_argument("--no-display", dest="display", action="store_false", default=None,
                     help="headless: no preview window; stop with Ctrl+C or end of input")
    run.add_argument("--max-frames", type=int, help="stop after N frames")

    sub.add_parser("calibrate", parents=[common], help="run 5-point calibration and save it")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `eyetrack` and `eyetrack --source 0` still mean `eyetrack run ...`
    if not argv or argv[0] not in (*COMMANDS, "-h", "--help", "--version"):
        argv.insert(0, "run")
    args = build_parser().parse_args(argv)

    overrides = {k: getattr(args, k, None) for k in (
        "source", "output_dir", "export_ply", "display", "max_frames", "log_level", "log_format")}
    try:
        cfg = load_config(args.config, overrides)
    except ConfigError as e:
        print(f"eyetrack: error: {e}", file=sys.stderr)
        return 2
    setup_logging(cfg.log_level, cfg.log_format)

    calibrating = args.command == "calibrate" or getattr(args, "calibrate", False)
    if calibrating and not cfg.display:
        log.error("calibration needs a display; drop --no-display")
        return 2

    try:
        source = VideoSource(cfg.source)
    except SourceError as e:
        log.error(str(e))
        return 1

    with source:
        log.info("opened source %s (%dx%d, live=%s)", cfg.source, source.width,
                 source.height, source.is_live,
                 extra={"source": cfg.source, "width": source.width,
                        "height": source.height, "live": source.is_live})
        tracker = _make_tracker(cfg)
        if args.command == "calibrate":
            return 0 if _calibrate(cfg, source, tracker) else 1
        calibrator = _calibrate(cfg, source, tracker) if args.calibrate else _load_calibration(cfg)
        return _run(cfg, source, tracker, calibrator)


def _make_tracker(cfg: Config) -> EyeTracker:
    return EyeTracker(ear_blink_threshold=cfg.ear_blink_threshold,
                      fixation_velocity=cfg.fixation_velocity,
                      min_fixation_secs=cfg.min_fixation_secs)


def _calibrate(cfg: Config, source: VideoSource, tracker: EyeTracker) -> GazeCalibrator | None:
    log.info("starting 5-point calibration: follow the dot with your eyes")
    try:
        calibrator = GazeCalibrator().run(source.cap, tracker, source.width, source.height)
    finally:
        cv2.destroyAllWindows()
    if not calibrator.is_fitted:
        log.error("calibration did not collect enough samples")
        return None
    path = calibrator.save(cfg.calibration_path)
    log.info("calibration saved to %s", path, extra={"path": str(path)})
    return calibrator


def _load_calibration(cfg: Config) -> GazeCalibrator | None:
    return _load_optional(cfg.calibration_path, GazeCalibrator.load, "calibration")


def _load_optional(path: Path, loader, what: str):
    if not path.exists():
        log.debug("no %s at %s", what, path)
        return None
    try:
        obj = loader(path)
    except Exception as e:
        log.warning("could not load %s from %s: %s", what, path, e)
        return None
    log.info("loaded %s from %s", what, path)
    return obj


def _run(cfg: Config, source: VideoSource, tracker: EyeTracker,
         calibrator: GazeCalibrator | None) -> int:
    zone_clf = _load_optional(cfg.classifier_path, GazeZoneClassifier.load, "zone classifier")
    proc = FrameProcessor(source.width, source.height, calibrator=calibrator,
                          zone_classifier=zone_clf, aois=cfg.aois, tracker=tracker)
    recorder = SessionRecorder(source.width, source.height, export_ply=cfg.export_ply)
    last_overlay = None

    log.info("running: %s", "press 'q' in the window to stop" if cfg.display
             else "headless, Ctrl+C to stop")
    try:
        for frame, ts in source.frames():
            res = proc.process(frame, ts)
            recorder.add(res)
            if cfg.max_frames is not None and len(recorder.rows) >= cfg.max_frames:
                break
            if not cfg.display:
                continue
            draw_result(frame, res, proc)
            if res.frame > _HEATMAP_WARMUP_FRAMES:
                last_overlay = recorder.heatmap_overlay(frame)
            cv2.imshow(_WINDOW, last_overlay if last_overlay is not None else frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        log.info("interrupted, saving session")
    finally:
        if cfg.display:
            cv2.destroyAllWindows()

    outputs = recorder.save(cfg.output_dir, tracker.fixations, proc.aoi.time_spent, last_overlay)
    if outputs.csv is None:
        log.warning("no frames processed; nothing saved")
        return 1

    faces = sum(r["gaze_x"] is not None for r in recorder.rows)
    log.info("saved %d frames (%d with a face) to %s", len(recorder.rows), faces, outputs.csv,
             extra={"frames": len(recorder.rows), "face_frames": faces, "csv": str(outputs.csv)})
    for label, path in (("summary", outputs.summary), ("heatmap", outputs.heatmap),
                        ("face mesh PLY", outputs.face_mesh_ply),
                        ("gaze trajectory PLY", outputs.gaze_trajectory_ply)):
        if path is not None:
            log.info("%s saved to %s", label, path)
    print(f"\n{outputs.summary_text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
