import argparse
from pathlib import Path

import cv2

from .calibration import DEFAULT_CALIB_PATH, GazeCalibrator
from .eye_tracker import EyeTracker
from .gaze_classifier import DEFAULT_MODEL_PATH, GazeZoneClassifier
from .pipeline import FrameProcessor
from .render import draw_result
from .session import SessionRecorder
from .sources import SourceError, VideoSource


def parse_args():
    p = argparse.ArgumentParser(description="Eye Tracking System")
    p.add_argument(
        "--source", default="0",
        help="Webcam index or path to a video file (default: 0)",
    )
    p.add_argument(
        "--output-dir", default="../data",
        help="Directory for CSV and heatmap output (default: ../data)",
    )
    p.add_argument(
        "--export-ply", action="store_true",
        help="Export face mesh and gaze trajectory as PLY point clouds on exit",
    )
    p.add_argument(
        "--calibrate", action="store_true",
        help="Run 5-point gaze calibration before starting the session",
    )
    return p.parse_args()


def main():
    args   = parse_args()
    out    = Path(args.output_dir)

    try:
        source = VideoSource(args.source)
    except SourceError as e:
        print(f"Error: {e}")
        return

    try:
        frame_w, frame_h = source.width, source.height
        tracker = EyeTracker()

        # --- calibration ---
        calibrator = None
        if args.calibrate:
            print("Starting 5-point calibration — follow the dot with your eyes.")
            calibrator = GazeCalibrator().run(source.cap, tracker, frame_w, frame_h)
            if calibrator.is_fitted:
                calib_path = calibrator.save()
                print(f"Calibration saved → {calib_path}")
        elif DEFAULT_CALIB_PATH.exists():
            try:
                calibrator = GazeCalibrator.load(DEFAULT_CALIB_PATH)
                print(f"Loaded calibration from {DEFAULT_CALIB_PATH}")
            except Exception as e:
                print(f"Warning: could not load calibration ({e})")

        zone_clf = None
        if DEFAULT_MODEL_PATH.exists():
            try:
                zone_clf = GazeZoneClassifier.load(DEFAULT_MODEL_PATH)
                print(f"Loaded zone classifier from {DEFAULT_MODEL_PATH}")
            except Exception as e:
                print(f"Warning: could not load zone classifier ({e})")

        proc = FrameProcessor(frame_w, frame_h, calibrator=calibrator,
                              zone_classifier=zone_clf, tracker=tracker)
        recorder     = SessionRecorder(frame_w, frame_h, export_ply=args.export_ply)
        last_overlay = None

        print("Running — press 'q' to quit and save results.")

        for frame, ts in source.frames():
            res = proc.process(frame, ts)
            recorder.add(res)
            draw_result(frame, res, proc)

            if res.frame > 10:
                last_overlay = recorder.heatmap_overlay(frame)
                cv2.imshow("Eye Tracker", last_overlay)
            else:
                cv2.imshow("Eye Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        outputs = recorder.save(out, tracker.fixations, proc.aoi.time_spent, last_overlay)
        if outputs.csv is None:
            return

        print(f"Saved {len(recorder.rows)} frames → {outputs.csv}")
        print(f"\n{outputs.summary_text}")
        print(f"Summary saved  → {outputs.summary}")
        print(f"Heatmap saved  → {outputs.heatmap}")
        if outputs.face_mesh_ply:
            print(f"Face mesh PLY  → {outputs.face_mesh_ply}")
        if outputs.gaze_trajectory_ply:
            print(f"Gaze trajectory PLY → {outputs.gaze_trajectory_ply}")

    finally:
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
