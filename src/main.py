import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from eye_tracker import EyeTracker
from head_pose import HeadPoseEstimator
from gaze_analysis import generate_heatmap
from AOI import AOITracker


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
    return p.parse_args()


def _gaze_label(ratio_h, ratio_v):
    h = "LEFT" if ratio_h < 0.40 else ("RIGHT" if ratio_h > 0.60 else "CENTER")
    v = "UP"   if ratio_v < 0.35 else ("DOWN"  if ratio_v > 0.65 else "")
    return f"{v} {h}".strip() if v else h


def _print_summary(df, fixations):
    n = len(df)
    blinks      = int(df["is_blink"].sum())
    fix_frames  = int(df["is_fixation"].sum())

    print(f"\n--- Session Summary ---")
    print(f"Frames recorded:  {n}")
    print(f"Blinks detected:  {blinks}")
    print(f"Fixation frames:  {fix_frames}  ({100 * fix_frames / n:.1f}%)")

    if fixations:
        durs = [f["duration"] for f in fixations]
        print(
            f"Fixations:        {len(fixations)}"
            f"  avg={np.mean(durs):.2f}s"
            f"  max={np.max(durs):.2f}s"
        )

    aoi_col = df["active_aoi"].dropna()
    if not aoi_col.empty:
        print("AOI dwell (frames):")
        for aoi, cnt in aoi_col.value_counts().items():
            print(f"  {aoi}: {cnt}  ({100 * cnt / n:.1f}%)")


def main():
    args   = parse_args()
    out    = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    source = int(args.source) if args.source.isdigit() else args.source
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: cannot open source '{args.source}'")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker  = EyeTracker()
    pose_est = HeadPoseEstimator(frame_w, frame_h)
    aoi      = AOITracker()

    records      = []
    gaze_pts     = []
    frame_idx    = 0
    last_frame   = None
    last_overlay = None

    print("Running — press 'q' to quit and save results.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ts      = time.time()
        results = tracker.process(frame)

        row = dict(
            frame=frame_idx, timestamp=round(ts, 4),
            gaze_x=None, gaze_y=None, gaze_ratio_h=None, gaze_ratio_v=None,
            pitch=None, yaw=None, roll=None,
            left_ear=None, right_ear=None,
            is_blink=False, is_fixation=False, active_aoi=None,
        )

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0]

            # --- iris-based gaze ---
            gx, gy, rh, rv = tracker.get_iris_gaze(lms, frame.shape)
            row.update(gaze_x=gx, gaze_y=gy,
                       gaze_ratio_h=round(rh, 3), gaze_ratio_v=round(rv, 3))
            gaze_pts.append((gx, gy))

            # --- fixation ---
            row["is_fixation"] = tracker.update_fixation((gx, gy), ts)

            # --- blink / EAR ---
            is_blink, l_ear, r_ear = tracker.detect_blink(lms, frame.shape)
            row.update(is_blink=is_blink,
                       left_ear=round(l_ear, 3), right_ear=round(r_ear, 3))

            # --- head pose ---
            pitch, yaw, roll = pose_est.estimate(lms, frame.shape)
            if pitch is not None:
                row.update(pitch=round(pitch, 1), yaw=round(yaw, 1), roll=round(roll, 1))

            # --- AOI ---
            row["active_aoi"] = aoi.track(frame, (gx, gy))

            # --- draw overlays ---
            tracker.draw_overlays(frame, lms)
            pose_est.draw_axes(frame, lms)

            label = "BLINK" if is_blink else _gaze_label(rh, rv)
            color = (0, 0, 255) if is_blink else (0, 255, 0)
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            if pitch is not None:
                cv2.putText(
                    frame, f"P:{pitch:.1f}  Y:{yaw:.1f}  R:{roll:.1f}",
                    (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1,
                )

        records.append(row)
        last_frame = frame.copy()

        if len(gaze_pts) > 10:
            heatmap      = generate_heatmap(frame, gaze_pts)
            last_overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
            cv2.imshow("Eye Tracker", last_overlay)
        else:
            cv2.imshow("Eye Tracker", frame)

        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not records:
        return

    ts_str = time.strftime("%Y%m%d_%H%M%S")
    df     = pd.DataFrame(records)

    csv_path = out / f"gaze_{ts_str}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(records)} frames → {csv_path}")

    _print_summary(df, tracker.fixations)

    if gaze_pts and last_frame is not None:
        heatmap = generate_heatmap(last_frame, gaze_pts)
        cv2.imwrite(str(out / f"heatmap_{ts_str}.jpg"), heatmap)
        if last_overlay is not None:
            cv2.imwrite(str(out / f"heatmap_overlay_{ts_str}.jpg"), last_overlay)
        print(f"Heatmap saved → {out}/heatmap_{ts_str}.jpg")

    aoi.print_summary()


if __name__ == "__main__":
    main()
