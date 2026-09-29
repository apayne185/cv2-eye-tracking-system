"""
Evaluate the attention-zone classifier on the labelled real-clip dataset.

    python scripts/evaluate_zones.py [--manifest ...] [--report reports/zone_eval.md]

Downloads only the parts of each source video the manifest needs (cached in
data/clips/), runs the pipeline over every segment, and writes a markdown
report comparing synthetic-only and subject-held-out real-data models.
"""

import argparse
import logging
from pathlib import Path

from eyetrack.dataset import extract_features, fetch, load_manifest
from eyetrack.evaluation import render_report, run_all, train_final
from eyetrack.logs import setup_logging

log = logging.getLogger("evaluate_zones")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--manifest", type=Path, default=Path("datasets/real_clips/manifest.csv"))
    p.add_argument("--cache", type=Path, default=Path("data/clips"))
    p.add_argument("--fps", type=float, default=5.0, help="frames sampled per second of video")
    p.add_argument("--features", type=Path, default=Path("data/real_clip_features.csv"),
                   help="where to save extracted per-frame features")
    p.add_argument("--report", type=Path, default=Path("reports/zone_eval.md"))
    p.add_argument("--save-model", type=Path, metavar="PATH",
                   help="also train the real + mirrored model on every subject and save it "
                        "(e.g. models/gaze_zone_classifier.joblib)")
    args = p.parse_args()
    setup_logging()

    segments = load_manifest(args.manifest)
    log.info("%d segments from %s", len(segments), args.manifest)
    paths = fetch(segments, args.cache)
    df = extract_features(segments, paths, sample_fps=args.fps)
    args.features.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.features, index=False)
    log.info("%d frames extracted; features saved to %s", len(df), args.features)

    results = run_all(df)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(results, df))
    for r in results:
        log.info("%-28s system macro-F1 %.3f  accuracy %.3f", r.name, r.system.macro_f1, r.system.accuracy)
    log.info("report written to %s", args.report)

    if args.save_model:
        path = train_final(df).save(args.save_model)
        log.info("real + mirrored model trained on all %d subjects saved to %s",
                 df["subject"].nunique(), path)


if __name__ == "__main__":
    main()
