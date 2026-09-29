# Real-clip gaze-zone dataset

Labelled segments of public-domain video, used to evaluate the attention-zone
classifier on real faces (it is otherwise trained only on synthetic data).
`manifest.csv` lists every segment; the videos themselves are not stored in
the repo.

```bash
python scripts/evaluate_zones.py          # downloads what it needs into data/clips/
```

## Labels

The camera stands in for the screen: a person looking into the lens is
treated as looking at the screen.

| Label | Meaning | Typical footage |
|---|---|---|
| `on_screen` | Gaze into the lens | Speaker addressing the camera |
| `peripheral` | Gaze just off the lens, roughly 15–30° | Interviewee answering someone beside the camera |
| `away` | Gaze well off the lens, roughly 40°+ | Head turned far aside, profile, looking at other things |

Labels are per segment, assigned by visually reviewing frames sampled every
3 seconds. Segment windows were then trimmed so that every frame of an
`on_screen` or `peripheral` segment shows the subject's face (cutaways to
titles, graphics and wide shots removed, verified frame by frame).
**All labels are unreviewed by a second person** (see `notes`).

## Composition

| Label | People | Segments | Frames at 5 fps |
|---|---|---|---|
| `on_screen` | 8 | 16 | 2398 |
| `peripheral` | 4 | 9 | 594 |
| `away` | 2 sources | 11 | 173 |

13 identified speakers (4 women) plus one B-roll source showing several
unnamed people; two speakers were interviewed in Spanish. Several wear
glasses. Lighting ranges from studio to an overexposed laptop webcam on
the ISS.

## Known limitations

- **`away` is thin.** One person (head turned ~40° with eyes further aside)
  plus short B-roll segments of people in profile. MediaPipe usually does not
  detect a face in profile, so most B-roll `away` frames have no face; the
  evaluation scores the system by treating "no face" as `away`.
- **Single annotator, coarse labels.** Segment-level labels can't capture
  brief glances within a segment, and the 15–30° / 40°+ boundaries were
  judged by eye, not measured.
- **Camera ≠ screen.** Real users look at a screen near, not at, the camera;
  on-screen gaze in deployment sits a few degrees off-lens.
- **Frames within a segment are strongly correlated.** Any train/test split
  must be by `subject`, never by frame.
- **Not a benchmark.** Broadcast NASA footage (good lighting, cooperative
  speakers) is easier than typical webcam use.

## Sources and licence

Every source is a NASA video on Wikimedia Commons, public domain as a work of
the U.S. federal government; `source_page` links to each file's Commons page
and `license` records its status. `load_manifest` rejects any licence other
than public domain or CC0. Inclusion does not imply endorsement by NASA or
the people shown.
