# MarScenes3D Track Evaluation

This directory contains the MIT-licensed TrackEval components needed by the
MarScenes3D 3D tracking benchmark. The bundled dataset adapter evaluates the
single `vessel` class with 3D LiDAR-box IoU.

The included metrics are:

- HOTA with the standard alpha thresholds from the TrackEval reference
  implementation;
- CLEAR metrics, including MOTA, with similarity threshold `0.5`;
- Identity metrics, including IDF1, with similarity threshold `0.5`;
- Count summaries.

Run the dataset-specific command from the repository root:

```bash
python eval/eval_track/evaluation/static_evaluation/custom/evaluation_HOTA/scripts/run_maritime3d.py \
  --gt-folder /path/to/MarScenes3D/Label/LabelTrack \
  --trackers-folder /path/to/results \
  --tracker-name my_tracker \
  --split-folder /path/to/MarScenes3D/Task/Track/val \
  --split val
```

The expected tracker layout is `results/<tracker>/data/<sequence>.txt`.
Input rows and output files are documented in the repository-level
`docs/ANNOTATION_FORMATS.md`.

The evaluator is adapted from [TrackEval](https://github.com/JonathonLuiten/TrackEval).
The upstream MIT license is retained in this directory. Other TrackEval
benchmarks, segmentation metrics, and example data are intentionally not
included in this release.
