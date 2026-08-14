# Baseline Workflows

## 3D detection

[OpenPCDet](https://github.com/open-mmlab/OpenPCDet) conventions.
The provided `adapters/openpcdet/` dataset adapter and configuration follow
those conventions.
OpenPCDet itself, detector implementations, and checkpoints are external
dependencies. A reproducible baseline records the selected upstream commit,
the MarScenes3D conversion command, and a downloadable checkpoint.

Expected stages are data preparation, external-model training/inference,
evaluation, and optional visualization. Data preparation is
`tools/prepare_openpcdet.py`; detection evaluation is
`tools/evaluate_3d_detection.py`.

## 3D tracking

The repository supplies only a model-agnostic TrackEval adapter. Trackers such
as [MCTrack](https://github.com/megvii-research/MCTrack) or
[HybridTrack](https://github.com/leandro-svg/HybridTrack) must be obtained from
their original repositories and are not included here. Export their per-frame
predictions in the documented MarScenes3D format, then use
`run_maritime3d.py`.

The bundled `eval/eval_track/.../trackeval_maritime3d` directory is the only
tracking evaluator intended for this dataset. Legacy KITTI evaluation code is
not part of the MarScenes3D workflow.

## 2D detection

The optional image baselines use Ultralytics `8.4.30` with YOLOv8l and
YOLOv10l detection weights:

```bash
yolo detect train model=yolov8l.pt data=tools/cfgs/dataset_configs/marscenes3d_yolo.yaml
yolo detect train model=yolov10l.pt data=tools/cfgs/dataset_configs/marscenes3d_yolo.yaml
```

Ultralytics is an external AGPL-3.0 dependency. Record the model size and
hyperparameters used for paper results in `docs/BENCHMARKS.md`.
