# MarScenes3D

MarScenes3D is a multimodal maritime perception dataset for 3D detection, 3D
multi-object tracking, and image-based 2D detection. It contains synchronized
LiDAR point clouds, six-view images, calibration, platform pose, velocity, and
identity-preserving annotations collected in real maritime environments.

The dataset is available from [Science Data Bank](https://doi.org/10.57760/sciencedb.35872).
The code in this repository is
released under the MIT license. The dataset is released separately under
[CC BY-NC 4.0](DATA_LICENSE.md); third-party components retain their original
licenses listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Supported tasks

| Task | Input | Output | Main code |
| --- | --- | --- | --- |
| 3D detection | LiDAR point cloud | 3D boxes and classes | `adapters/openpcdet/`, `tools/prepare_openpcdet.py`, `tools/evaluate_3d_detection.py` |
| 3D tracking | Per-frame 3D detections | Boxes with persistent IDs | `tools/3D_detection_track_viewer/`, `eval/eval_track/` |
| 2D detection | Camera image | YOLO 2D boxes and classes | `tools/cfgs/dataset_configs/marscenes3d_yolo.yaml` |

This repository does not define a 2D tracker. The 2D task is detection only;
3D tracking IDs are evaluated separately.

## Repository status

This repository is the MarScenes3D `v1.0` release. Model implementations and
checkpoints are intentionally external; record their exact upstream commit,
package version, and command in `docs/BENCHMARKS.md`.

## Installation

The core tools are CPU-friendly and are tested with Ubuntu 20.04, Python 3.10,
NumPy 1.24.0, and SciPy 1.14.1. PyTorch, CUDA, and spconv are only needed for
an external OpenPCDet model environment. See
[docs/INSTALL.md](docs/INSTALL.md) and [requirements.txt](requirements.txt).

```bash
conda create -n marscenes3d python=3.10 -y
conda activate marscenes3d
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This is a dataset adapter and evaluation repository, not a standalone model
implementation. Install an external OpenPCDet-compatible repository only when
you need model training or inference. No precompiled CUDA extension is shipped.

## Dataset organization

The raw archive and prepared task-specific layouts are documented in
[docs/DATASET.md](docs/DATASET.md). The raw archive contains:

```text
Data/{Calibs,PCD,Pose,Images}
Task/{Detection2D,Detection3D,Track}
Label/{Label2D,Label3D,LabelTrack,Language}
```

The exact annotation fields are defined in
[docs/ANNOTATION_FORMATS.md](docs/ANNOTATION_FORMATS.md).

## 3D detection

Prepare an OpenPCDet-compatible dataset under `data/custom/` as described in
[docs/DATASET.md](docs/DATASET.md). The repository supplies the data/evaluation
adapter and preparation command; model implementations and training scripts
remain in the user's chosen public OpenPCDet-compatible repository:

For the raw ASCII/binary PCD files, convert point clouds first:

```bash
python tools/data/convert_pcd_to_npy.py \
  /path/to/MarScenes3D/Data/PCD \
  /path/to/MarScenes3D/data/custom/points
```

```bash
python tools/prepare_openpcdet.py \
  --input-root /path/to/prepared/custom \
  --output-root /path/to/MarScenes3D/data/custom \
  --train-list /path/to/MarScenes3D/Task/Detection3D/train.txt \
  --val-list /path/to/MarScenes3D/Task/Detection3D/val.txt \
  --create-database
```

The preparation command is CPU-friendly and does not import `pcdet`. Training
and inference stay in the external model repository; this repository provides
the data preparation and evaluation adapters.

For model-agnostic text results, append a confidence score to every 15-field
prediction row and run:

```bash
python tools/evaluate_3d_detection.py \
  --gt-folder /path/to/Label/Label3D \
  --pred-folder /path/to/detection/results \
  --split-file /path/to/MarScenes3D/Task/Detection3D/val.txt
```

This reports per-class 3D AP_R40 at IoU `0.5` for every class. The current
benchmark values in `docs/BENCHMARKS.md` are intentionally `TBD` placeholders
until the paper experiments are finalized.

## 3D tracking

This repository provides generic 3D tracking input and TrackEval-based static
metrics. It does not redistribute MCTrack, HybridTrack, or another tracker.
Use a tracker from its original repository, export one result file per
sequence, and evaluate it with:

```bash
python eval/eval_track/evaluation/static_evaluation/custom/evaluation_HOTA/scripts/run_maritime3d.py \
  --gt-folder /path/to/MarScenes3D/Label/LabelTrack \
  --trackers-folder /path/to/results \
  --tracker-name my_tracker \
  --split-folder /path/to/MarScenes3D/Task/Track/val \
  --split val
```

The expected result layout is `/path/to/results/my_tracker/data/<sequence>.txt`.
The evaluator reports HOTA, CLEAR/MOTA, and Identity/IDF1 for `vessel`.

To inspect a prepared sequence after setting its data and result paths:

```bash
python tools/3D_detection_track_viewer/maritime3d_viewer.py \
  --root /path/to/prepared/marscenes3d_viewer \
  --seq-id 0 \
  --label-path /path/to/tracking/results.txt
```

The viewer currently expects its prepared `points/`, `image/`, `calib/`, and
`label/` layout; it does not read the raw archive tree directly.

The tracking annotation format and evaluation workflow are described in
[docs/ANNOTATION_FORMATS.md](docs/ANNOTATION_FORMATS.md) and
[docs/BASELINES.md](docs/BASELINES.md). The repository does not ship a copy of
the full dataset or tracking ground truth.

## 2D detection

The supplied 2D labels use normalized YOLO format for the front camera only
(`CAM_FRONT`). There is one class, `0 = vessel`. Convert images and labels to
the layout described in `docs/DATASET.md`, update the dataset `path`, and run
Ultralytics 8.4.117 (the latest release checked on 2026-08-10):

```bash
python tools/prepare_yolo.py \
  --dataset-root /path/to/MarScenes3D \
  --output-root /path/to/MarScenes3D/data/marscenes3d_yolo
```

```bash
yolo detect train \
  model=yolo26m.pt \
  data=tools/cfgs/dataset_configs/marscenes3d_yolo.yaml
```

Ultralytics is an optional external dependency under AGPL-3.0. Do not copy its
source or weights into this MIT-licensed repository. Record image size, epochs,
split, seed, and per-class metrics in `docs/BENCHMARKS.md`.


## Citation

See [CITATION.cff](CITATION.cff). Please cite the MarScenes3D dataset DOI and
the accompanying paper when using the data or code.

## License and acknowledgements

Original code is MIT licensed. Dataset terms are in [DATA_LICENSE.md](DATA_LICENSE.md).
This project builds on OpenPCDet and TrackEval; optional model repositories are
listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
