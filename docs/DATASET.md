# Dataset Organization

Download MarScenes3D from [Science Data Bank](https://doi.org/10.57760/sciencedb.35872).
The raw archive separates sensor
data, task splits, and labels:

```text
MarScenes3D/
|-- Data/
|   |-- Calibs/                 # per-frame calibration
|   |-- PCD/                    # point clouds
|   |-- Pose/                   # platform pose
|   `-- Images/
|       |-- CAM_FRONT/
|       |-- CAM_FRONT_LEFT/
|       |-- CAM_FRONT_RIGHT/
|       |-- CAM_BACK/
|       |-- CAM_BACK_LEFT/
|       `-- CAM_BACK_RIGHT/
|-- Task/
|   |-- Detection2D/           # train.txt and val.txt
|   |-- Detection3D/           # train.txt and val.txt
|   `-- Track/{train,val}/     # one <sequence>.txt selector per sequence
`-- Label/
    |-- Label2D/
    |-- Label3D/
    |-- LabelTrack/
    `-- Language/
```

## OpenPCDet preparation

The OpenPCDet adapter in `adapters/openpcdet/` reads NumPy point arrays and
per-frame labels:

```text
data/custom/
|-- ImageSets/
|   |-- train.txt
|   `-- val.txt
|-- points/
|   `-- <sample_id>.npy         # N x 4: x, y, z, intensity
|-- labels/
|   `-- <sample_id>.txt
|-- custom_infos_train.pkl
|-- custom_infos_val.pkl
`-- custom_dbinfos_train.pkl
```

For the standard ASCII or binary PCD files supplied with MarScenes3D, the
included converter preserves the
fields without axis reordering:

```bash
python tools/data/convert_pcd_to_npy.py \
  /path/to/MarScenes3D/Data/PCD \
  /path/to/MarScenes3D/data/custom/points
```

After conversion, place the matching 3D label files in `labels/` and generate
OpenPCDet metadata with:

```bash
python tools/prepare_openpcdet.py \
  --input-root /path/to/prepared/custom \
  --output-root /path/to/MarScenes3D/data/custom \
  --train-list /path/to/MarScenes3D/Task/Detection3D/train.txt \
  --val-list /path/to/MarScenes3D/Task/Detection3D/val.txt \
  --create-database
```

The input root must contain `points/<sample_id>.npy` and
`labels/<sample_id>.txt`. The command writes `ImageSets/`, info pickle files,
and the optional ground-truth database without requiring a GPU or model code.
Use `adapters/openpcdet/marscenes3d_dataset.yaml` when registering the adapter
in an external OpenPCDet-compatible project.

The coordinate convention is `X` right, `Y` forward, and `Z` up. Each output
row is `[x, y, z, intensity]` in float32. `binary_compressed` PCD requires
exporting to ASCII or standard binary first.

## 3D tracking evaluation layout

The generic evaluator reads one ground-truth and one prediction file per
sequence. Sequence names are matched by filename:

```text
tracking_gt/val/
|-- 0.txt
`-- 1.txt

tracking_results/
`-- my_tracker/
    `-- data/
        |-- 0.txt
        `-- 1.txt
```

Frame count is inferred as `max(frame_id) + 1`. For a sequence containing no
annotations, pass `--seqmap-file`; each non-comment row is `sequence length`.
The split is supplied as metadata with `--split`. Use `--split-folder` to
select the sequence stems listed under `Task/Track/train` or `Task/Track/val`:

```bash
python eval/eval_track/evaluation/static_evaluation/custom/evaluation_HOTA/scripts/run_maritime3d.py \
  --gt-folder /path/to/MarScenes3D/Label/LabelTrack \
  --trackers-folder /path/to/results \
  --tracker-name my_tracker \
  --split-folder /path/to/MarScenes3D/Task/Track/val \
  --split val
```

## YOLO preparation

The 2D detection task uses the `CAM_FRONT` image stream and one class (`0 =
vessel`) in the standard Ultralytics layout:

Prepare that layout directly from the archive:

```bash
python tools/prepare_yolo.py \
  --dataset-root /path/to/MarScenes3D \
  --output-root /path/to/MarScenes3D/data/marscenes3d_yolo
```

```text
data/marscenes3d_yolo/
|-- images/{train,val,test}/
`-- labels/{train,val,test}/
```

