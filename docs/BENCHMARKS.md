# Baseline Results

This file provides placeholders for the paper results. Replace `TBD` values
only after the corresponding experiment, configuration, and evaluation output
has been verified.

## 3D detection

| Model | Split | AP / mAP | Config | Checkpoint |
| --- | --- | --- | --- | --- |
| TBD (external OpenPCDet-compatible model) | TBD | TBD | TBD | TBD |

Record the OpenPCDet/model commit, per-class metrics, IoU `0.5` threshold,
random seed, GPU model, number of GPUs, epochs, batch size, and training time
when the paper results are available.

## 3D tracking

| Tracker (external) | Detector (external) | Split | HOTA | MOTA | IDF1 | Command |
| --- | --- | --- | --- | --- | --- | --- |
| TBD | TBD | TBD | TBD | TBD | TBD | `run_maritime3d.py` |

Record the external tracker commit, input detection format, confidence
threshold, sequence list, frame rate, and whether the run is online or global.

## 2D detection

| Model | Camera(s) | Split | mAP50 | mAP50-95 | Config / command |
| --- | --- | --- | --- | --- | --- |
| YOLOv8l (Ultralytics 8.4.30) | CAM_FRONT | TBD | TBD | TBD | `yolo detect train model=yolov8l.pt data=tools/cfgs/dataset_configs/marscenes3d_yolo.yaml` |
| YOLOv10l (Ultralytics 8.4.30) | CAM_FRONT | TBD | TBD | TBD | `yolo detect train model=yolov10l.pt data=tools/cfgs/dataset_configs/marscenes3d_yolo.yaml` |

Record the exact Ultralytics version, pretrained weights, image size, epochs,
batch size, augmentation settings, random seed, and per-class metrics.
