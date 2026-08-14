# Changelog

## [1.0.0] - 2026-08-14

- Added the initial release structure for MarScenes3D.
- Added 3D detection and tracking annotation readers.
- Added ASCII/binary PCD to NumPy conversion tools.
- Added model-agnostic 3D AP_R40 and TrackEval-based HOTA/MOTA/IDF1 evaluation.
- Added CAM_FRONT YOLOv8l and YOLOv10l baselines (`class 0 = vessel`) using
  Ultralytics 8.4.30.
- Added MIT code licensing, CC BY-NC 4.0 dataset terms, third-party notices,
  environment pins, and release documentation.
- Removed MCTrack/HybridTrack-derived configuration and motion code, cached
  bytecode, and non-portable compiled CUDA extensions.
- Added CPU-friendly OpenPCDet metadata/database preparation CLI and split
  optional dependency files.
- Removed the legacy KITTI detection evaluator, standalone DSVT example config,
  and viewer mesh assets with unverified provenance.
