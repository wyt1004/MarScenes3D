# Changelog

## [Unreleased]

- Repository cleanup and reproducibility work is being developed on the
  `temp` branch before the `v1.0` release branch is created.
- Added the initial release structure for MarScenes3D.
- Added 3D detection and tracking annotation readers.
- Added ASCII/binary PCD to NumPy conversion tools.
- Added model-agnostic 3D AP_R40 and TrackEval-based HOTA/MOTA/IDF1 evaluation.
- Added the CAM_FRONT YOLO26 dataset interface (`class 0 = vessel`).
- Added MIT code licensing, CC BY-NC 4.0 dataset terms, third-party notices,
  environment pins, and release documentation.
- Removed MCTrack/HybridTrack-derived configuration and motion code, cached
  bytecode, and non-portable compiled CUDA extensions.
- Added CPU-friendly OpenPCDet metadata/database preparation CLI and split
  optional dependency files.
- Removed the legacy KITTI detection evaluator, standalone DSVT example config,
  and viewer mesh assets with unverified provenance.

The `v1.0` release entry and release date will be added when the release branch
is created and the paper metadata is finalized.
