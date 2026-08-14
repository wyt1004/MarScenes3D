# Third-Party Notices

The MIT license in the repository root applies only to original MarScenes3D
code. Third-party components retain their own licenses. Their source notices
must be preserved when the corresponding files are redistributed.

| Component | Use in this repository | License / required action |
| --- | --- | --- |
| [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) | Dataset adapter conventions and optional CUDA/model integration | Apache-2.0; retain upstream copyright and license notices; see `LICENSES/Apache-2.0.txt` |
| [TrackEval](https://github.com/JonathonLuiten/TrackEval) | HOTA, CLEAR, Identity and Count evaluation | MIT; retained upstream revision `12c8791b303e0a0b50f753af204249e622d0281a` with MarScenes3D adaptations; license is in `eval/eval_track/evaluation/static_evaluation/custom/evaluation_HOTA/LICENSE` |
| [MCTrack](https://github.com/megvii-research/MCTrack) / [HybridTrack](https://github.com/leandro-svg/HybridTrack) | Optional external 3D tracking baselines | Not redistributed. Obtain code from the original repository and comply with its license/usage terms |
| Ultralytics 8.4.30 (YOLOv8l and YOLOv10l) | Optional 2D detection baselines | AGPL-3.0; installed as an external dependency only, not copied into this repository |

Some files in `eval/eval_track/evaluation/static_evaluation/` already contain
upstream license files. Those files remain authoritative for the bundled
component.

## Audit notes

- The retained TrackEval core files are under `trackeval_maritime3d/` and were
  compared with the upstream MIT repository at revision
  `12c8791b303e0a0b50f753af204249e622d0281a`.
- `_base_metric.py` and `count.py` are unchanged from that upstream revision.
  The evaluator, timing helper, dataset base class, HOTA, CLEAR, and Identity
  files contain MarScenes3D compatibility or output changes.
- `maritime_3d_box.py` and `iou_utils.py` are MarScenes3D adapter code and are
  not claimed as upstream TrackEval files.
- Other TrackEval benchmark adapters, segmentation metrics, scripts, example
  data, and generated logs are not redistributed.
- OpenPCDet, MCTrack, HybridTrack, and Ultralytics source code and weights are
  not copied into this repository. They remain external dependencies with
  their own license obligations.

The viewer uses line-box rendering by default and does not require external
vehicle mesh assets.
