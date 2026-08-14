"""Evaluate model-agnostic MarScenes3D 3D detection text results."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.marscenes3d_detection_eval import (  # noqa: E402
    CLASS_NAMES,
    evaluate_detection,
    format_results,
)
from utils.marscenes3d_label import (  # noqa: E402
    load_3d_detection_labels,
    load_3d_predictions,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-folder", required=True, help="Ground-truth 15-field label directory")
    parser.add_argument("--pred-folder", required=True, help="Prediction directory (same stems, score appended)")
    parser.add_argument("--split-file", default=None, help="Optional file listing sample stems")
    parser.add_argument("--classes", nargs="+", choices=CLASS_NAMES, default=list(CLASS_NAMES))
    parser.add_argument("--json-output", default=None, help="Optional path for machine-readable results")
    return parser


def _sample_ids(gt_folder, split_file):
    if split_file:
        return [line.strip() for line in Path(split_file).read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    return sorted(path.stem for path in gt_folder.glob("*.txt"))


def main(argv=None):
    args = build_parser().parse_args(argv)
    gt_folder = Path(args.gt_folder)
    pred_folder = Path(args.pred_folder)
    if not gt_folder.is_dir() or not pred_folder.is_dir():
        raise FileNotFoundError("--gt-folder and --pred-folder must be directories")
    ground_truth, predictions = [], []
    sample_ids = _sample_ids(gt_folder, args.split_file)
    if not sample_ids:
        raise ValueError("no ground-truth label files found")
    for sample_id in sample_ids:
        gt_path = gt_folder / f"{sample_id}.txt"
        if not gt_path.is_file():
            raise FileNotFoundError(f"ground-truth label not found: {gt_path}")
        for label in load_3d_detection_labels(gt_path):
            ground_truth.append((sample_id, label.class_name, label.lidar_box))
        pred_path = pred_folder / f"{sample_id}.txt"
        if pred_path.is_file():
            for prediction in load_3d_predictions(pred_path):
                predictions.append((sample_id, prediction.class_name,
                                    prediction.lidar_box, prediction.confidence))
    result = evaluate_detection(ground_truth, predictions, args.classes)
    report, _ = format_results(result)
    print(report)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
