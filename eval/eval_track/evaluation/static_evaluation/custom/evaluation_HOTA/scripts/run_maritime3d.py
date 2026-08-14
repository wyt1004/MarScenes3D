"""Evaluate MarScenes3D 3D tracks with HOTA, CLEAR and Identity metrics.

Example::

    python run_maritime3d.py --gt-folder /data/Label/LabelTrack \
        --trackers-folder /results --tracker-name my_tracker \
        --split-folder /data/Task/Track/val

The tracker directory must contain ``data/<sequence>.txt`` files by default.
See ``docs/ANNOTATION_FORMATS.md`` for the row format.
"""

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import trackeval_maritime3d  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-folder", required=True, help="Directory containing one GT .txt per sequence")
    parser.add_argument("--trackers-folder", required=True, help="Directory containing tracker result directories")
    parser.add_argument("--tracker-name", required=True, help="Tracker directory name")
    parser.add_argument("--tracker-sub-folder", default="data", help="Result subdirectory (default: data)")
    parser.add_argument("--output-folder", default=None, help="Optional directory for summary files")
    parser.add_argument("--seqmap-file", default=None, help="Optional seqmap with sequence lengths")
    parser.add_argument(
        "--split-folder",
        default=None,
        help="Optional Task/Track/{split} folder; .txt stems select sequences from GT_FOLDER",
    )
    parser.add_argument("--sequences", nargs="+", default=None, help="Optional sequence names to evaluate")
    parser.add_argument("--split", default=None, help="Dataset split label for reproducibility metadata")
    parser.add_argument("--classes", nargs="+", default=["vessel"], choices=["vessel"])
    parser.add_argument("--metrics", nargs="+", default=["HOTA", "CLEAR", "Identity"],
                        choices=["HOTA", "CLEAR", "Identity", "Count"])
    parser.add_argument("--no-plots", action="store_true", help="Do not write metric curve plots")
    parser.add_argument("--quiet", action="store_true", help="Suppress TrackEval tables")
    return parser


def eval_maritime3d(args):
    sequences = getattr(args, "sequences", None)
    split_folder_arg = getattr(args, "split_folder", None)
    if split_folder_arg:
        split_folder = Path(split_folder_arg).expanduser()
        if not split_folder.is_dir():
            raise FileNotFoundError(f"split folder does not exist: {split_folder}")
        folder_sequences = sorted(path.stem for path in split_folder.glob("*.txt"))
        if not folder_sequences:
            raise ValueError(f"no sequence .txt files found in split folder: {split_folder}")
        if sequences:
            sequences = sorted(set(sequences).intersection(folder_sequences))
            if not sequences:
                raise ValueError("--sequences and --split-folder have no common sequence")
        else:
            sequences = folder_sequences
    dataset_config = {
        "GT_FOLDER": args.gt_folder,
        "TRACKERS_FOLDER": args.trackers_folder,
        "TRACKERS_TO_EVAL": [args.tracker_name],
        "TRACKER_SUB_FOLDER": args.tracker_sub_folder,
        "OUTPUT_FOLDER": args.output_folder,
        "SEQMAP_FILE": args.seqmap_file,
        "SEQUENCES_TO_EVAL": sequences,
        "CLASSES_TO_EVAL": args.classes,
        "SPLIT_TO_EVAL": args.split or "unspecified",
    }
    dataset = trackeval_maritime3d.datasets.Maritime3DBox(dataset_config)
    eval_config = trackeval_maritime3d.Evaluator.get_default_eval_config()
    eval_config.update({
        "PRINT_RESULTS": not args.quiet,
        "PRINT_ONLY_COMBINED": True,
        "PLOT_CURVES": not args.no_plots,
        "BREAK_ON_ERROR": True,
    })
    evaluator = trackeval_maritime3d.Evaluator(eval_config)
    metric_classes = {
        "HOTA": trackeval_maritime3d.metrics.HOTA,
        "CLEAR": trackeval_maritime3d.metrics.CLEAR,
        "Identity": trackeval_maritime3d.metrics.Identity,
        "Count": trackeval_maritime3d.metrics.Count,
    }
    metrics = [metric_classes[name]() for name in args.metrics]
    return evaluator.evaluate([dataset], metrics)


def main(argv=None):
    return eval_maritime3d(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
