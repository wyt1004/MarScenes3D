#!/usr/bin/env python3
"""Prepare an OpenPCDet-compatible MarScenes3D directory.

This command deliberately has no import from ``pcdet``.  It converts prepared
``points/<sample>.npy`` and ``labels/<sample>.txt`` files into the standard
OpenPCDet info and optional ground-truth database files.  Model training is
left to an external OpenPCDet-compatible repository.
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.marscenes3d_label import load_3d_detection_labels


CLASS_NAMES = (
    "large_vessel", "medium_vessel", "small_vessel",
    "sign", "piers", "lighthouse",
)


def _read_ids(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            # Split files may contain a relative path; OpenPCDet uses its stem.
            ids.append(Path(value).stem)
    return ids


def _discover_ids(root: Path) -> List[str]:
    return sorted(path.stem for path in (root / "labels").glob("*.txt"))


def _load_sample(root: Path, sample_id: str):
    point_path = root / "points" / f"{sample_id}.npy"
    label_path = root / "labels" / f"{sample_id}.txt"
    if not point_path.is_file():
        raise FileNotFoundError(point_path)
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    points = np.asarray(np.load(point_path), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"{point_path}: expected an N x 4+ NumPy array")
    labels = load_3d_detection_labels(label_path)
    names = np.asarray([label.class_name for label in labels])
    boxes = np.asarray([label.lidar_box for label in labels], dtype=np.float32).reshape(-1, 7)
    track_ids = np.asarray([label.track_id for label in labels], dtype=np.int64)
    point_counts = np.asarray([label.point_count for label in labels], dtype=np.int64)
    unknown = sorted(set(names.tolist()) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"{label_path}: unsupported classes: {unknown}")
    return points[:, :4], names, boxes, track_ids, point_counts


def _make_info(root: Path, sample_id: str):
    points, names, boxes, track_ids, point_counts = _load_sample(root, sample_id)
    return {
        "point_cloud": {"num_features": int(points.shape[1]), "lidar_idx": sample_id},
        "annos": {
            "name": names,
            "gt_boxes_lidar": boxes,
            "track_ids": track_ids,
            "num_points_in_gt": point_counts,
        },
    }


def _points_in_box(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    center = box[:3]
    dimensions = box[3:6]
    yaw = float(box[6])
    delta = points[:, :3] - center
    cosine, sine = np.cos(yaw), np.sin(yaw)
    local_x = delta[:, 0] * cosine + delta[:, 1] * sine
    local_y = -delta[:, 0] * sine + delta[:, 1] * cosine
    return (
        (np.abs(local_x) <= dimensions[0] / 2)
        & (np.abs(local_y) <= dimensions[1] / 2)
        & (np.abs(delta[:, 2]) <= dimensions[2] / 2)
    )


def _create_database(root: Path, infos: Iterable[dict], output_root: Path, split: str):
    database_root = output_root / ("gt_database" if split == "train" else f"gt_database_{split}")
    database_root.mkdir(parents=True, exist_ok=True)
    database_infos = {name: [] for name in CLASS_NAMES}
    for info in infos:
        sample_id = info["point_cloud"]["lidar_idx"]
        points = np.asarray(np.load(root / "points" / f"{sample_id}.npy"), dtype=np.float32)[:, :4]
        annos = info["annos"]
        for index, (name, box) in enumerate(zip(annos["name"], annos["gt_boxes_lidar"])):
            object_points = points[_points_in_box(points, box)].copy()
            object_points[:, :3] -= box[:3]
            filename = f"{sample_id}_{name}_{index}.bin"
            object_points.tofile(database_root / filename)
            database_infos[name].append({
                "name": str(name),
                "path": str((database_root / filename).relative_to(output_root)),
                "gt_idx": index,
                "box3d_lidar": np.asarray(box, dtype=np.float32),
                "num_points_in_gt": int(len(object_points)),
            })
    with (output_root / f"custom_dbinfos_{split}.pkl").open("wb") as stream:
        pickle.dump(database_infos, stream)


def prepare(root: Path, output_root: Path, splits, create_database: bool = False):
    root = root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    image_sets = output_root / "ImageSets"
    image_sets.mkdir(parents=True, exist_ok=True)
    for split, source in splits.items():
        ids = _read_ids(source) if source else _discover_ids(root)
        if not ids:
            raise ValueError(f"No samples found for split {split}")
        (image_sets / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        infos = [_make_info(root, sample_id) for sample_id in ids]
        with (output_root / f"custom_infos_{split}.pkl").open("wb") as stream:
            pickle.dump(infos, stream)
        if create_database and split == "train":
            _create_database(root, infos, output_root, split)
        print(f"Prepared {split}: {len(infos)} samples")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="Prepared root containing points/ and labels/")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="OpenPCDet data/custom output directory")
    parser.add_argument("--train-list", type=Path, default=None)
    parser.add_argument("--val-list", type=Path, default=None)
    parser.add_argument("--create-database", action="store_true",
                        help="Create gt_database_train and custom_dbinfos_train.pkl")
    args = parser.parse_args(argv)
    prepare(args.input_root, args.output_root, {
        "train": args.train_list,
        "val": args.val_list,
    }, create_database=args.create_database)


if __name__ == "__main__":
    main()
