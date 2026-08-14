#!/usr/bin/env python3
"""Prepare the CAM_FRONT YOLO layout from a MarScenes3D archive."""

import argparse
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.marscenes3d_label import load_yolo_labels  # noqa: E402


def _read_ids(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            ids.append(Path(value).stem)
    return ids


def _link_or_copy(source: Path, target: Path, mode: str, overwrite: bool):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Output exists (use --overwrite): {target}")
        target.unlink()
    if mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        shutil.copy2(source, target)


def prepare(dataset_root: Path, output_root: Path, mode="symlink", overwrite=False):
    dataset_root = dataset_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    image_root = dataset_root / "Data" / "Images" / "CAM_FRONT"
    label_root = dataset_root / "Label" / "Label2D"
    split_root = dataset_root / "Task" / "Detection2D"
    if not image_root.is_dir() or not label_root.is_dir() or not split_root.is_dir():
        raise FileNotFoundError(
            "Expected Data/Images/CAM_FRONT, Label/Label2D, and Task/Detection2D "
            f"under {dataset_root}"
        )

    counts = {}
    for split in ("train", "val"):
        ids = _read_ids(split_root / f"{split}.txt")
        if not ids:
            raise ValueError(f"No samples listed in {split_root / f'{split}.txt'}")
        for sample_id in ids:
            image = image_root / f"{sample_id}.png"
            label = label_root / f"{sample_id}.txt"
            if not image.is_file():
                raise FileNotFoundError(image)
            if not label.is_file():
                raise FileNotFoundError(label)
            load_yolo_labels(label)
            _link_or_copy(image, output_root / "images" / split / image.name, mode, overwrite)
            _link_or_copy(label, output_root / "labels" / split / label.name, mode, overwrite)
        counts[split] = len(ids)
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    counts = prepare(args.dataset_root, args.output_root, args.mode, args.overwrite)
    print("Prepared YOLO CAM_FRONT layout: " + ", ".join(
        f"{split}={count}" for split, count in counts.items()
    ))


if __name__ == "__main__":
    main()
