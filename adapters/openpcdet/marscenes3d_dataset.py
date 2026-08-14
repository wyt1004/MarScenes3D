"""OpenPCDet dataset adapter for MarScenes3D.

This module imports only the public OpenPCDet dataset base class. It does not
vendor a model, CUDA operator, augmentation implementation, or checkpoint.
"""

import copy
import pickle
from pathlib import Path

import numpy as np

from utils.marscenes3d_detection_eval import evaluate_openpcdet_annos, format_results
from utils.marscenes3d_label import load_3d_detection_labels

try:
    from pcdet.datasets.dataset import DatasetTemplate as _DatasetTemplate
except ImportError:  # Keep lightweight preparation tools importable on CPU-only hosts.
    class _DatasetTemplate:
        def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
            self.dataset_cfg = dataset_cfg
            self.class_names = class_names
            self.training = training
            self.mode = "train" if training else "test"
            self.root_path = root_path
            self.logger = logger


class MarScenes3DDataset(_DatasetTemplate):
    """Dataset class to be mixed into an OpenPCDet checkout.

    The class is intentionally defined without importing ``pcdet`` at module
    import time, so annotation tools remain usable without a GPU environment.
    ``bind_to_openpcdet`` creates the concrete subclass when OpenPCDet exists.
    """

    CLASS_NAMES = (
        "large_vessel", "medium_vessel", "small_vessel",
        "sign", "piers", "lighthouse",
    )

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_path = Path(root_path or self.dataset_cfg.DATA_PATH)
        split_file = self.root_path / "ImageSets" / f"{self.split}.txt"
        self.sample_id_list = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines()
                               if line.strip()] if split_file.is_file() else []
        self.custom_infos = []
        self.include_data(self.mode)

    def include_data(self, mode):
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            path = self.root_path / info_path
            if path.is_file():
                with path.open("rb") as stream:
                    self.custom_infos.extend(pickle.load(stream))
        if self.logger:
            self.logger.info("Total MarScenes3D samples: %d", len(self.custom_infos))

    def get_label(self, sample_id):
        path = self.root_path / "labels" / f"{sample_id}.txt"
        labels = load_3d_detection_labels(path)
        boxes = np.asarray([label.lidar_box for label in labels], dtype=np.float32).reshape(-1, 7)
        names = np.asarray([label.class_name for label in labels])
        return boxes, names

    def get_lidar(self, sample_id):
        path = self.root_path / "points" / f"{sample_id}.npy"
        points = np.asarray(np.load(path), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 4:
            raise ValueError(f"Expected N x 4+ points in {path}, got {points.shape}")
        return points[:, :4]

    def set_split(self, split):
        self.split = split
        split_file = self.root_path / "ImageSets" / f"{split}.txt"
        self.sample_id_list = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines()
                               if line.strip()]
        self.custom_infos = []
        self.include_data(self.mode)

    def __len__(self):
        return len(self.custom_infos)

    def __getitem__(self, index):
        info = copy.deepcopy(self.custom_infos[index])
        sample_id = info["point_cloud"]["lidar_idx"]
        points = self.get_lidar(sample_id)
        input_dict = {"frame_id": sample_id, "points": points}
        if "annos" in info:
            input_dict.update({
                "gt_names": info["annos"]["name"],
                "gt_boxes": info["annos"]["gt_boxes_lidar"],
            })
        return self.prepare_data(data_dict=input_dict)

    def evaluation(self, det_annos, class_names, **kwargs):
        if not self.custom_infos or "annos" not in self.custom_infos[0]:
            return "No ground-truth boxes for evaluation", {}
        gt_annos = [copy.deepcopy(info["annos"]) for info in self.custom_infos]
        result = evaluate_openpcdet_annos(gt_annos, det_annos, class_names)
        return format_results(result)

    def get_infos(self, class_names=None, num_workers=1, has_label=True, sample_id_list=None,
                  num_features=4):
        sample_ids = sample_id_list or self.sample_id_list
        infos = []
        for sample_id in sample_ids:
            info = {"point_cloud": {"num_features": num_features, "lidar_idx": sample_id}}
            if has_label:
                boxes, names = self.get_label(sample_id)
                info["annos"] = {"name": names, "gt_boxes_lidar": boxes}
            infos.append(info)
        return infos


def bind_to_openpcdet():
    """Return the adapter class using the installed OpenPCDet base class.

    The fallback base class only supports importing the module without
    OpenPCDet. Training still requires the external OpenPCDet installation.
    """
    if _DatasetTemplate.__module__ == __name__:
        raise RuntimeError(
            "OpenPCDet is not installed. Install it externally before binding "
            "MarScenes3DDataset to a training project."
        )
    return MarScenes3DDataset
