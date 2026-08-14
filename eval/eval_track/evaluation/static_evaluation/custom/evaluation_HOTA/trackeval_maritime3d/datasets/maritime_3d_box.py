"""MarScenes3D 3D tracking dataset adapter for TrackEval.

The adapter is model agnostic.  It reads one text file per sequence and does
not import or depend on a tracker implementation.
"""

from pathlib import Path

import numpy as np

from ._base_dataset import _BaseDataset
from .. import _timing, utils
from ..iou_utils import iou3d_matrix_lwhxyzyaw
from ..utils import TrackEvalException


class Maritime3DBox(_BaseDataset):
    """Evaluate vessel tracks using 3D IoU in the LiDAR coordinate frame."""

    VALID_CLASS = "vessel"

    @staticmethod
    def get_default_dataset_config():
        code_path = Path(utils.get_code_path())
        return {
            "GT_FOLDER": str(code_path / "../gt"),
            "TRACKERS_FOLDER": None,
            "OUTPUT_FOLDER": None,
            "TRACKERS_TO_EVAL": None,
            "CLASSES_TO_EVAL": ["vessel"],
            "SPLIT_TO_EVAL": "val",
            "SEQMAP_FILE": None,
            "TRACKER_SUB_FOLDER": "data",
            "OUTPUT_SUB_FOLDER": "",
            "TRACKER_DISPLAY_NAMES": None,
            "INPUT_AS_ZIP": False,
            "PRINT_CONFIG": True,
        }

    def __init__(self, config=None):
        super().__init__()
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())
        if self.config["INPUT_AS_ZIP"]:
            raise TrackEvalException("MarScenes3D adapter accepts plain text files; unzip inputs first.")
        self.gt_fol = Path(self.config["GT_FOLDER"]).expanduser()
        self.tracker_fol = Path(self.config["TRACKERS_FOLDER"]).expanduser()
        if not self.gt_fol.is_dir():
            raise TrackEvalException("GT_FOLDER does not exist: %s" % self.gt_fol)
        if not self.tracker_fol.is_dir():
            raise TrackEvalException("TRACKERS_FOLDER does not exist: %s" % self.tracker_fol)

        classes = [str(value).lower() for value in self.config["CLASSES_TO_EVAL"]]
        if classes != [self.VALID_CLASS]:
            raise TrackEvalException("Only class 'vessel' (class 0 for YOLO) is supported.")
        self.class_list = [self.VALID_CLASS]
        self.class_name_to_class_id = {self.VALID_CLASS: 1}
        self.should_classes_combine = False
        self.use_super_categories = False
        self.output_fol = Path(self.config["OUTPUT_FOLDER"] or self.tracker_fol)
        self.output_sub_fol = self.config["OUTPUT_SUB_FOLDER"]
        self.tracker_sub_fol = str(self.config["TRACKER_SUB_FOLDER"]).strip("/")

        self.seq_list, self.seq_lengths = self._discover_sequences()
        requested = self.config.get("SEQUENCES_TO_EVAL")
        if requested:
            requested = [str(item) for item in requested]
            missing = sorted(set(requested) - set(self.seq_list))
            if missing:
                raise TrackEvalException("Requested sequence(s) not found: %s" % ", ".join(missing))
            self.seq_list = requested
        requested_trackers = self.config.get("TRACKERS_TO_EVAL")
        if requested_trackers:
            self.tracker_list = [str(item) for item in requested_trackers]
        else:
            self.tracker_list = sorted(
                item.name for item in self.tracker_fol.iterdir()
                if item.is_dir() and not item.name.startswith(".")
            )
        if not self.tracker_list:
            raise TrackEvalException("No tracker directories found in TRACKERS_FOLDER.")
        display_names = self.config.get("TRACKER_DISPLAY_NAMES")
        if display_names is None:
            display_names = self.tracker_list
        if len(display_names) != len(self.tracker_list):
            raise TrackEvalException("TRACKER_DISPLAY_NAMES must match TRACKERS_TO_EVAL.")
        self.tracker_to_disp = dict(zip(self.tracker_list, display_names))
        for tracker in self.tracker_list:
            for seq in self.seq_list:
                if not self._tracker_path(tracker, seq).is_file():
                    raise TrackEvalException(
                        "Tracker file not found: %s" % self._tracker_path(tracker, seq)
                    )

    def _sequence_path(self, seq):
        direct = self.gt_fol / (str(seq) + ".txt")
        legacy = self.gt_fol / "label_02" / (str(seq) + ".txt")
        if direct.is_file():
            return direct
        if legacy.is_file():
            return legacy
        return direct

    def _tracker_path(self, tracker, seq):
        base = self.tracker_fol / tracker
        if self.tracker_sub_fol:
            base /= self.tracker_sub_fol
        return base / (str(seq) + ".txt")

    def _discover_sequences(self):
        seqmap = self.config.get("SEQMAP_FILE")
        lengths = {}
        names = []
        if seqmap:
            seqmap_path = Path(seqmap).expanduser()
            if not seqmap_path.is_file():
                raise TrackEvalException("SEQMAP_FILE does not exist: %s" % seqmap_path)
            for line in seqmap_path.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if not fields or fields[0].startswith("#") or fields[0].lower() in {"name", "seq"}:
                    continue
                seq = fields[0]
                length = int(fields[-1]) if len(fields) > 1 and fields[-1].isdigit() else None
                names.append(seq)
                if length:
                    lengths[seq] = length
        else:
            source = self.gt_fol / "label_02" if (self.gt_fol / "label_02").is_dir() else self.gt_fol
            names = sorted(path.stem for path in source.glob("*.txt"))
        if not names:
            raise TrackEvalException("No sequence .txt files found in GT_FOLDER.")
        for seq in names:
            path = self._sequence_path(seq)
            if not path.is_file():
                raise TrackEvalException("GT file not found: %s" % path)
            if seq not in lengths:
                max_frame = -1
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip() and not line.lstrip().startswith("#"):
                        max_frame = max(max_frame, int(line.split()[0]))
                lengths[seq] = max_frame + 1
            if lengths[seq] < 1:
                raise TrackEvalException("Sequence %s has no frames; provide a SEQMAP_FILE length." % seq)
        return names, lengths

    @staticmethod
    def _parse_row(fields, line_number, source):
        if len(fields) not in (12, 13, 16, 17):
            raise TrackEvalException(
                "%s:%d must contain 12/13 compact or 16/17 expanded fields" % (source, line_number)
            )
        frame = int(fields[0])
        if frame < 0:
            raise TrackEvalException("%s:%d frame_id must be non-negative" % (source, line_number))
        # compact: frame, supercategory, class, id, dx,dy,dz,x,y,z,yaw,points[,score]
        offset = 4 if len(fields) in (12, 13) else 8
        track_id = int(fields[3])
        values = np.asarray([float(value) for value in fields[offset:offset + 7]], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values[:3] <= 0):
            raise TrackEvalException("%s:%d contains invalid box values" % (source, line_number))
        confidence = float(fields[-1]) if len(fields) in (13, 17) else 1.0
        if not np.isfinite(confidence):
            raise TrackEvalException("%s:%d confidence must be finite" % (source, line_number))
        return frame, str(fields[1]).lower(), track_id, values, confidence

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _load_raw_file(self, tracker, seq, is_gt):
        path = self._sequence_path(seq) if is_gt else self._tracker_path(tracker, seq)
        rows = {}
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            frame, supercategory, track_id, box, confidence = self._parse_row(
                line.split(), line_number, path
            )
            if supercategory != self.VALID_CLASS or track_id < 0:
                continue
            rows.setdefault(frame, []).append((track_id, box, confidence))
        num_timesteps = self.seq_lengths[seq]
        extra = sorted(frame for frame in rows if frame >= num_timesteps)
        if extra:
            raise TrackEvalException("%s contains frame(s) outside sequence %s: %s" % (path, seq, extra))
        ids = [np.asarray([item[0] for item in rows.get(t, [])], dtype=int) for t in range(num_timesteps)]
        dets = [np.asarray([item[1] for item in rows.get(t, [])], dtype=float).reshape(-1, 7)
                for t in range(num_timesteps)]
        classes = [np.ones(len(item), dtype=int) for item in ids]
        confidences = [np.asarray([item[2] for item in rows.get(t, [])], dtype=float) for t in range(num_timesteps)]
        if is_gt:
            return {
                "gt_ids": ids, "gt_classes": classes, "gt_dets": dets,
                "gt_crowd_ignore_regions": [np.empty((0, 7)) for _ in ids],
                "gt_extras": [{"occlusion": np.zeros(len(x)), "truncation": np.zeros(len(x))} for x in ids],
                "num_timesteps": num_timesteps, "seq": seq,
            }
        return {
            "tracker_ids": ids, "tracker_classes": classes, "tracker_dets": dets,
            "tracker_confidences": confidences, "num_timesteps": num_timesteps, "seq": seq,
        }

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        if cls != self.VALID_CLASS:
            raise TrackEvalException("Only class 'vessel' is supported.")
        data = {key: [] for key in (
            "gt_ids", "tracker_ids", "gt_dets", "tracker_dets", "tracker_confidences", "similarity_scores"
        )}
        gt_ids_all, tracker_ids_all = [], []
        for t in range(raw_data["num_timesteps"]):
            gt_ids = raw_data["gt_ids"][t]
            tracker_ids = raw_data["tracker_ids"][t]
            gt_dets = raw_data["gt_dets"][t]
            tracker_dets = raw_data["tracker_dets"][t]
            data["gt_ids"].append(gt_ids)
            data["tracker_ids"].append(tracker_ids)
            data["gt_dets"].append(gt_dets)
            data["tracker_dets"].append(tracker_dets)
            data["tracker_confidences"].append(raw_data["tracker_confidences"][t])
            data["similarity_scores"].append(self._calculate_similarities(gt_dets, tracker_dets))
            gt_ids_all.extend(gt_ids.tolist())
            tracker_ids_all.extend(tracker_ids.tolist())
        gt_unique = np.unique(gt_ids_all) if gt_ids_all else np.empty(0, dtype=int)
        tracker_unique = np.unique(tracker_ids_all) if tracker_ids_all else np.empty(0, dtype=int)
        gt_map = {value: i for i, value in enumerate(gt_unique)}
        tracker_map = {value: i for i, value in enumerate(tracker_unique)}
        for key, mapping in (("gt_ids", gt_map), ("tracker_ids", tracker_map)):
            data[key] = [np.asarray([mapping[value] for value in ids], dtype=int) for ids in data[key]]
        data.update({
            "num_gt_dets": sum(len(ids) for ids in data["gt_ids"]),
            "num_tracker_dets": sum(len(ids) for ids in data["tracker_ids"]),
            "num_gt_ids": len(gt_unique), "num_tracker_ids": len(tracker_unique),
            "num_timesteps": raw_data["num_timesteps"], "seq": raw_data["seq"],
        })
        self._check_unique_ids(data)
        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        return iou3d_matrix_lwhxyzyaw(gt_dets_t, tracker_dets_t)
