"""Parsers for the public MarScenes3D annotation formats."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union


PathLike = Union[str, Path]


@dataclass(frozen=True)
class Box3DLabel:
    supercategory: str
    class_name: str
    track_id: int
    dimensions: Tuple[float, float, float]
    center: Tuple[float, float, float]
    yaw: float
    point_count: int
    frame_id: Optional[int] = None

    @property
    def lidar_box(self) -> Tuple[float, float, float, float, float, float, float]:
        """Return ``(x, y, z, dx, dy, dz, yaw)`` for OpenPCDet."""
        return (*self.center, *self.dimensions, self.yaw)


@dataclass(frozen=True)
class YoloLabel:
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass(frozen=True)
class Box3DPrediction:
    label: Box3DLabel
    confidence: float

    @property
    def class_name(self) -> str:
        return self.label.class_name

    @property
    def lidar_box(self) -> Tuple[float, float, float, float, float, float, float]:
        return self.label.lidar_box


def _finite_float(value: str, field: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return parsed


def _integer(value: str, field: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc


def _parse_box_fields(fields: Sequence[str], frame_id: Optional[int]) -> Box3DLabel:
    if len(fields) != 15:
        raise ValueError(
            "a 3D detection annotation must contain 15 fields: "
            "supercategory class track_id attr0 attr1 attr2 attr3 "
            "dx dy dz x y z yaw point_count"
        )

    dimensions = tuple(
        _finite_float(value, name)
        for value, name in zip(fields[7:10], ("dx", "dy", "dz"))
    )
    if any(value <= 0 for value in dimensions):
        raise ValueError(f"box dimensions must be positive, got {dimensions}")

    center = tuple(
        _finite_float(value, name)
        for value, name in zip(fields[10:13], ("x", "y", "z"))
    )
    return Box3DLabel(
        supercategory=fields[0],
        class_name=fields[1],
        track_id=_integer(fields[2], "track_id"),
        dimensions=dimensions,
        center=center,
        yaw=_finite_float(fields[13], "yaw"),
        point_count=_integer(fields[14], "point_count"),
        frame_id=frame_id,
    )


def parse_3d_detection_line(line: str) -> Box3DLabel:
    """Parse one 3D detection annotation line."""
    return _parse_box_fields(line.split(), frame_id=None)


def parse_3d_tracking_line(line: str) -> Box3DLabel:
    """Parse a compact 12-field or expanded 16-field 3D tracking row."""
    fields = line.split()
    if len(fields) == 12:
        detection_fields = [*fields[1:4], "None", "None", "None", "None", *fields[4:]]
    elif len(fields) == 16:
        detection_fields = fields[1:]
    else:
        raise ValueError(
            "a 3D tracking annotation must contain 12 compact fields or 16 "
            "fields including the four optional attributes"
        )
    return _parse_box_fields(detection_fields, frame_id=_integer(fields[0], "frame_id"))


def parse_yolo_line(line: str) -> YoloLabel:
    """Parse ``class_id center_x center_y width height`` in normalized units."""
    fields = line.split()
    if len(fields) != 5:
        raise ValueError("a YOLO annotation must contain exactly 5 fields")
    values = tuple(
        _finite_float(value, name)
        for value, name in zip(fields[1:], ("center_x", "center_y", "width", "height"))
    )
    if any(value < 0 or value > 1 for value in values):
        raise ValueError(f"YOLO coordinates must be normalized to [0, 1], got {values}")
    if values[2] <= 0 or values[3] <= 0:
        raise ValueError("YOLO width and height must be positive")
    class_id = _integer(fields[0], "class_id")
    if class_id != 0:
        raise ValueError(f"class_id must be 0 (vessel), got {class_id}")
    return YoloLabel(class_id, *values)


def parse_3d_prediction_line(line: str) -> Box3DPrediction:
    """Parse a 15-field detection row followed by a confidence score."""
    fields = line.split()
    if len(fields) != 16:
        raise ValueError("a 3D prediction must contain 15 label fields and confidence")
    label = _parse_box_fields(fields[:15], frame_id=None)
    confidence = _finite_float(fields[15], "confidence")
    return Box3DPrediction(label=label, confidence=confidence)


def _load_lines(path: PathLike) -> Iterable[Tuple[int, str]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if stripped:
                yield line_number, stripped


def _load(path: PathLike, parser) -> List:
    labels = []
    for line_number, line in _load_lines(path):
        try:
            labels.append(parser(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return labels


def load_3d_detection_labels(path: PathLike) -> List[Box3DLabel]:
    return _load(path, parse_3d_detection_line)


def load_3d_tracking_labels(path: PathLike) -> List[Box3DLabel]:
    return _load(path, parse_3d_tracking_line)


def load_yolo_labels(path: PathLike) -> List[YoloLabel]:
    return _load(path, parse_yolo_line)


def load_3d_predictions(path: PathLike) -> List[Box3DPrediction]:
    return _load(path, parse_3d_prediction_line)
