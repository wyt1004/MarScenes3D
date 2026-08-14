#!/usr/bin/env python3
"""Visualize one MarScenes3D point cloud with 3D detection boxes."""

import argparse
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.convert_pcd_to_npy import read_pcd  # noqa: E402
from utils.marscenes3d_label import (  # noqa: E402
    load_3d_detection_labels,
    load_3d_predictions,
)


GT_COLOR = (0.2, 1.0, 0.2)
PREDICTION_COLOR = (0.2, 0.55, 1.0)


def load_points(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".pcd":
        points = read_pcd(path)
    elif suffix == ".npy":
        points = np.load(path)
    elif suffix == ".bin":
        values = np.fromfile(path, dtype=np.float32)
        if values.size % 4:
            raise ValueError(f"{path}: binary point cloud must contain N x 4 float32 values")
        points = values.reshape(-1, 4)
    else:
        raise ValueError(f"Unsupported point-cloud extension {suffix!r}; use .pcd, .npy, or .bin")

    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"{path}: expected an N x C point array with C >= 3, got {points.shape}")
    if not np.isfinite(points[:, :3]).all():
        raise ValueError(f"{path}: point coordinates must be finite")
    return points.astype(np.float32, copy=False)


def intensity_colors(points: np.ndarray) -> np.ndarray:
    if points.shape[1] < 4 or not np.isfinite(points[:, 3]).all():
        return np.full((len(points), 3), 0.85, dtype=np.float64)

    intensity = points[:, 3].astype(np.float64)
    lower, upper = np.percentile(intensity, (1.0, 99.0))
    if upper <= lower:
        scaled = np.full(len(points), 0.75, dtype=np.float64)
    else:
        scaled = np.clip((intensity - lower) / (upper - lower), 0.0, 1.0)
    scaled = 0.25 + 0.75 * scaled
    return np.repeat(scaled[:, None], 3, axis=1)


def add_box(open3d, visualizer, box, color) -> None:
    center = np.asarray(box[:3], dtype=np.float64)
    dimensions = np.asarray(box[3:6], dtype=np.float64)
    yaw = float(box[6])
    rotation = open3d.geometry.get_rotation_matrix_from_axis_angle((0.0, 0.0, yaw))
    oriented_box = open3d.geometry.OrientedBoundingBox(center, rotation, dimensions)
    lines = open3d.geometry.LineSet.create_from_oriented_bounding_box(oriented_box)
    lines.paint_uniform_color(color)
    visualizer.add_geometry(lines)

    heading_end = center + rotation @ np.asarray((dimensions[0] / 2.0, 0.0, 0.0))
    heading = open3d.geometry.LineSet()
    heading.points = open3d.utility.Vector3dVector(np.vstack((center, heading_end)))
    heading.lines = open3d.utility.Vector2iVector(np.asarray(((0, 1),), dtype=np.int32))
    heading.paint_uniform_color(color)
    visualizer.add_geometry(heading)


def visualize(points, gt_boxes, prediction_boxes, point_size: float, use_intensity: bool) -> None:
    try:
        import open3d
    except ImportError as exc:
        raise RuntimeError(
            "Open3D is required for 3D detection visualization. "
            "Install requirements-visualization.txt first."
        ) from exc

    visualizer = open3d.visualization.Visualizer()
    if not visualizer.create_window(window_name="MarScenes3D 3D Detection", width=1280, height=720):
        raise RuntimeError("Unable to create an Open3D window; check the desktop/display environment")

    render = visualizer.get_render_option()
    render.background_color = np.asarray((0.0, 0.0, 0.0))
    render.point_size = point_size

    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(points[:, :3])
    if use_intensity:
        cloud.colors = open3d.utility.Vector3dVector(intensity_colors(points))
    else:
        cloud.paint_uniform_color((0.85, 0.85, 0.85))
    visualizer.add_geometry(cloud)
    visualizer.add_geometry(
        open3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=(0.0, 0.0, 0.0))
    )

    for box in gt_boxes:
        add_box(open3d, visualizer, box, GT_COLOR)
    for box in prediction_boxes:
        add_box(open3d, visualizer, box, PREDICTION_COLOR)

    view = visualizer.get_view_control()
    view.set_lookat((0.0, 40.0, 0.0))
    view.set_up((0.0, 0.0, 1.0))
    view.set_front((0.0, -1.0, -0.35))
    view.set_zoom(0.3)
    visualizer.run()
    visualizer.destroy_window()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--point-cloud",
        required=True,
        type=Path,
        help="Single MarScenes3D .pcd, .npy, or OpenPCDet-style .bin file",
    )
    parser.add_argument("--gt-label", type=Path, help="Optional 15-field Label3D file")
    parser.add_argument(
        "--pred-label",
        type=Path,
        help="Optional 16-field detection result file with confidence scores",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Minimum prediction confidence to display (default: 0.0)",
    )
    parser.add_argument("--point-size", type=float, default=2.0, help="Rendered point size")
    parser.add_argument(
        "--uniform-points",
        action="store_true",
        help="Use uniform gray points instead of intensity-based grayscale",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.point_size <= 0:
        raise ValueError("--point-size must be positive")

    points = load_points(args.point_cloud)
    gt_labels = load_3d_detection_labels(args.gt_label) if args.gt_label else []
    predictions = load_3d_predictions(args.pred_label) if args.pred_label else []
    predictions = [item for item in predictions if item.confidence >= args.score_threshold]

    gt_boxes = [label.lidar_box for label in gt_labels]
    prediction_boxes = [prediction.lidar_box for prediction in predictions]
    print(
        f"Visualizing {len(points)} points, {len(gt_boxes)} ground-truth boxes, "
        f"and {len(prediction_boxes)} prediction boxes"
    )
    print("Colors: ground truth = green, predictions = blue")
    visualize(points, gt_boxes, prediction_boxes, args.point_size, not args.uniform_points)


if __name__ == "__main__":
    main()
