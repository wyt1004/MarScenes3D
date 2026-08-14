"""Model-agnostic 3D detection evaluation for MarScenes3D."""

from collections import defaultdict
import math

import numpy as np


CLASS_NAMES = (
    "large_vessel", "medium_vessel", "small_vessel",
    "sign", "piers", "lighthouse",
)
DEFAULT_IOU_THRESHOLDS = {
    "large_vessel": (0.5, 0.5),
    "medium_vessel": (0.5, 0.5),
    "small_vessel": (0.5, 0.5),
    "sign": (0.5, 0.5),
    "piers": (0.5, 0.5),
    "lighthouse": (0.5, 0.5),
}
EPS = 1e-9


def _corners(box):
    x, y, _, dx, dy, _, yaw = box
    local = np.asarray([
        [dx / 2, dy / 2], [-dx / 2, dy / 2],
        [-dx / 2, -dy / 2], [dx / 2, -dy / 2],
    ])
    rotation = np.asarray([
        [math.cos(yaw), -math.sin(yaw)],
        [math.sin(yaw), math.cos(yaw)],
    ])
    return local @ rotation.T + np.asarray([x, y])


def _inside(point, start, end):
    return ((end[0] - start[0]) * (point[1] - start[1])
            - (end[1] - start[1]) * (point[0] - start[0])) >= -EPS


def _intersection(start1, end1, start2, end2):
    direction1 = end1 - start1
    direction2 = end2 - start2
    denominator = direction1[0] * direction2[1] - direction1[1] * direction2[0]
    if abs(denominator) < EPS:
        return start1.copy()
    delta = start2 - start1
    factor = (delta[0] * direction2[1] - delta[1] * direction2[0]) / denominator
    return start1 + factor * direction1


def _clip(subject, clipper):
    output = subject
    for index, clip_start in enumerate(clipper):
        clip_end = clipper[(index + 1) % len(clipper)]
        input_points = output
        if len(input_points) == 0:
            break
        output = []
        previous = input_points[-1]
        for current in input_points:
            if _inside(current, clip_start, clip_end):
                if not _inside(previous, clip_start, clip_end):
                    output.append(_intersection(previous, current, clip_start, clip_end))
                output.append(current)
            elif _inside(previous, clip_start, clip_end):
                output.append(_intersection(previous, current, clip_start, clip_end))
            previous = current
        output = np.asarray(output, dtype=float).reshape(-1, 2)
    return output


def box3d_iou(box1, box2):
    """Return 3D IoU for two ``(x,y,z,dx,dy,dz,yaw)`` LiDAR boxes."""
    box1 = np.asarray(box1, dtype=float)
    box2 = np.asarray(box2, dtype=float)
    if box1.shape != (7,) or box2.shape != (7,):
        raise ValueError("3D boxes must have shape (7,)")
    if np.any(box1[3:6] <= 0) or np.any(box2[3:6] <= 0):
        return 0.0
    polygon = _clip(_corners(box1), _corners(box2))
    if len(polygon) < 3:
        return 0.0
    area = 0.5 * abs(
        np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
        - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
    )
    bottom1, top1 = box1[2] - box1[5] / 2, box1[2] + box1[5] / 2
    bottom2, top2 = box2[2] - box2[5] / 2, box2[2] + box2[5] / 2
    overlap_height = max(0.0, min(top1, top2) - max(bottom1, bottom2))
    intersection = area * overlap_height
    volume1 = np.prod(box1[3:6])
    volume2 = np.prod(box2[3:6])
    union = volume1 + volume2 - intersection
    return float(intersection / union) if union > EPS else 0.0


def _ap_r40(ground_truth, predictions, threshold):
    """Evaluate one class using score ordering and 40 recall positions."""
    gt_by_sample = defaultdict(list)
    for sample_id, box in ground_truth:
        gt_by_sample[str(sample_id)].append(np.asarray(box, dtype=float))
    matched = {sample_id: np.zeros(len(boxes), dtype=bool) for sample_id, boxes in gt_by_sample.items()}
    predictions = sorted(predictions, key=lambda item: item[2], reverse=True)
    true_positive = np.zeros(len(predictions), dtype=float)
    false_positive = np.zeros(len(predictions), dtype=float)
    for index, (sample_id, box, _) in enumerate(predictions):
        sample_id = str(sample_id)
        candidates = gt_by_sample.get(sample_id, [])
        available = [i for i in range(len(candidates)) if not matched[sample_id][i]] if candidates else []
        if available:
            overlaps = [box3d_iou(box, candidates[i]) for i in available]
            best_position = int(np.argmax(overlaps))
            if overlaps[best_position] >= threshold:
                matched[sample_id][available[best_position]] = True
                true_positive[index] = 1
                continue
        false_positive[index] = 1
    num_gt = sum(len(boxes) for boxes in gt_by_sample.values())
    if num_gt == 0:
        return None
    if not predictions:
        return {"AP_R40": 0.0, "recall": 0.0, "precision": 0.0, "num_gt": num_gt, "num_pred": 0}
    tp = np.cumsum(true_positive)
    fp = np.cumsum(false_positive)
    recall = tp / num_gt
    precision = tp / np.maximum(tp + fp, EPS)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    samples = []
    for target_recall in np.arange(1, 41) / 40:
        valid = precision[recall >= target_recall]
        samples.append(float(valid.max()) if len(valid) else 0.0)
    return {
        "AP_R40": 100.0 * float(np.mean(samples)),
        "recall": 100.0 * float(recall[-1]),
        "precision": 100.0 * float(tp[-1] / max(tp[-1] + fp[-1], EPS)),
        "num_gt": num_gt,
        "num_pred": len(predictions),
    }


def evaluate_detection(ground_truth, predictions, class_names=CLASS_NAMES,
                       thresholds=DEFAULT_IOU_THRESHOLDS):
    """Evaluate lists of ``(sample_id, class_name, box[, score])`` tuples."""
    result = {"classes": {}}
    strict_values, loose_values = [], []
    for class_name in class_names:
        class_gt = [(item[0], item[2]) for item in ground_truth if item[1] == class_name]
        class_predictions = [(item[0], item[2], float(item[3])) for item in predictions if item[1] == class_name]
        strict, loose = thresholds[class_name]
        strict_result = _ap_r40(class_gt, class_predictions, strict)
        loose_result = _ap_r40(class_gt, class_predictions, loose)
        result["classes"][class_name] = {
            "strict_iou": strict, "strict": strict_result,
            "loose_iou": loose, "loose": loose_result,
        }
        if strict_result is not None:
            strict_values.append(strict_result["AP_R40"])
            loose_values.append(loose_result["AP_R40"])
    result["mAP_R40_strict"] = float(np.mean(strict_values)) if strict_values else None
    result["mAP_R40_loose"] = float(np.mean(loose_values)) if loose_values else None
    return result


def evaluate_openpcdet_annos(gt_annos, det_annos, class_names=CLASS_NAMES):
    ground_truth, predictions = [], []
    for sample_id, annotation in enumerate(gt_annos):
        boxes = annotation.get("gt_boxes_lidar", annotation.get("boxes_lidar"))
        for name, box in zip(annotation["name"], boxes):
            ground_truth.append((sample_id, str(name), np.asarray(box)[:7]))
    for sample_id, annotation in enumerate(det_annos):
        boxes = annotation.get("boxes_lidar", annotation.get("gt_boxes_lidar"))
        scores = annotation.get("score", np.ones(len(annotation["name"])))
        for name, box, score in zip(annotation["name"], boxes, scores):
            predictions.append((sample_id, str(name), np.asarray(box)[:7], float(score)))
    return evaluate_detection(ground_truth, predictions, class_names)


def format_results(result):
    lines = [
        "MarScenes3D 3D detection evaluation (40-point interpolated AP)",
        "class                 IoU       AP_R40",
    ]
    flat = {}
    for name, values in result["classes"].items():
        strict = values["strict"]
        loose = values["loose"]
        strict_ap = "N/A" if strict is None else f'{strict["AP_R40"]:.4f}'
        loose_ap = "N/A" if loose is None else f'{loose["AP_R40"]:.4f}'
        lines.append(f'{name:18s} {values["strict_iou"]:8.2f} {strict_ap:>12s}')
        if strict is not None:
            flat[f"{name}/AP_R40@{values['strict_iou']:.2f}"] = strict["AP_R40"]
            flat[f"{name}/AP_R40@{values['loose_iou']:.2f}"] = loose["AP_R40"]
    lines.append("mAP_R40: %s" % (
        "N/A" if result["mAP_R40_strict"] is None else f'{result["mAP_R40_strict"]:.4f}'))
    flat["mAP_R40_strict"] = result["mAP_R40_strict"]
    flat["mAP_R40_loose"] = result["mAP_R40_loose"]
    return "\n".join(lines), flat
