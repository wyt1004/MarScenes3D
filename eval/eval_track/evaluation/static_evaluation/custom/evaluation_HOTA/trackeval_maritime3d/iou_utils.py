import numpy as np
from typing import Iterable, Tuple
from .utils import TrackEvalException

EPS = 1e-9


# ============ 基础几何工具 ============

def _rect_corners_xy(x: float, y: float, l: float, w: float, yaw: float) -> np.ndarray:
    """
    生成绕 z 轴旋转 yaw 的矩形四个角点在平面 (x, y) 的坐标，顺序为 CCW（逆时针）。
    输入: 中心 (x, y)、尺寸 l(沿 x 轴), w(沿 y 轴)、yaw(弧度, 绕 z 轴, 左手系逆时针为正)
    返回: (4, 2) 顶点数组，CCW 顺序
    """
    c, s = np.cos(yaw), np.sin(yaw)
    # 局部坐标系下 CCW 四角：(+dx,+dy) -> (-dx,+dy) -> (-dx,-dy) -> (+dx,-dy)
    dx, dy = l * 0.5, w * 0.5
    local = np.array([[+dx, +dy],
                      [-dx, +dy],
                      [-dx, -dy],
                      [+dx, -dy]], dtype=float)
    R = np.array([[c, -s],
                  [s,  c]], dtype=float)
    pts = local @ R.T
    pts[:, 0] += x
    pts[:, 1] += y
    return pts  # (4,2), CCW

def _poly_area(pts: np.ndarray) -> float:
    """
    多边形面积（正值）。pts 为 (K,2) 并按 CCW 或 CW 给出。
    """
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _is_left(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> bool:
    """
    判断点 p 是否在有向边 a->b 的左侧（包括共线），用于 Sutherland–Hodgman 裁剪。
    """
    return ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= -EPS

def _segment_intersection(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    计算线段 ab 与 cd 的交点（假设它们在凸裁剪中会相交或共线延拓相交）。
    返回点坐标 (2,)。
    """
    # 解 a + t*(b-a) 与 c + u*(d-c) 的交点
    ab = b - a
    cd = d - c
    denom = ab[0] * cd[1] - ab[1] * cd[0]
    if abs(denom) < EPS:
        # 近似平行，取 a 与 c 的中点方向上的一个估计（不会用于面积计算的严格场景）
        # 但在 S-H 算法里，只有当需要交点时才会调用，且理论上此时不应平行。
        # 为了数值稳定，退化时返回 a
        return a.copy()
    t = ((c[0] - a[0]) * cd[1] - (c[1] - a[1]) * cd[0]) / denom
    return a + t * ab

def _convex_clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """
    Sutherland–Hodgman：将 subject 多边形（(Ns,2)）按 clipper（(Nc,2), CCW）裁剪，返回交集多边形顶点（可能为空）。
    适用于凸 clipper（矩形是凸的）。
    """
    output = subject
    if output.shape[0] == 0:
        return output
    for i in range(clipper.shape[0]):
        input_list = output
        output = []
        A = clipper[i]
        B = clipper[(i + 1) % clipper.shape[0]]
        if input_list.shape[0] == 0:
            break
        S = input_list[-1]
        for E in input_list:
            if _is_left(A, B, E):
                if not _is_left(A, B, S):
                    inter = _segment_intersection(S, E, A, B)
                    output.append(inter)
                output.append(E)
            elif _is_left(A, B, S):
                inter = _segment_intersection(S, E, A, B)
                output.append(inter)
            S = E
        if len(output) == 0:
            return np.zeros((0, 2), dtype=float)
        output = np.array(output, dtype=float)
    return output

# ============ 单对 3D 框 IoU ============
def iou3d_single_lwhxyzyaw(box1: Iterable[float], box2: Iterable[float], return_mode: str = "iou") -> float:
    """
    return_mode: "iou" | "ioa1" | "ioa2"
      - "iou":  inter_vol / (v1 + v2 - inter_vol)
      - "ioa1": inter_vol / v1
      - "ioa2": inter_vol / v2
    """
    box1 = np.asarray(box1).reshape(-1)
    box2 = np.asarray(box2).reshape(-1)
    if box1.size != 7 or box2.size != 7:
        raise TrackEvalException("Each box must be 7 elements (l,w,h,x,y,z,yaw).")

    l1, w1, h1, x1, y1, z1, yaw1 = box1.astype(float)
    l2, w2, h2, x2, y2, z2, yaw2 = box2.astype(float)

    # 体积
    v1 = max(l1, 0.0) * max(w1, 0.0) * max(h1, 0.0)
    v2 = max(l2, 0.0) * max(w2, 0.0) * max(h2, 0.0)
    if v1 < EPS or v2 < EPS:
        return 0.0

    # BEV 相交面积
    rect1 = _rect_corners_xy(x1, y1, l1, w1, yaw1)
    rect2 = _rect_corners_xy(x2, y2, l2, w2, yaw2)
    inter_poly = _convex_clip(rect1, rect2)
    inter_area = _poly_area(inter_poly)
    if inter_area < EPS:
        return 0.0

    # z 向重叠
    z1_bot, z1_top = z1 - h1 * 0.5, z1 + h1 * 0.5
    z2_bot, z2_top = z2 - h2 * 0.5, z2 + h2 * 0.5
    inter_h = max(0.0, min(z1_top, z2_top) - max(z1_bot, z2_bot))
    if inter_h < EPS:
        return 0.0

    inter_vol = inter_area * inter_h

    if return_mode == "ioa1":
        return float(inter_vol / v1) if v1 > EPS else 0.0
    if return_mode == "ioa2":
        return float(inter_vol / v2) if v2 > EPS else 0.0

    # 默认 IoU
    union_vol = v1 + v2 - inter_vol
    if union_vol <= EPS:
        return 0.0
    return float(inter_vol / union_vol)

# def iou3d_single_lwhxyzyaw(box1: Iterable[float], box2: Iterable[float]) -> float:
#     """
#     计算两个 3D 旋转包围盒的 IoU。
#     输入/格式： (l, w, h, x, y, z, yaw)  —— 其中 yaw 为弧度，绕 z 轴。
#     返回：一个标量 IoU（[0,1]）
#     """
#     l1, w1, h1, x1, y1, z1, yaw1 = [float(v) for v in box1]
#     l2, w2, h2, x2, y2, z2, yaw2 = [float(v) for v in box2]

#     # 体积（非负保护）
#     v1 = max(l1, 0.0) * max(w1, 0.0) * max(h1, 0.0)
#     v2 = max(l2, 0.0) * max(w2, 0.0) * max(h2, 0.0)
#     if v1 < EPS or v2 < EPS:
#         return 0.0

#     # BEV 矩形交面积
#     rect1 = _rect_corners_xy(x1, y1, l1, w1, yaw1)   # (4,2)
#     rect2 = _rect_corners_xy(x2, y2, l2, w2, yaw2)   # (4,2)
#     inter_poly = _convex_clip(rect1, rect2)          # (K,2) or (0,2)
#     inter_area = _poly_area(inter_poly)

#     if inter_area < EPS:
#         return 0.0

#     # z 方向重叠
#     z1_bot, z1_top = z1 - h1 * 0.5, z1 + h1 * 0.5
#     z2_bot, z2_top = z2 - h2 * 0.5, z2 + h2 * 0.5
#     inter_h = max(0.0, min(z1_top, z2_top) - max(z1_bot, z2_bot))
#     if inter_h < EPS:
#         return 0.0

#     inter_vol = inter_area * inter_h
#     union_vol = v1 + v2 - inter_vol
#     if union_vol <= EPS:
#         return 0.0
#     return float(inter_vol / union_vol)

# ============ 批量 (N, M) IoU ============
def iou3d_matrix_lwhxyzyaw(bboxes1: np.ndarray, bboxes2: np.ndarray, return_mode: str = "iou") -> np.ndarray:
    """
    return_mode: "iou" | "ioa1" | "ioa2"
    """
    b1 = np.asarray(bboxes1, dtype=float)
    b2 = np.asarray(bboxes2, dtype=float)
    if b1.ndim != 2 or b2.ndim != 2 or b1.shape[1] != 7 or b2.shape[1] != 7:
        raise TrackEvalException("Inputs must be (N,7) and (M,7) with format (l,w,h,x,y,z,yaw).")

    N, M = b1.shape[0], b2.shape[0]
    out = np.zeros((N, M), dtype=float)
    for i in range(N):
        for j in range(M):
            out[i, j] = iou3d_single_lwhxyzyaw(b1[i], b2[j], return_mode=return_mode)
    return out

# def iou3d_matrix_lwhxyzyaw(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
#     """
#     计算两组 3D 旋转包围盒的两两 IoU 矩阵。
#     输入:
#         bboxes1: (N, 7)  每行为 (l, w, h, x, y, z, yaw)
#         bboxes2: (M, 7)  每行为 (l, w, h, x, y, z, yaw)
#     返回:
#         ious: (N, M)
#     """
#     b1 = np.asarray(bboxes1, dtype=float)
#     b2 = np.asarray(bboxes2, dtype=float)
#     if b1.ndim != 2 or b2.ndim != 2 or b1.shape[1] != 7 or b2.shape[1] != 7:
#         raise TrackEvalException("Inputs must be (N,7) and (M,7) with format (l,w,h,x,y,z,yaw).")

#     N, M = b1.shape[0], b2.shape[0]
#     out = np.zeros((N, M), dtype=float)
#     for i in range(N):
#         for j in range(M):
#             out[i, j] = iou3d_single_lwhxyzyaw(b1[i], b2[j])
#     return out
