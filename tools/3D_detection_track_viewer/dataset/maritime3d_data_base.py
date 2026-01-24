import os
import cv2
import re
import numpy as np

"""
input: calib txt path
return: P0: (4,4) 3D camera coordinates to 2D image pixels
        vtc_mat: (4,4) 3D velodyne Lidar coordinates to 3D camera coordinates
"""
def read_calib(calib_path):
    with open(calib_path) as f:
        for line in f.readlines():
            if line[:2] == "P0":
                P0 = re.split(" ", line.strip())
                P0 = np.array(P0[-12:], np.float32)
                P0 = P0.reshape((3, 4))
            if line[:14] == "Tr_velo_to_cam" or line[:11] == "Tr_velo_cam":
                vtc_mat = re.split(" ", line.strip())
                vtc_mat = np.array(vtc_mat[-12:], np.float32)
                vtc_mat = vtc_mat.reshape((3, 4))
                vtc_mat = np.concatenate([vtc_mat, [[0, 0, 0, 1]]])
            # if line[:7] == "R0_rect" or line[:6] == "R_rect":
            #     R0 = re.split(" ", line.strip())
            #     R0 = np.array(R0[-9:], np.float32)
            #     R0 = R0.reshape((3, 3))
            #     R0 = np.concatenate([R0, [[0], [0], [0]]], -1)
            #     R0 = np.concatenate([R0, [[0, 0, 0, 1]]])
    # vtc_mat = np.matmul(R0, vtc_mat)
    return (P0, vtc_mat)


"""
description: read lidar data given 
input: lidar bin path "path", cam 3D to cam 2D image matrix (4,4), lidar 3D to cam 3D matrix (4,4)
output: valid points in lidar coordinates (PointsNum,4)
"""
# def read_velodyne(path, P, vtc_mat,IfReduce=True):
#     # max_row = 1079  # y
#     # max_col = 1919  # x

#     max_row = 2047  # y
#     max_col = 2447  # x

#     lidar = np.fromfile(path, dtype=np.float32).reshape((-1, 4))

#     if not IfReduce:
#         return lidar

#     mask = lidar[:, 1] > 0
#     lidar = lidar[mask]
#     lidar_copy = np.zeros(shape=lidar.shape)
#     lidar_copy[:, :] = lidar[:, :]

#     velo_tocam = vtc_mat
#     lidar[:, 3] = 1
#     lidar = np.matmul(lidar, velo_tocam.T)
#     img_pts = np.matmul(lidar, P.T)
#     # velo_tocam = np.mat(velo_tocam).I
#     velo_tocam = np.linalg.inv(velo_tocam)
#     # velo_tocam = np.array(velo_tocam)
#     normal = velo_tocam
#     normal = normal[0:3, 0:4]
#     lidar = np.matmul(lidar, normal.T)
#     lidar_copy[:, 0:3] = lidar
#     x, y = img_pts[:, 0] / img_pts[:, 2], img_pts[:, 1] / img_pts[:, 2]
#     mask = np.logical_and(np.logical_and(x >= 0, x < max_col), np.logical_and(y >= 0, y < max_row))

#     # return lidar
#     return lidar_copy[mask]

def read_velodyne(path, P, vtc_mat, IfReduce=True,
                #   max_row=1079, max_col=1919):
                  max_row=2047, max_col=2447):
    """
    读取 Velodyne 点云，并根据相机内外参筛选出投影在图像范围内的点。

    参数
    ----
    path : str
        .bin 点云文件路径（每点 [x, y, z, reflect]，单位通常为米）。
    P : np.ndarray
        相机内参或投影矩阵：
        - (3,3): K = [[fx,0,cx],[0,fy,cy],[0,0,1]]
        - (3,4): 完整投影矩阵 (如 KITTI 的 P2)
    vtc_mat : np.ndarray
        4x4 齐次外参矩阵，将点从雷达坐标系变换到相机坐标系：
        X_cam = vtc_mat @ [X_l, Y_l, Z_l, 1]^T
    IfReduce : bool
        True  时只返回投影在图像范围内的点
        False 时直接返回全部点云（原始雷达坐标系）。
    max_row, max_col : int
        图像高(H)和宽(W)，用于 FOV 过滤。

    返回
    ----
    lidar_filtered : np.ndarray, shape (M, 4)
        筛选后的点云（雷达坐标系下的 [x,y,z,reflect]）。
        IfReduce=False 时，等同于原始点云。
    """

    # 读取原始 Velodyne 点云 [x, y, z, reflect]
    lidar = np.fromfile(path, dtype=np.float32).reshape((-1, 4))

    if not IfReduce:
        # 不做任何过滤，直接返回
        return lidar

    # -----------------------------
    # 1) 准备雷达齐次坐标
    # -----------------------------
    pts_lidar = lidar[:, :3]  # (N,3)
    N = pts_lidar.shape[0]
    ones = np.ones((N, 1), dtype=np.float32)
    pts_lidar_h = np.hstack([pts_lidar, ones])  # (N,4) -> [x,y,z,1]

    # -----------------------------
    # 2) 雷达 -> 相机 坐标系
    # -----------------------------
    assert vtc_mat.shape == (4, 4), "vtc_mat 必须是 4x4 雷达到相机的齐次外参矩阵"
    pts_cam_h = (vtc_mat @ pts_lidar_h.T).T  # (N,4)
    Xc, Yc, Zc = pts_cam_h[:, 0], pts_cam_h[:, 1], pts_cam_h[:, 2]

    # 只保留在相机前方的点 (Zc > 0)
    front_mask = Zc > 0
    if not np.any(front_mask):
        # 没有任何在前方的点，直接返回空
        return lidar[:0, :]

    pts_cam_h_front = pts_cam_h[front_mask]
    Xc_f, Yc_f, Zc_f = pts_cam_h_front[:, 0], pts_cam_h_front[:, 1], pts_cam_h_front[:, 2]

    # -----------------------------
    # 3) 相机坐标系 -> 像素坐标
    # -----------------------------
    P = np.asarray(P)
    if P.shape == (3, 3):
        # 当作内参 K
        fx, fy = P[0, 0], P[1, 1]
        cx, cy = P[0, 2], P[1, 2]
        u = fx * (Xc_f / Zc_f) + cx
        v = fy * (Yc_f / Zc_f) + cy
    elif P.shape == (3, 4):
        # 当作完整投影矩阵
        img_pts = (P @ pts_cam_h_front.T).T  # (M,3)
        u = img_pts[:, 0] / img_pts[:, 2]
        v = img_pts[:, 1] / img_pts[:, 2]
    else:
        raise ValueError("P 必须是 3x3 (K) 或 3x4 (投影矩阵)，当前形状为 {}".format(P.shape))

    # -----------------------------
    # 4) FOV 过滤：只保留投影落在图像内的点
    # -----------------------------
    in_img = (u >= 0) & (u < max_col) & (v >= 0) & (v < max_row)
    if not np.any(in_img):
        return lidar[:0, :]

    # front_mask 是针对全部 N 点的，in_img 是 front 子集里的
    idx_front = np.flatnonzero(front_mask)       # 原始 N 中 Z>0 的索引
    idx_keep = idx_front[in_img]                 # 同时满足 Z>0 且落在图像内的索引

    # 在雷达坐标系下返回这些点（保留原始反射值）
    lidar_filtered = lidar[idx_keep]

    return lidar_filtered


"""
description: convert 3D camera coordinates to Lidar 3D coordinates.
input: (PointsNum,3)
output: (PointsNum,3)
"""
def cam_to_velo(cloud,vtc_mat):
    mat=np.ones(shape=(cloud.shape[0],4),dtype=np.float32)
    mat[:,0:3]=cloud[:,0:3]
    # mat=np.mat(mat)
    # normal=np.mat(vtc_mat).I
    normal = np.linalg.inv(np.asarray(vtc_mat, dtype=np.float32))
    normal=normal[0:3,0:4]
    transformed_mat = normal @ mat.T
    # transformed_mat = normal * mat.T
    T=np.array(transformed_mat.T,dtype=np.float32)
    return T

"""
description: convert 3D camera coordinates to Lidar 3D coordinates.
input: (PointsNum,3)
output: (PointsNum,3)
"""
def velo_to_cam(cloud,vtc_mat):
    mat=np.ones(shape=(cloud.shape[0],4),dtype=np.float32)
    mat[:,0:3]=cloud[:,0:3]
    # mat=np.mat(mat)
    # normal=np.mat(vtc_mat).I
    normal = np.linalg.inv(vtc_mat)
    normal=normal[0:3,0:4]
    transformed_mat = normal @ mat.T
    # transformed_mat = normal * mat.T
    T=np.array(transformed_mat.T,dtype=np.float32)
    return T

def read_image(path):
    im=cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return im

def read_detection_label(path):

    boxes = []
    names = []

    with open(path) as f:
        for line in f.readlines():
            line = line.split()
            this_name = line[0]
            if this_name != "DontCare":
                line = np.array(line[-7:],np.float32)
                boxes.append(line)
                names.append(this_name)

    return np.array(boxes),np.array(names)

def read_tracking_label(path):

    frame_dict={}

    names_dict={}

    with open(path) as f:
        for line in f.readlines():
            line = line.split()
            this_name = line[1]
            frame_id = int(line[0])
            ob_id = int(line[3])

            #可视化修正
            if ob_id in [22, 8, 26, 29, 31, 39]:
                continue

            if this_name != "DontCare":
                line = np.array(line[4:11],np.float32).tolist()
                line.append(ob_id)


                if frame_id in frame_dict.keys():
                    frame_dict[frame_id].append(line)
                    names_dict[frame_id].append(this_name)
                else:
                    frame_dict[frame_id] = [line]
                    names_dict[frame_id] = [this_name]

    return frame_dict,names_dict

def get_sorted_file_list(folder):
    # 1. 获取所有文件名
    files = [f for f in os.listdir(folder)]
    # 2. 按照文件名的字典序排序
    files.sort()
    # 3. 去掉扩展名
    names = [os.path.splitext(f)[0] for f in files]
    return names

if __name__ == '__main__':
    path = 'H:/dataset/traking/training/label_02/0000.txt'
    labels,a = read_tracking_label(path)
    print(a)

