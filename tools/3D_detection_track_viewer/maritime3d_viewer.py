import argparse

from viewer.viewer import Viewer
import numpy as np
from dataset.maritime3d_dataset import Maritime3dTrackingDataset
from vedo import Plotter, Points
import os
from viewer.box_op import get_line_boxes, convert_box_type, get_mesh_boxes
from viewer.add_ships import add_3D_ships
from viewer.color_map import generate_objects_colors,generate_objects_color_map

def kitti_viewer(root, seq_id, label_path=None, frame_ids=None):
    dataset = Maritime3dTrackingDataset(root, seq_id=seq_id, label_path=label_path)

    vi = Viewer(box_type="maritime3d")

    plotter = Plotter(size=(1280, 360))

    for i in range(len(dataset)):
        P2, V2C, points, image, labels, label_names, point_cloud = dataset[i]
        # print(f"idx: {i}")

        if frame_ids is not None and i not in frame_ids:
            continue

        if labels is not None:
            ids=labels[:, -1].astype(int)
            objects_color_map = generate_objects_color_map('rainbow')
            colors = generate_objects_colors(ids,objects_color_map)
                        
            labels = convert_box_type(labels, "maritime3d")
            vi.add_2D_text(boxes=labels, ids=ids, colors=colors, box_info=label_names,add_to_2D_scene=True)
            plotter += get_line_boxes(labels, colors=colors, show_heading=False)

            # Use line boxes by default; optional third-party mesh assets are
            # intentionally not required by the public viewer.
        vi.add_points(points[:,:3])

        points = Points(point_cloud[:,:3], r=2.5, c=(175, 175, 175))

        # 
        plotter += points
        
        # 
        plotter.show(resetcam=False,  camera={'pos': (0, -150, 100), 'focalPoint': (0, 100, -10), 'viewup': (0, 0, 1)})

        # clear point
        plotter.clear()

        vi.add_image(image)
        vi.set_extrinsic_mat(V2C)
        vi.set_intrinsic_mat(P2)

        vi.show_2D()

        # vi.show_3D()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize a MarScenes3D tracking sequence.')
    parser.add_argument('--root', required=True, help='Root of the prepared tracking dataset')
    parser.add_argument('--seq-id', type=int, required=True, help='Sequence ID')
    parser.add_argument('--label-path', default=None, help='Tracking result file; defaults to dataset labels')
    parser.add_argument('--frames', nargs='*', type=int, default=None, help='Optional frame IDs to display')
    args = parser.parse_args()
    kitti_viewer(args.root, args.seq_id, args.label_path, args.frames)
