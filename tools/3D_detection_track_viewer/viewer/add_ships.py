from vedo import *
from .box_op import convert_box_type
import numpy as np
from .color_map import generate_objects_colors,generate_objects_color_map

objects_color_map = generate_objects_color_map('rainbow')

def add_3D_ships(boxes=None,
                     ids=None,
                     box_info=None,
                     color="blue",
                     mesh_alpha = 0.1,
                     show_ids = False,
                     show_box_info=False,
                    #  del_after_show=True,
                     car_model_path="viewer/car.obj",
                     caption_size = (0.1, 0.1)
                    ):

        if boxes is None:
            return
        # boxes= convert_box_type(boxes,"maritime3d")
        # if boxes is None:
        #     return

        # if ids is not None:
        #     colors = generate_objects_colors(ids,objects_color_map)
        # else:
        #     colors = color
        colors = color

        tracks_actors_dict = {}

        for i in range(len(boxes)):
            bb = boxes[i]

            size = bb[3:6]

            ang=bb[6]
            ang = int(ang / (2 * np.pi) * 360)

            if type(colors) is str:
                color = colors
            else:
                color = colors[i]

            if ids is not None:
                ob_id = ids[i]
                if ob_id in tracks_actors_dict.keys():
                    previous_ori=tracks_actors_dict[ob_id].GetOrientation()[2]
                    tracks_actors_dict[ob_id].pos(0,0,0)
                    tracks_actors_dict[ob_id].rotateZ(ang-previous_ori)
                    tracks_actors_dict[ob_id].pos(bb[0], bb[1], bb[2])

                    info = ""
                    if ids is not None and show_ids:
                        info = "ID: " + str(ids[i]) + '\n'
                    if box_info is not None and show_box_info:
                        info += str(box_info[i])
                    if info != '':
                        tracks_actors_dict[ob_id].caption(info,
                                                               point=(bb[0], bb[1] - bb[4] / 2, bb[2] + bb[5] / 2),
                                                               size=caption_size,
                                                               alpha=1,
                                                               c=color,
                                                               font="Calco",
                                                               justify='left')
                        tracks_actors_dict[ob_id]._caption.SetBorder(False)
                        tracks_actors_dict[ob_id]._caption.SetLeader(False)

                    # if del_after_show:
                    #     self.actors.append(tracks_actors_dict[ob_id])
                    # else:
                    #     self.actors_without_del.append(tracks_actors_dict[ob_id])
                    # return tracks_actors_dict
                else:

                    new_car=load(car_model_path)
                    # new_car.scale((1,0.3,0.3))
                    new_car.scale((0.12,0.3,0.3))

                    new_car.scale(size)
                    new_car.rotateZ(ang)
                    new_car.pos(bb[0], bb[1], bb[2])

                    new_car.c(color)
                    new_car.alpha(mesh_alpha)
                    tracks_actors_dict[ob_id]=new_car
                    info = ""
                    if ids is not None and show_ids:
                        info = "ID: " + str(ids[i]) + '\n'
                    if box_info is not None and show_box_info:
                        info += str(box_info[i])
                    if info != '':
                        tracks_actors_dict[ob_id].caption(info,
                                                               point=(bb[0], bb[1] - bb[4] / 2, bb[2] + bb[5] / 2),
                                                               size=caption_size,
                                                               alpha=1,
                                                               c=color,
                                                               font="Calco",
                                                               justify='left')
                        tracks_actors_dict[ob_id]._caption.SetBorder(False)
                        tracks_actors_dict[ob_id]._caption.SetLeader(False)

                    # return tracks_actors_dict

            else:
                new_car = load(car_model_path)
                new_car.scale((0.12, 0.3, 0.3))

                new_car.scale(size)
                new_car.rotateZ(ang)
                new_car.pos(bb[0], bb[1], bb[2])

                new_car.c(color)
                new_car.alpha(mesh_alpha)

                info = ""

                if box_info is not None and show_box_info:
                    info += str(box_info[i])
                if info != '':
                    new_car.caption(info,
                                   point=(bb[0], bb[1] - bb[4] / 2, bb[2] + bb[5] / 2),
                                   size=caption_size,
                                   alpha=1,
                                   c=color,
                                   font="Calco",
                                   justify='cent')
                    new_car._caption.SetBorder(False)
                    new_car._caption.SetLeader(False)
                return new_car
        return tracks_actors_dict, ids