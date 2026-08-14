import numpy as np
import re
from .maritime3d_data_base import *
import os

class Maritime3dDetectionDataset:
    def __init__(self,root_path,label_path = None):
        self.root_path = root_path
        self.velo_path = os.path.join(self.root_path,"points")
        self.image_path = os.path.join(self.root_path,"images")
        self.calib_path = os.path.join(self.root_path,"calib") #to do
        if label_path is None:
            self.label_path = os.path.join(self.root_path, "labels")
        else:
            self.label_path = label_path

        self.frame_names = get_sorted_file_list(self.velo_path)

    def __len__(self):
        return len(self.frame_names)
    def __getitem__(self, item):

        name = self.frame_names[item]

        velo_path = os.path.join(self.velo_path,name+'.npy')
        image_path = os.path.join(self.image_path, name+'.png')
        calib_path = os.path.join(self.calib_path, name+'.txt')
        label_path = os.path.join(self.label_path, name+".txt")

        P2,V2C = read_calib(calib_path)
        points = read_velodyne(velo_path,P2,V2C)
        image = read_image(image_path)
        labels,label_names = read_detection_label(label_path)

        return P2,V2C,points,image,labels,label_names

class Maritime3dTrackingDataset:
    def __init__(self,root_path,seq_id,label_path=None):
        self.seq_name = str(seq_id)
        self.root_path = root_path
        self.velo_path = os.path.join(self.root_path,"points",self.seq_name)
        self.image_path = os.path.join(self.root_path,"image",self.seq_name)
        self.calib_path = os.path.join(self.root_path,"calib",self.seq_name.zfill(4))



        calib_path = self.calib_path + '.txt'

        if label_path is None:

            label_path = os.path.join(self.root_path, "label", self.seq_name.zfill(4)+'.txt')


        self.P2, self.V2C = read_calib(calib_path)
        self.labels, self.label_names = read_tracking_label(label_path)

        img_names = set(get_sorted_file_list(self.image_path))
        pcd_names = set(get_sorted_file_list(self.velo_path))
        self.frame_names = sorted(img_names & pcd_names)
        if not self.frame_names:
            raise ValueError(f'No synchronized image/point-cloud frames found for sequence {self.seq_name}')

    def __len__(self):
        return len(self.frame_names)
    def __getitem__(self, item):

        img_name = self.frame_names[item]
        pcd_name = self.frame_names[item]

        velo_path = os.path.join(self.velo_path, pcd_name+'.npy')
        image_path = os.path.join(self.image_path, img_name+'.png')



        points = read_velodyne(velo_path,self.P2,self.V2C)
        image = read_image(image_path)

        point_cloud = np.load(velo_path)
        point_cloud = point_cloud[:, :3]


        if item in self.labels.keys():
            labels = self.labels[item]
            labels = np.array(labels)
            # labels[:,3:6] = cam_to_velo(labels[:,3:6],self.V2C)[:,:3]
            label_names = self.label_names[item]
            label_names = np.array(label_names)
        else:
            labels = None
            label_names = None

        return self.P2,self.V2C,points,image,labels,label_names, point_cloud
