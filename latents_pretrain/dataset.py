
import torch
import scipy.io
import numpy as np

class PointValueData(object):

    def __init__(self, data_dir, data_list, data_label, sample_reso):
        self.data_dir = data_dir
        self.data_list = data_list
        self.data_label = data_label  
        self.sample_reso = sample_reso

    def load_data(self, path):

        mat_file = scipy.io.loadmat(path)
        colored_voxel = mat_file['colored_voxel']

        data = mat_file['data_'+str(self.sample_reso)]
        points = data[:,:3]
        values = data[:,3:4]
        colors = data[:,4:]

        return colored_voxel.astype(np.float32), points.astype(np.float32), values.astype(np.float32), colors.astype(np.float32)

    def __getitem__(self, index):

        file_path = self.data_dir+self.data_list[index]
        colored_voxel, points, values, colors = self.load_data(file_path)
        sample_name = self.data_list[index][:-4]
        class_label = self.data_label[index]

        return colored_voxel, points, values, colors, np.int(class_label), sample_name, index

    def __len__(self):
        return len(self.data_list)
