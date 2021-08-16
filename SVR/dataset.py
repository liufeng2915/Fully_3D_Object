
import torch
import scipy.io
import numpy as np
from torchvision import transforms
from PIL import Image
from imageio import imread
import scipy.io

class SynData(object):

    def __init__(self, config):

        self.img_dir = config.synthetic_img_dir
        self.fc = config.fc  
        self.fa = config.fa 
        self.fs = config.fs  
        self.image_list = config.image_list
        self.num_views = config.num_views 
        self.latent_label = config.latent_label

        self.mean_p = config.mean_p
        self.std_p = config.std_p
        self.transform_input = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 
        self.transform_gt = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()])

    def load_input_img(self, path):
        img = imread(path)
        img = img[:,:,:3]
        input_img = self.transform_input(img)

        gt_img = self.transform_gt(img)
        gt_img = 2.0 * gt_img - 1
        return input_img, gt_img

    def load_albedo_img(self, path):
        img = imread(path)
        img = img[:,:,:3]

        img = self.transform_gt(img)
        img = img*2.0-1
        return img

    def load_mask_img(self, path):
        img = imread(path)
        if img.size != img.shape[0]*img.shape[1]:
            img = img[:,:,0]
        img = (img==255)
        img = np.asarray(img, dtype=np.float32)
        return torch.FloatTensor(img).unsqueeze(0)

    def load_cam_proj_txt(self, path):
        proj = np.loadtxt(path)
        proj = np.asarray(proj)
        T = np.asarray([[0,0,-0.5,0],[0,0.5,0,0],[0.5,0,0,0],[0,0,0,1]])
        n_proj = np.matmul(proj, T)
        proj = np.reshape(n_proj, [1, 12])
        proj = np.divide(np.subtract(proj, self.mean_p), self.std_p)
        return torch.FloatTensor(proj).squeeze(0)


    def __getitem__(self, index):

        image_name = self.image_list[index][0]
        cate_model_name = image_name.split('/')[0] + '/' + image_name.split('/')[1]
        shape_id = self.latent_label[cate_model_name]

        fc = self.fc[shape_id]
        fs = self.fs[shape_id]
        fa = self.fa[shape_id]

        image_fn = ['%s/%s_composite.png' % (self.img_dir, image_name)]
        albedo_fn = ['%s/%s_albedo.png' % (self.img_dir, image_name)]
        cam_fn = ['%s/%s_camera.txt' % (self.img_dir, image_name)]
        mask_fn = ['%s/%s_mask.png' % (self.img_dir, image_name)]

        input_img, gt_img = self.load_input_img(image_fn[0])
        proj_mats = self.load_cam_proj_txt(cam_fn[0])
        mask_img = self.load_mask_img(mask_fn[0])
        albedo_img = self.load_albedo_img(albedo_fn[0])

        return input_img, gt_img, albedo_img, mask_img, proj_mats, fc, fs, fa, image_name

    def __len__(self):
        return len(self.image_list)
