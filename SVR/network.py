

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from resnet_gn import resnet18

class ImageEncoder(nn.Module):
    def __init__(self, fc_dim=3, fa_dim=256, fs_dim=256, light_dim=27, cam_dim=12):
        super(ImageEncoder, self).__init__()
        self.features = resnet18()

        hidden_dim = 512
        self.features.fc = nn.Sequential()
        self.fc_class1 = nn.Linear(512, hidden_dim)
        self.fc_class2 = nn.Linear(hidden_dim, fc_dim)
        self.fc_albedo1 = nn.Linear(512, hidden_dim)
        self.fc_albedo2 = nn.Linear(hidden_dim, fa_dim)
        self.fc_shape1  = nn.Linear(512, hidden_dim)
        self.fc_shape2  = nn.Linear(hidden_dim, fs_dim)
        self.fc_cam1 = nn.Linear(512, hidden_dim)
        self.fc_cam2 = nn.Linear(hidden_dim, cam_dim)
        self.fc_light1 = nn.Linear(512, hidden_dim)
        self.fc_light2 = nn.Linear(hidden_dim, light_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        feat = self.features(x)
        #
        fc = self.relu(self.fc_class1(feat))
        fc = self.fc_class2(fc)
        #
        fa = self.relu(self.fc_albedo1(feat))
        fa = self.fc_albedo2(fa)
        #
        fs = self.relu(self.fc_shape1(feat))
        fs = self.fc_shape2(fs)
        #
        cam_ = self.relu(self.fc_cam1(feat))
        cam_ = self.fc_cam2(cam_)
        #
        light_ = self.relu(self.fc_light1(feat))
        light_ = self.fc_light2(light_)

        return fc, fa, fs, cam_, light_


class ShapeDecoder(nn.Module):
    def __init__(self, fc_dim=3, fs_dim=256, num_branch=12, point_dim=3):
        super(ShapeDecoder, self).__init__()
        self.f_dim = fc_dim + fs_dim + point_dim
        self.sd_fc1 = nn.Linear(self.f_dim, 3072)
        self.sd_fc2 = nn.Linear(3072, 384)
        self.sd_fc3 = nn.Linear(384, num_branch)

        nn.init.normal_(self.sd_fc1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.sd_fc1.bias,0)
        nn.init.normal_(self.sd_fc2.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.sd_fc2.bias,0)
        nn.init.normal_(self.sd_fc3.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.sd_fc3.bias,0)

    def forward(self, fc, fs, points):

        num_points = points.shape[1]
        fc = fc.unsqueeze(1).repeat(1,num_points,1)
        fs = fs.unsqueeze(1).repeat(1,num_points,1)
        fconca = torch.cat([fc,points,fs],dim=-1)

        l1 = self.sd_fc1(fconca)
        l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

        l2 = self.sd_fc2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

        l3 = self.sd_fc3(l2)
        l3 = torch.sigmoid(l3)
        l4 = F.max_pool1d(l3, l3.shape[-1])

        return l3, l4


class AlbedoDecoder(nn.Module):
    def __init__(self, fc_dim=3, fa_dim=256, fs_dim=256, num_branch=12, point_dim=3, pos_encoding=False):
        super(AlbedoDecoder, self).__init__()
        if pos_encoding:
            self.f_dim = fc_dim + fa_dim + fs_dim + point_dim*10
        else:
            self.f_dim = fc_dim + fa_dim + fs_dim + point_dim
        self.pos_encoding = pos_encoding
        self.num_branch = num_branch

        self.ad_fc1 = nn.Linear(self.f_dim, 512, bias=True)
        self.ad_fc2 = nn.Linear(512, 512, bias=True)
        self.ad_fc3 = nn.Linear(512, 512, bias=True)
        self.ad_fc4 = nn.Linear(512, 512, bias=True)
        self.ad_fc5 = nn.Linear(512, 512, bias=True)
        self.ad_fc6 = nn.Linear(512, 512, bias=True)
        self.ad_fc7 = nn.Linear(512, 3*num_branch, bias=True)

        nn.init.normal_(self.ad_fc1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.ad_fc1.bias,0)
        nn.init.normal_(self.ad_fc2.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.ad_fc2.bias,0)
        nn.init.normal_(self.ad_fc3.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.ad_fc3.bias,0)
        nn.init.normal_(self.ad_fc4.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.ad_fc4.bias,0)
        nn.init.normal_(self.ad_fc5.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.ad_fc5.bias,0)
        nn.init.normal_(self.ad_fc6.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.ad_fc6.bias,0)
        nn.init.normal_(self.ad_fc7.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.ad_fc7.bias,0)


    def positional_encoding(self, points): 

        shape = points.shape
        posenc_L = 5
        points_enc = []
        if posenc_L:
            freq = 2**torch.arange(posenc_L,dtype=torch.float32)*np.pi 
            freq = freq.cuda()
            spectrum = points[...,None]*freq 
            sin,cos = spectrum.sin(),spectrum.cos()
            points_enc_L = torch.cat([sin,cos],dim=-1).view(*shape[:-1],6*posenc_L) 
            points_enc.append(points_enc_L)
        points_enc = torch.cat(points_enc,dim=-1) 

        return points_enc

    def forward(self, fc, fa, fs, branch_values, points):
        
        num_points = points.shape[1]
        fc = fc.unsqueeze(1).repeat(1,num_points,1)
        fa = fa.unsqueeze(1).repeat(1,num_points,1)
        fs = fs.unsqueeze(1).repeat(1,num_points,1)
        if self.pos_encoding:
            fconca = torch.cat([fc,self.positional_encoding(points),fs,fa],dim=-1)
        else:
            fconca = torch.cat([fc,points,fs,fa],dim=-1)

        l1 = self.ad_fc1(fconca)
        l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

        l2 = self.ad_fc2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

        l3 = self.ad_fc3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

        l4 = self.ad_fc4(l3)
        l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

        l5 = self.ad_fc5(l4)
        l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

        l6 = self.ad_fc6(l5)
        l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

        l7 = self.ad_fc7(l6)
        l7 = torch.tanh(l7)

        # one hot
        max_idx = torch.argmax(branch_values, dim=-1, keepdim=True)
        branch_values.zero_().scatter_(-1, max_idx, 1)

        l7 = l7.view(l7.shape[0], l7.shape[1], self.num_branch, 3)
        l8 = torch.matmul(branch_values.unsqueeze(2), l7)
        l8 = l8.squeeze(2)

        return l8


class SingleViewRecon(nn.Module):
    def __init__(self, fc_dim=3, fa_dim=256, fs_dim=256, light_dim=27, cam_dim=12, point_dim=3, num_branch=12, pos_encoding=False):
        super(SingleViewRecon, self).__init__()

        self.ImgEncoder = ImageEncoder(fc_dim, fa_dim, fs_dim, light_dim, cam_dim)
        self.ShapeDecoder = ShapeDecoder(fc_dim, fs_dim, num_branch, point_dim)
        self.AlbedoDecoder = AlbedoDecoder(fc_dim, fa_dim, fs_dim, num_branch, point_dim, pos_encoding)

    def forward(self, img=None, is_training=True):

        if is_training:
            esti_fc, esti_fa, esti_fs, esti_cam, esti_light = self.ImgEncoder(img)
            return esti_fc, esti_fa, esti_fs, esti_cam, esti_light
        

    