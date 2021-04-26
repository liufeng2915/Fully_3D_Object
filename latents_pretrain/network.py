

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad

#---#  The Encoder and ShapeDecoder networks are adopted from BAE-NET (https://github.com/czq142857/BAE-NET)

class Encoder(nn.Module):
    def __init__(self, fc_dim=3, fa_dim=256, fs_dim=256, num_category=13):
        super(Encoder, self).__init__()
        ef_dim = 32
        self.f_dim = fc_dim + fa_dim + fs_dim
        self.fc_dim = fc_dim
        self.fa_dim = fa_dim
        self.fs_dim = fs_dim
        self.conv_1 = nn.Conv3d(3, ef_dim, 4, stride=2, padding=1, bias=False)
        self.in_1 = nn.InstanceNorm3d(ef_dim)
        self.conv_2 = nn.Conv3d(ef_dim, ef_dim*2, 4, stride=2, padding=1, bias=False)
        self.in_2 = nn.InstanceNorm3d(ef_dim*2)
        self.conv_3 = nn.Conv3d(ef_dim*2, ef_dim*4, 4, stride=2, padding=1, bias=False)
        self.in_3 = nn.InstanceNorm3d(ef_dim*4)
        self.conv_4 = nn.Conv3d(ef_dim*4, ef_dim*8, 4, stride=2, padding=1, bias=False)
        self.in_4 = nn.InstanceNorm3d(ef_dim*8)
        self.conv_5 = nn.Conv3d(ef_dim*8, self.f_dim, 4, stride=1, padding=0, bias=True)
        self.linear = nn.Linear(fc_dim, num_category)

        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.xavier_uniform_(self.conv_3.weight)
        nn.init.xavier_uniform_(self.conv_4.weight)
        nn.init.xavier_uniform_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias,0)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear.bias,0)

    def forward(self, colored_voxel):

        e_1 = self.in_1(self.conv_1(colored_voxel))
        e_1 = F.leaky_relu(e_1, negative_slope=0.02, inplace=True)

        e_2 = self.in_2(self.conv_2(e_1))
        e_2 = F.leaky_relu(e_2, negative_slope=0.02, inplace=True)
        
        e_3 = self.in_3(self.conv_3(e_2))
        e_3 = F.leaky_relu(e_3, negative_slope=0.02, inplace=True)

        e_4 = self.in_4(self.conv_4(e_3))
        e_4 = F.leaky_relu(e_4, negative_slope=0.02, inplace=True)

        e_5 = self.conv_5(e_4)
        e_5 = e_5.view(-1, self.f_dim)
        fc,fa,fs = torch.split(e_5,[self.fc_dim,self.fa_dim,self.fs_dim],dim=-1)
        fc = F.normalize(fc, p=2, dim=1)
        fa = F.normalize(fa, p=2, dim=1)
        fs = F.normalize(fs, p=2, dim=1)
        fclass = self.linear(fc)

        return fc,fa,fs,fclass


class LatentLayer(nn.Module):
    def __init__(self, fc_dim=3, fa_dim=256, fs_dim=256, num_category=13, num_samples=None):
        super(LatentLayer, self).__init__()

        self.c_latent = nn.Parameter(torch.FloatTensor(num_samples, fc_dim))
        self.a_latent = nn.Parameter(torch.FloatTensor(num_samples, fa_dim))
        self.s_latent = nn.Parameter(torch.FloatTensor(num_samples, fs_dim))

        self.linear = nn.Linear(fc_dim, num_category)

        nn.init.xavier_normal_(self.c_latent)
        nn.init.xavier_normal_(self.a_latent)
        nn.init.xavier_normal_(self.s_latent)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear.bias,0)

    def codes(self):
        return self.c_latent, self.a_latent, self.s_latent

    def forward(self, sample_index):

        fc = self.c_latent[sample_index]
        fa = self.a_latent[sample_index]
        fs = self.s_latent[sample_index]
        fclass = self.linear(fc)

        return fc,fa,fs,fclass


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



class pretrain_network(nn.Module):
    def __init__(self, fc_dim=3, fa_dim=256, fs_dim=256, num_branch=12, point_dim=3, num_category=13, num_samples=None, auto_decoder=True, pos_encoding=False):
        super(pretrain_network, self).__init__()

        self.auto_decoder = auto_decoder
        if auto_decoder:
            self.Encoder = LatentLayer(fc_dim, fa_dim, fs_dim, num_category, num_samples)
        else:
            self.Encoder = Encoder(fc_dim, fa_dim, fs_dim, num_category)
        self.ShapeDecoder = ShapeDecoder(fc_dim, fs_dim, num_branch, point_dim)
        self.AlbedoDecoder = AlbedoDecoder(fc_dim, fa_dim, fs_dim, num_branch, point_dim, pos_encoding)

    def forward(self, sample_index, colored_voxel, point_coord, is_training=True):
        if is_training:
            if self.auto_decoder:
                fc,fa,fs,fclass = self.Encoder(sample_index)
            else:
                fc,fa,fs,fclass = self.Encoder(colored_voxel)

            branch_values, values = self.ShapeDecoder(fc, fs, point_coord)
            albedo = self.AlbedoDecoder(fc, fa, fs, branch_values.clone(), point_coord)

            return fc, fa, fs, fclass, branch_values, values, albedo
        else:
            if self.auto_decoder:
                #c_latent, a_latent, s_latent = self.Encoder.codes()
                c_latent, a_latent, s_latent, _ = self.Encoder(sample_index)
            else:
                c_latent, a_latent, s_latent, _ = self.Encoder(colored_voxel)
            
            return c_latent, a_latent, s_latent
        

    