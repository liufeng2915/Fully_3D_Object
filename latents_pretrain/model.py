

import os
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import datetime
import time
import mcubes
from dataset import *
import network
from utils import *

class PretrainDecoders(object):
    def __init__(self, config):

        #---#
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        #---#    
        self.sample_reso = config.sample_reso
        if self.sample_reso == 16:
            self.batch_size = 32
        elif self.sample_reso == 32:
            self.batch_size = 16
        elif self.sample_reso == 64:
            self.batch_size = 4

        #---#
        self.epoch = config.epoch
        self.fc_dim = 3      # Dimension of the category code
        self.fa_dim = 256    # Dimension of the albedo code
        self.fs_dim = 256    # Dimension of the shape code
        self.point_dim = 3
        self.num_branch = 12
        self.num_category = 14  # 13 categories + 1 unknown categories

        #---#
        self.data_dir = config.data_dir
        fid = open('config/data_list.txt')
        self.data_list = fid.read().splitlines()
        fid.close()
        self.num_samples = len(self.data_list)
        fid = open('config/data_label.txt')
        self.data_label = fid.read().splitlines()
        fid.close()

        #---# data
        data = PointValueData(data_dir=config.data_dir, data_list=self.data_list, data_label=self.data_label, sample_reso=self.sample_reso)
        if config.train:
            self.data_loader = DataLoader(data, num_workers=4, batch_size=self.batch_size, shuffle=True, drop_last=True, pin_memory=True)
        else:
            self.data_loader = DataLoader(data, num_workers=4, batch_size=1, shuffle=False)

        #---# network
        self.auto_decoder = True    # training without encoder
        self.pos_encoding = False   # positional encoding
        self.model = network.pretrain_network(fc_dim=self.fc_dim, fa_dim=self.fa_dim, fs_dim=self.fs_dim, num_branch=self.num_branch, point_dim=self.point_dim, \
            num_category=self.num_category, num_samples=self.num_samples, auto_decoder=self.auto_decoder, pos_encoding=self.pos_encoding)
        self.model.to(self.device)
        print(self.model)

        #---# save model
        self.checkpoint_path = config.checkpoint_dir
        self.checkpoint_encoder_name = 'Encoder.pth'
        self.checkpoint_shapedecoder_name = 'ShapeDecoder.pth'
        self.checkpoint_albedodecoder_name = 'AlbedoDecoder.pth'

        #---# optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))

        #---# loss
        self.value_loss = torch.nn.MSELoss()    # occupancy value loss 
        self.class_loss = torch.nn.CrossEntropyLoss().to(self.device)


        #---# log
        sig = str(datetime.datetime.now())
        logs_dir = 'logs/'
        self.writer = SummaryWriter('%s/logs/%s' % (logs_dir, sig))

        #---# inferance mesh
        self.isosurface_threshold = 0.5   # marching cubes threshold
        self.infer_resolution = 256

    def train(self):

        #load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_name = fin.readline().strip()
            fin.close()
            encoder_state_dict  = torch.load(self.checkpoint_path+'/'+model_name+'/'+self.checkpoint_encoder_name)
            self.model.Encoder.load_state_dict(encoder_state_dict['Encoder_state_dict'])
            albedodecoder_state_dict  = torch.load(self.checkpoint_path+'/'+model_name+'/'+self.checkpoint_albedodecoder_name)
            self.model.AlbedoDecoder.load_state_dict(albedodecoder_state_dict['AlbedoDecoder_state_dict'])
            shapedecoder_state_dict  = torch.load(self.checkpoint_path+'/'+model_name+'/'+self.checkpoint_shapedecoder_name)
            self.model.ShapeDecoder.load_state_dict(shapedecoder_state_dict['ShapeDecoder_state_dict'])
            print(" [*] Load SUCCESS")
            print(model_name)
        else:
            print(" [!] Load failed...")

        #---#
        print("-------------------------------\n")
        print("Start training\n")
        print("-------------------------------\n")
        start_time = time.time()
        self.model.train()
        num_iter = 0
        for epoch in range(0, self.epoch):

            avg_loss_occ = 0
            avg_loss_albedo = 0
            avg_loss_class = 0
            avg_num = 0
            for it, data in enumerate(self.data_loader):

                colored_voxel, point_coord, point_values, point_colors, class_label, sample_name, sample_index = data
                colored_voxel = colored_voxel.to(self.device)
                point_coord = point_coord.to(self.device)
                point_values = point_values.to(self.device)
                point_colors = point_colors.to(self.device)
                class_label = class_label.to(self.device)

                self.model.zero_grad()
                fc, fa, fs, fclass, branch_values, values, albedo = self.model(sample_index, \
                    colored_voxel, point_coord, is_training=True)

                err_class = self.class_loss(fclass, class_label)
                err_occ = self.value_loss(values, point_values)
                err_albedo = self.value_loss(albedo, point_colors)
                err =  err_occ + err_class + err_albedo

                err.backward()
                self.optimizer.step()

                #---#
                avg_loss_class += err_class.item()
                avg_loss_albedo += err_albedo.item()
                avg_loss_occ += err_occ.item()
                avg_num += 1
                num_iter += 1
                self.writer.add_scalar('loss/loss', err, num_iter)
                self.writer.add_scalar('loss/loss_class', err_class, num_iter)
                self.writer.add_scalar('loss/loss_occ', err_occ, num_iter)
                self.writer.add_scalar('loss/loss_albedo', err_albedo, num_iter)
                if (it%10)==0:
                    self.writer.add_histogram('embedding/fc', fc, num_iter)
                    self.writer.add_histogram('embedding/fa', fa, num_iter)
                    self.writer.add_histogram('embedding/fs', fs, num_iter)

            print(str(self.sample_reso)+" Epoch: [%2d/%2d], time: %4.4f, loss_class: %.6f, loss_albedo: %.6f, loss_occupancy: %.6f" \
            % (epoch, self.epoch, time.time() - start_time, avg_loss_class/avg_num, avg_loss_albedo/avg_num, avg_loss_occ/avg_num))

            #---# save model
            save_dir = os.path.join(self.checkpoint_path,str(self.sample_reso)+'-'+str(epoch))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save({'Encoder_state_dict': self.model.Encoder.state_dict()}, save_dir+'/'+self.checkpoint_encoder_name)
            torch.save({'AlbedoDecoder_state_dict': self.model.AlbedoDecoder.state_dict()}, save_dir+'/'+self.checkpoint_albedodecoder_name)
            torch.save({'ShapeDecoder_state_dict': self.model.ShapeDecoder.state_dict()}, save_dir+'/'+self.checkpoint_shapedecoder_name)
            checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
            fout = open(checkpoint_txt, 'w')
            fout.write(str(self.sample_reso)+'-'+str(epoch)+"\n")
            fout.close()


    def get_latent(self):

        #load trained models
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_name = fin.readline().strip()
            fin.close()
            encoder_state_dict  = torch.load(self.checkpoint_path+'/'+model_name+'/'+self.checkpoint_encoder_name)
            self.model.Encoder.load_state_dict(encoder_state_dict['Encoder_state_dict'])
            print(" [*] Load SUCCESS")
            print(model_name)
        else:
            print(" [!] Load failed...")

        self.model.eval()
        fc = np.zeros((self.num_samples, self.fc_dim))
        fa = np.zeros((self.num_samples, self.fa_dim))
        fs = np.zeros((self.num_samples, self.fs_dim))
        for it, data in enumerate(self.data_loader):
            colored_voxel, _, _, _, _, _, sample_index = data
            colored_voxel = colored_voxel.to(self.device)
            t_fc, t_fa, t_fs = self.model(sample_index, colored_voxel, None, is_training=False)
            fc[it] = t_fc.data.cpu().numpy()
            fa[it] = t_fa.data.cpu().numpy()
            fs[it] = t_fs.data.cpu().numpy()

        scipy.io.savemat(self.checkpoint_path+'/'+model_name+'/pretained_latents.mat', {'fc':fc, 'fa':fa, 'fs':fs})


    def generate_mesh(self):

        #load trained models
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_name = fin.readline().strip()
            fin.close()
            encoder_state_dict  = torch.load(self.checkpoint_path+'/'+model_name+'/'+self.checkpoint_encoder_name)
            self.model.Encoder.load_state_dict(encoder_state_dict['Encoder_state_dict'])
            shapedecoder_state_dict  = torch.load(self.checkpoint_path+'/'+model_name+'/'+self.checkpoint_shapedecoder_name)
            self.model.ShapeDecoder.load_state_dict(shapedecoder_state_dict['ShapeDecoder_state_dict'])
            print(" [*] Load SUCCESS")
            print(model_name)
        else:
            print(" [!] Load failed...")

        # results folder
        save_dir = 'results'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        #
        dima = 32
        voxel_size = dima*dima*dima
        multiplier = int(self.infer_resolution/dima)
        multiplier2 = multiplier*multiplier
        multiplier3 = multiplier*multiplier*multiplier
        aux_x = np.zeros([dima,dima,dima],np.int32)
        aux_y = np.zeros([dima,dima,dima],np.int32)
        aux_z = np.zeros([dima,dima,dima],np.int32)
        for i in range(dima):
            for j in range(dima):
                for k in range(dima):
                    aux_x[i,j,k] = i*multiplier
                    aux_y[i,j,k] = j*multiplier
                    aux_z[i,j,k] = k*multiplier
        coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
        for i in range(multiplier):
            for j in range(multiplier):
                for k in range(multiplier):
                    coords[i*multiplier2+j*multiplier+k,:,:,:,0] = aux_x+i
                    coords[i*multiplier2+j*multiplier+k,:,:,:,1] = aux_y+j
                    coords[i*multiplier2+j*multiplier+k,:,:,:,2] = aux_z+k
        coords = (coords+0.5)/self.infer_resolution*2.0-1.0
        coords = np.reshape(coords,[multiplier3,voxel_size,3])

        self.model.eval()
        for it, data in enumerate(self.data_loader):
            if it > 50:
                break
            colored_voxel, _, _, _, _, sample_name, sample_index = data
            colored_voxel = colored_voxel.to(self.device)
            t_fc, t_fa, t_fs = self.model(sample_index, colored_voxel, None, is_training=False)
            voxel = np.zeros([self.infer_resolution+2,self.infer_resolution+2,self.infer_resolution+2],np.float32)
            for i in range(multiplier):
                for j in range(multiplier):
                    for k in range(multiplier):
                        print(it,i,j,k)
                        minib = i*multiplier2+j*multiplier+k
                        input_points = torch.from_numpy(coords[minib]).to(self.device)
                        input_points = input_points.unsqueeze(0)
                        _, values = self.model.ShapeDecoder(t_fc, t_fs, input_points)
                        voxel[aux_x+i+1,aux_y+j+1,aux_z+k+1] = np.reshape(values.data.cpu().numpy(), [dima,dima,dima])
            vertices, triangles = mcubes.marching_cubes(voxel, self.isosurface_threshold)
            write_ply_triangle(save_dir+'/'+sample_name[0].split('/')[-1]+'.ply', vertices, triangles)


    def generate_mesh_from_trained_models(self):

        shapedecoder_state_dict  = torch.load('models/ShapeDecoder.pth')
        self.model.ShapeDecoder.load_state_dict(shapedecoder_state_dict['ShapeDecoder_state_dict'])
        mat_file = scipy.io.loadmat('models/pretained_latents.mat')
        fc = mat_file['fc']
        fa = mat_file['fa']
        fs = mat_file['fs']

        # results folder
        save_dir = 'results'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        #
        dima = 32
        voxel_size = dima*dima*dima
        multiplier = int(self.infer_resolution/dima)
        multiplier2 = multiplier*multiplier
        multiplier3 = multiplier*multiplier*multiplier
        aux_x = np.zeros([dima,dima,dima],np.int32)
        aux_y = np.zeros([dima,dima,dima],np.int32)
        aux_z = np.zeros([dima,dima,dima],np.int32)
        for i in range(dima):
            for j in range(dima):
                for k in range(dima):
                    aux_x[i,j,k] = i*multiplier
                    aux_y[i,j,k] = j*multiplier
                    aux_z[i,j,k] = k*multiplier
        coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
        for i in range(multiplier):
            for j in range(multiplier):
                for k in range(multiplier):
                    coords[i*multiplier2+j*multiplier+k,:,:,:,0] = aux_x+i
                    coords[i*multiplier2+j*multiplier+k,:,:,:,1] = aux_y+j
                    coords[i*multiplier2+j*multiplier+k,:,:,:,2] = aux_z+k
        coords = (coords+0.5)/self.infer_resolution*2.0-1.0
        coords = np.reshape(coords,[multiplier3,voxel_size,3])

        for isample in range(self.num_samples):
            if isample > 50:
                break
            t_fc = torch.from_numpy(fc[isample:isample+1]).to(self.device)
            t_fa = torch.from_numpy(fa[isample:isample+1]).to(self.device)
            t_fs = torch.from_numpy(fs[isample:isample+1]).to(self.device)
            voxel = np.zeros([self.infer_resolution+2,self.infer_resolution+2,self.infer_resolution+2],np.float32)
            for i in range(multiplier):
                for j in range(multiplier):
                    for k in range(multiplier):
                        print(isample,i,j,k)
                        minib = i*multiplier2+j*multiplier+k
                        input_points = torch.from_numpy(coords[minib]).to(self.device)
                        input_points = input_points.unsqueeze(0)
                        _, values = self.model.ShapeDecoder(t_fc, t_fs, input_points)
                        voxel[aux_x+i+1,aux_y+j+1,aux_z+k+1] = np.reshape(values.data.cpu().numpy(), [dima,dima,dima])
            vertices, triangles = mcubes.marching_cubes(voxel, self.isosurface_threshold)
            write_ply_triangle(save_dir+'/'+self.data_list[isample].split('/')[-1]+'.ply', vertices, triangles)
