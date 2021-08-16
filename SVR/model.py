

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

class Pretrain(object):
    def __init__(self, config):

        #---#
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        #---#
        self.epoch = config.epoch
        self.fc_dim = 3      # Dimension of the category code
        self.fa_dim = 256    # Dimension of the albedo code
        self.fs_dim = 256    # Dimension of the shape code
        self.point_dim = 3
        self.num_branch = 12
        self.num_category = 14  # 13 categories + 1 unknown categories

        #---#
        config = PrepareSynData(config)


        #---# data
        data = SynData(config)
        self.data_loader = DataLoader(data, num_workers=config.num_workers, batch_size=config.batch_size, shuffle=True, drop_last=True, pin_memory=True)

        #---# network
        self.pos_encoding = False   # positional encoding
        self.model = network.SingleViewRecon(fc_dim=self.fc_dim, fa_dim=self.fa_dim, fs_dim=self.fs_dim, num_branch=self.num_branch, \
            point_dim=self.point_dim, pos_encoding=self.pos_encoding)

        self.model.to(self.device)
        print(self.model)

        #---# save model
        self.checkpoint_path = config.checkpoint_dir + '/syn' 
        self.checkpoint_encoder_name = 'ImgEncoder.pth'
        self.checkpoint_shapedecoder_name = 'ShapeDecoder.pth'
        self.checkpoint_albedodecoder_name = 'AlbedoDecoder.pth'

        #---# optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))

        #---# loss
        self.criterion = torch.nn.MSELoss()    # occupancy value loss 

        #---# log
        sig = str(datetime.datetime.now())
        logs_dir = 'logs/'
        self.writer = SummaryWriter('%s/logs/%s' % (logs_dir, sig))

        #---# inferance mesh
        self.isosurface_threshold = 0.5   # marching cubes threshold
        self.infer_resolution = 256

    def train(self):

        #---#
        print("-------------------------------\n")
        print("Start training\n")
        print("-------------------------------\n")
        start_time = time.time()
        self.model.train()
        num_iter = 0
        for epoch in range(0, self.epoch):

            avg_loss_fc = 0
            avg_loss_fa = 0
            avg_loss_fs = 0
            avg_loss_cam = 0
            num_iter = 0
            for it, data in enumerate(self.data_loader):

                input_img, _, _, _, cam, fc, fs, fa, image_name = data
                input_img = input_img.to(self.device)
                cam = cam.to(self.device)
                fc = fc.to(self.device)
                fs = fs.to(self.device)
                fa = fa.to(self.device)

                self.model.zero_grad()
                esti_fc, esti_fa, esti_fs, esti_cam, _ = self.model(input_img, is_training=True)
                loss_fc = self.criterion(esti_fc, fc)
                loss_fa = self.criterion(esti_fa, fa)
                loss_fs = self.criterion(esti_fs, fs)
                loss_cam = self.criterion(esti_cam, cam)

                loss = loss_fc + loss_fa + loss_fs + loss_cam
                loss.backward()
                self.optimizer.step()

                #---#
                avg_loss_fc += loss_fc.item()
                avg_loss_fa += loss_fa.item()
                avg_loss_fs += loss_fs.item()
                avg_loss_cam += loss_cam.item()

                num_iter += 1
                self.writer.add_scalar('loss/loss', loss, num_iter)
                self.writer.add_scalar('loss/loss_fc', loss_fc, num_iter)
                self.writer.add_scalar('loss/loss_fs', loss_fs, num_iter)
                self.writer.add_scalar('loss/loss_fa', loss_fa, num_iter)
                self.writer.add_scalar('loss/loss_cam', loss_cam, num_iter)
                if (it%10)==0:
                    self.writer.add_histogram('embedding/fc', esti_fc, num_iter)
                    self.writer.add_histogram('embedding/fa', esti_fa, num_iter)
                    self.writer.add_histogram('embedding/fs', esti_fs, num_iter)

                if (it % 20) == 0:
                    print("Epoch: [%2d/%2d], Iter: [%2d], time: %4.4f, loss_fc: %.6f, loss_fa: %.6f, loss_fs: %.6f, loss_cam: %.6f" \
                    % (epoch, self.epoch, it, time.time() - start_time, avg_loss_fc/num_iter, avg_loss_fa/num_iter, avg_loss_fs/num_iter, avg_loss_cam/num_iter))

            #---# save model
            if (epoch % 20):
                save_dir = os.path.join(self.checkpoint_path,str(self.sample_reso)+'-'+str(epoch))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save({'ImgEncoder_state_dict': self.model.ImgEncoder.state_dict()}, save_dir+'/'+self.checkpoint_encoder_name)
                torch.save({'AlbedoDecoder_state_dict': self.model.AlbedoDecoder.state_dict()}, save_dir+'/'+self.checkpoint_albedodecoder_name)
                torch.save({'ShapeDecoder_state_dict': self.model.ShapeDecoder.state_dict()}, save_dir+'/'+self.checkpoint_shapedecoder_name)
                checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
                fout = open(checkpoint_txt, 'w')
                fout.write(str(self.sample_reso)+'-'+str(epoch)+"\n")
                fout.close()


    