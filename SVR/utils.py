

import numpy as np  
import scipy.io  
import os

def PrepareSynData(config):

    ## mean and std of camera parameters
    cam_file = scipy.io.loadmat('data/normal_cam.mat')
    config.mean_p = cam_file['mean_cam'] # 1*12
    config.std_p = cam_file['std_cam']   # 1*12

    ## load pre-trained latent codes
    mat_file = scipy.io.loadmat('data/pretained_latents.mat')
    config.fa = mat_file['fa']
    config.fc = mat_file['fc']
    config.fs = mat_file['fs']
    fileID = open('data/data_list.txt')
    curr_id = 0
    latent_label = {}
    for line in fileID:
        line = line.strip()
        latent_label[line[:-4]] = curr_id
        curr_id += 1
    fileID.close() 
    config.latent_label = latent_label

    if config.train:
        fid = open('data/train_list.txt')
        train_list = fid.read().splitlines()
        fid.close()

        image_list = []
        cate_list = os.listdir(config.synthetic_img_dir)
        for t_cate in cate_list:
            image_model_list = os.listdir(config.synthetic_img_dir+'/'+t_cate+'/')
            for t_model in image_model_list:
                cate_model_name = t_cate + '/' + t_model
                if cate_model_name in train_list:
                    for img_idx in range(config.num_views):
                        temp_img_list = ['%s/%s_view%04d' % (cate_model_name, t_model, img_idx)]
                        image_list.append(temp_img_list)
        print('There are %d images for training' % len(image_list))
        config.image_list = image_list
        
    return config



