

import os
import numpy as np
import argparse
from model import Pretrain

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", action="store", dest="epoch", default=150, type=int, help="Epoch to train [0]")
parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.0001, type=float, help="Learning rate for adam [0.0001]")
parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float, help="Momentum term of adam [0.5]")
parser.add_argument("--batch_size", action="store", dest="batch_size", default=128, type=int, help="Batch size")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint", help="Directory name to save the checkpoints [checkpoint]")
parser.add_argument("--synthetic_img_dir", action="store", dest="synthetic_img_dir", default="image_data/syn/", help="Root directory of dataset [data]")
parser.add_argument("--real_img_dir", action="store", dest="real_img_dir", default="image_data/real/", help="Root directory of dataset [data]")
parser.add_argument("--rendering_reso", action="store", dest="rendering_reso", default=128, type=int, help="rendering image size [128]")
parser.add_argument("--train", action="store_true", dest="train", default=True, help="True for training, False for testing [False]")
parser.add_argument("--fine_tuning", action="store_true", dest="fine_tuning", default=False, help="True for getting latent codes [False]")
parser.add_argument("--resume", action="store", dest="resume", default=False, type=str, help="Resume training")
parser.add_argument('--num_workers', type=int, default=4, help='number of workers') 
parser.add_argument('--num_views', type=int, default=20, help='number of image views [20]')    

config = parser.parse_args()

if config.fine_tuning:
	finetune_real = FineTune(config)
	if config.train:
		finetune_real.train()
else:
	pretrain_synthetic = Pretrain(config)
	if config.train:
		pretrain_synthetic.train()
