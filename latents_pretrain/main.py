

import os
import numpy as np
import argparse
from model import PretrainDecoders

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", action="store", dest="epoch", default=50, type=int, help="Epoch to train [0]")
parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.0005, type=float, help="Learning rate for adam [0.0001]")
parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float, help="Momentum term of adam [0.5]")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint", help="Directory name to save the checkpoints [checkpoint]")
parser.add_argument("--data_dir", action="store", dest="data_dir", default="data/", help="Root directory of dataset [data]")
parser.add_argument("--sample_reso", action="store", dest="sample_reso", default=16, type=int, help="Voxel resolution for coarse-to-fine training [64]")
parser.add_argument("--train", action="store_true", dest="train", default=False, help="True for training, False for testing [False]")
parser.add_argument("--get_latent", action="store_true", dest="get_latent", default=False, help="True for getting latent codes [False]")

config = parser.parse_args()

pretrain_decoders = PretrainDecoders(config)

if config.train:
	pretrain_decoders.train()
elif config.get_latent:
	pretrain_decoders.get_latent()
else:
	pretrain_decoders.generate_mesh()
    