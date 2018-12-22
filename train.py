from data import PairedDataset
from model import SmartModel
import argparse
import logging
import torch
import torch.utils.data
import torch.optim as optim
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s') 

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import visdom
import numpy as np
vis = visdom.Visdom(env='main')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='./dataset/image_100x100x64x64_stand.npy')
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--sample_size', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--learn_rate', type=float, default=2*1e-4)
parser.add_argument('--pool_size', type=int, default=500)
parser.add_argument('--use_lsgan', type=bool, default=False)
parser.add_argument('--rec_freq', type=int, default=100)
parser.add_argument('--disp_freq', type=int, default=5)
parser.add_argument('--style_channels', type=int, default=16)

parser.add_argument('--encoder_model', type=str, default='resnet_encoder')
parser.add_argument('--transform_model', type=str, default='resnet')

args = parser.parse_args()

dataset = PairedDataset()
dataset.initialize(args)
data_loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = args.batch_size,
        shuffle=True,
        num_workers = 3)

model = SmartModel()
model.initialize(args)
model = model.cuda()


rec = []
for epoch in range(2000):
    for i,(imgs_A, imgs_B) in enumerate(data_loader):
        imgs_A = imgs_A.cuda()*2-1
        imgs_B = imgs_B.cuda()*2-1
        model.set_input(imgs_A, imgs_B)
        model.optimize_parameters()
        _loss_D, _loss_G = None, None

        if i%args.disp_freq == 0:
            _real_A = model.real_A[0].unsqueeze(1).cpu().detach().numpy()*.5+.5
            _real_B = model.real_B[0].unsqueeze(1).cpu().detach().numpy()*.5+.5
            _fake_A = model.fake_A[0].unsqueeze(1).cpu().detach().numpy()*.5+.5
            _fake_B = model.fake_B[0].unsqueeze(1).cpu().detach().numpy()*.5+.5
            if hasattr(model, 'loss_D'):
                _loss_D = model.loss_D.cpu().detach().numpy()
            if hasattr(model, 'loss_G'):
                _loss_G = model.loss_G.cpu().detach().numpy()
            vis.images(_real_A, win=1)
            vis.images(_real_B, win=2)
            vis.images(_fake_A, win=3)
            vis.images(_fake_B, win=4)
            print("D=",_loss_D,"G=", _loss_G)
