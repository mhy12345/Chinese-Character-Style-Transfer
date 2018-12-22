from data import PairedDataset,CrossDataset
from model import CrossModel
import argparse
import logging
import torch
import torch.utils.data
import torch.optim as optim
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s') 


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

dataset = CrossDataset()
dataset.initialize(args)
data_loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = args.batch_size,
        shuffle=True,
        num_workers = 3)

model = CrossModel()
model.initialize(args)
model = model


rec = []
for epoch in range(2000):
    for i,(texts, styles, target) in enumerate(data_loader):
        texts = texts*2-1
        styles = styles*2-1
        target = target*2-1
        model.set_input(texts, styles, target)
        model.optimize_parameters()
        _loss_D, _loss_G = None, None

        if i%args.disp_freq == 0:
            _real_img = model.real_A[0].unsqueeze(1).cpu().detach().numpy()*.5+.5
            _fake_img = model.fake_A[0].unsqueeze(1).cpu().detach().numpy()*.5+.5
            _loss_D = model.loss_D.cpu().detach().numpy()
            _loss_G = model.loss_G.cpu().detach().numpy()
            vis.images(_real_img, win=1)
            vis.images(_real_B, win=2)
            vis.images(_fake_img, win=3)
            vis.images(_fake_B, win=4)
            print("D=",_loss_D,"G=", _loss_G)
