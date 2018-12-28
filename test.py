
from data import PairedDataset,CrossDataset
from model import CrossModel
import argparse
import logging
import torch
import torch.utils.data
import torch.optim as optim
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s') 

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



import random
import visdom
import numpy as np
vis = visdom.Visdom(env='main')
vis2 = visdom.Visdom(env='exp')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='./dataset/image_2939x200x64x64_stand.npy')
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--sample_size', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--learn_rate', type=float, default=1e-4)
parser.add_argument('--pool_size', type=int, default=500)
parser.add_argument('--use_lsgan', type=bool, default=False)
parser.add_argument('--rec_freq', type=int, default=100)
parser.add_argument('--disp_freq', type=int, default=10)
parser.add_argument('--save_freq', type=int, default=50)
parser.add_argument('--style_channels', type=int, default=32)

parser.add_argument('--im2vec_model', type=str, default='resnet')
parser.add_argument('--im2im_model', type=str, default='resnet')
parser.add_argument('--vec2im_model', type=str, default='conv')
parser.add_argument('--transform_model', type=str, default='resnet')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
parser.add_argument('--name', type=str, default='./main')

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
model = model.cuda()

try:
    model.load_networks('latest_5')
    print("Model loaded...")
    pass
except RuntimeError:
    print("Cannot load network!")
    pass


while True:
    texts, styles, target = dataset[random.randint(0,len(dataset)-1)]
    texts, styles, target = torch.tensor(texts).cuda(), torch.tensor(styles).cuda(), torch.tensor(target).cuda()
    t1 = []
    t2 = []
    for i in range(32):
        t = random.randint(-1,1)
        t1.append(t)
        t2.append(2-t)
    t1 = torch.tensor(np.array(t1).astype(np.float32)).cuda()/8
    t2 = torch.tensor(np.array(t2).astype(np.float32)).cuda()/8

    r = []
    for i in range(8):
        for j in range(8):
            res = model.netG.genFont(texts, t1*i+t2*j, style_count = 8).detach()
            r.append(torch.split(res,1,1)[0].squeeze(0))
    r = torch.cat(r, 0).unsqueeze(1)
    print(r.shape)
    vis2.images(r)
