from data import SimpleDataset
from model import SmartModel
from simple_model import SimpleModel
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
parser.add_argument('--learn_rate', type=float, default=5*1e-4)
parser.add_argument('--pool_size', type=int, default=500)
parser.add_argument('--use_lsgan', type=bool, default=False)
parser.add_argument('--rec_freq', type=int, default=100)
parser.add_argument('--disp_freq', type=int, default=5)
args = parser.parse_args()

dataset = SimpleDataset()
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
    for i,(content_imgs, style_imgs, target) in enumerate(data_loader):
        content_imgs = content_imgs.cuda()
        style_imgs = style_imgs.cuda()
        target = target.cuda()
        model.set_input(content_imgs, style_imgs, target)
        model.optimize_parameters()
        pred = model.fake_img

        if i%args.rec_freq == 0:
            _pred = pred[0].cpu().detach().numpy()
            rec.append(_pred)
            vis.images(np.array(rec), win=10)
        if i%args.disp_freq == 0:
            _pred = pred[0].cpu().detach().numpy()
            _targ = target[0].cpu().detach().numpy()
            _contents = content_imgs[0].unsqueeze(1).cpu().detach().numpy()
            _styles = style_imgs[0].unsqueeze(1).cpu().detach().numpy()
            _compare = np.concatenate([_contents, np.broadcast_to(_pred, _contents.shape), np.zeros_like(_contents)], axis=1)
            vis.image(_targ, win=1)
            vis.image(_pred, win=2)
            vis.images(_contents, win=3)
            vis.images(_styles, win=4)
            vis.images(_compare, win=5)
            print("D=",model.loss_D_fake, model.loss_D_real,"G=", model.loss_G)
