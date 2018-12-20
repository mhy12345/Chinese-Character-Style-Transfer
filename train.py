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
parser.add_argument('--learn_rate', type=float, default=1e-2)
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
        _pred = pred[0].cpu().detach().numpy()

        if i%100 == 0:
            rec.append(_pred)
        vis.image(pred[0].cpu().detach().numpy(), win=1)
        vis.image(target[0].cpu().detach().numpy(), win=2)
        vis.images(content_imgs[0].unsqueeze(1).cpu().detach().numpy(), win=3)
        vis.images(style_imgs[0].unsqueeze(1).cpu().detach().numpy(), win=4)
        vis.images(np.array(rec), win=5)
        print("D=",model.loss_D,"G=", model.loss_G)
