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
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learn_rate', type=float, default=1e-4)
args = parser.parse_args()

dataset = SimpleDataset()
dataset.initialize(args)
data_loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = args.batch_size,
        shuffle=True,
        num_workers = 1)

model = SmartModel().cuda()
optimizer = optim.Adam(model.parameters(), lr=args.learn_rate)

for epoch in range(2000):
    for i,(content_imgs, style_imgs, target) in enumerate(data_loader):
        content_imgs = content_imgs.cuda()
        style_imgs = style_imgs.cuda()
        target = target.cuda()
        pred = model(content_imgs, style_imgs)
        assert(pred.shape == target.shape)
        loss = (pred - target).abs().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _pred = pred[0].cpu().detach().numpy()
        #print(np.isnan(_pred).astype(np.float32).mean())
        #print((_pred>=0).astype(np.float32).mean())
        vis.image(pred[0].cpu().detach().numpy(), win=1)
        vis.image(target[0].cpu().detach().numpy(), win=2)
        vis.images(content_imgs[0].unsqueeze(1).cpu().detach().numpy(), win=3)
        vis.images(style_imgs[0].unsqueeze(1).cpu().detach().numpy(), win=4)
        print(i, loss.cpu())
