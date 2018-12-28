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



import visdom
import numpy as np
vis = visdom.Visdom(env='main')
vis2 = visdom.Visdom(env='exp')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='./dataset/image_2939x200x64x64_stand.npy')
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--sample_size', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--learn_rate', type=float, default=1e-5)
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


rec = []
for epoch in range(2000):
    l = len(data_loader)
    history = None
    for i,(texts, styles, target) in enumerate(data_loader):
        texts = texts.cuda()*2-1
        styles = styles.cuda()*2-1
        target = target.cuda()*2-1
        model.set_input(texts, styles, target)
        model.optimize_parameters()
        _loss_D, _loss_G = None, None
        idx = l*epoch+i

        if i%args.disp_freq == 0:
            _real_img = model.real_img[0].cpu().detach()*.5+.5
            _fake_img = model.fake_imgs[0][0].cpu().detach()*.5+.5
            _loss_D = model.loss_D.cpu().detach()
            _loss_G = model.loss_G.cpu().detach()
            _texts = model.texts[0].unsqueeze(1).cpu().detach()*.5+.5
            _styles = model.styles[0].unsqueeze(1).cpu().detach()*.5+.5
            _texts_cmp = torch.cat(
                    (model.texts[0].unsqueeze(1), 
                        model.fake_imgs[0][0].unsqueeze(0).unsqueeze(0).expand(16, 2, -1, -1))
                    , 1).cpu()*.5+.5
            _styles_cmp = torch.cat(
                    (model.styles[0].unsqueeze(1), 
                        model.fake_imgs[0][0].unsqueeze(0).unsqueeze(0).expand(16, 2, -1, -1))
                    , 1).cpu()*.5+.5
            vis.images(_real_img, win='real_img')
            vis.images(_fake_img, win='fake_img')
            vis.images(_texts, win='texts')
            vis.images(_styles, win='styles')
            vis.images(_texts_cmp, win= 'texts_cmp')
            vis.images(_styles_cmp, win= 'styles_cmp')
            pair = torch.cat((_real_img, _fake_img.unsqueeze(0)), 0)
            if history is None:
                history = pair
            elif history.shape[0]<80:
                history = torch.cat((pair, history), 0)
            else:
                history = torch.cat((pair, torch.split(history,[78,2],0)[0]), 0)
            vis.images(history.unsqueeze(1), win='his')
            print("D=",_loss_D,"G=", _loss_G)
            loss_win = vis.line(
                    Y = torch.stack([_loss_G,_loss_D],0).unsqueeze(0), 
                    X = np.array([idx]), 
                    win = 7 if idx == 0 else loss_win, 
                    update='append' if idx != 0 else None)
            if idx%args.save_freq == 0:
                model.save_networks('latest_5')
                pass
