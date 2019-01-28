from data import PairedDataset,CrossDataset
from models import CrossModel
import random
import argparse
import logging
import os
import torch
import torch.utils.data
import torch.optim as optim
from options.train_options import TrainOptions
import numpy as np
import vistool
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s') 
logger = logging.getLogger(__name__)


opt = TrainOptions().parse() 
opt.name = 'exp'
opt.sample_size=20
print(opt)

dataset = CrossDataset()
dataset.initialize(opt)
data_loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = opt.batch_size,
        shuffle=True,
        num_workers = 3)

model = CrossModel()
model.initialize(opt)
model = model.cuda()
model.train()

vistool = vistool.VisTool(opt.name+'_main')
vistool.register_data('results', 'images')
vistool.register_window('results', 'images', source='results')

if opt.load_model:
    try:
        model.load_networks('latest', opt.ignore_pattern)
        logger.info("Model loaded...")
        pass
    except RuntimeError as e:
        logger.warn("Cannot load network! %s"%str(e))
        pass
    except FileNotFoundError:
        logger.warn("Cannot load network! %s"%str(e))
        pass
else:
    logger.info("Model init...")

for i,(texts, styles, target) in enumerate(data_loader):
    texts = texts.cuda()*2-1
    styles = styles.cuda()*2-1
    target = target.cuda()*2-1
    t1 = []
    t2 = []
    t0 = []
    for i in range(32):
        t = random.randint(0,1)
        t1.append(t)
        t2.append(1-t)
        t0.append(-1)
    t0 = torch.tensor(np.array(t0).astype(np.float32)).cuda()
    t1 = torch.tensor(np.array(t1).astype(np.float32)).cuda()/8*2
    t2 = torch.tensor(np.array(t2).astype(np.float32)).cuda()/8*2


    r = []
    for i in range(8):
        for j in range(8):
            res = model.netG.genFont(texts, styles, t0+t1*i+t2*j).detach()
            r.append(torch.split(res,1,1)[0].squeeze(0))
    r = torch.cat(r, 0).unsqueeze(1)
    vistool.update('results', r)
    vistool.sync()
