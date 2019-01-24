from data import PairedDataset,CrossDataset
from models import CrossModel
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
vistool.register_data('lossD', 'scalar_ma')
vistool.register_data('lossG', 'scalar_ma')
vistool.register_data('lossE', 'scalar_ma')
vistool.register_data('real_img', 'images')
vistool.register_data('fake_img', 'images')
vistool.register_data('texts', 'images')
vistool.register_data('styles', 'images')
vistool.register_data('texts_cmp', 'images')
vistool.register_data('styles_cmp', 'images')
vistool.register_data('history', 'image_gallery')
vistool.register_window('losses', 'lines', sources=['lossD','lossG','lossE'])
vistool.register_window('real_img', 'images', source='real_img')
vistool.register_window('fake_img', 'images', source='fake_img')
vistool.register_window('texts', 'images', source='texts')
vistool.register_window('styles', 'images', source='styles')
vistool.register_window('texts_cmp', 'images', source='texts_cmp')
vistool.register_window('styles_cmp', 'images', source='styles_cmp')
vistool.register_window('history', 'images', source='history')

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
        vistool.update('lossD',model.loss_D)
        vistool.update('lossG',model.loss_G)
        vistool.update('lossE',model.loss_E)
        if i%opt.display_freq == 0:
            _texts_cmp = torch.cat(
                    (model.texts[0].unsqueeze(1), 
                        model.fake_imgs[0][0].unsqueeze(0).unsqueeze(0).expand(opt.sample_size, 2, -1, -1))
                    , 1).cpu().detach()*.5+.5
            _styles_cmp = torch.cat(
                    (model.styles[0].unsqueeze(1), 
                        model.fake_imgs[0][0].unsqueeze(0).unsqueeze(0).expand(opt.sample_size, 2, -1, -1))
                    , 1).cpu().detach()*.5+.5
            vistool.update('texts', model.texts[0].unsqueeze(1)*.5+.5)
            vistool.update('styles', model.styles[0].unsqueeze(1)*.5+.5)
            vistool.update('real_img', model.real_img[0]*.5+.5)
            vistool.update('fake_img', model.fake_imgs[0][0]*.5+.5)
            vistool.update('texts_cmp',_texts_cmp)
            vistool.update('styles_cmp',_styles_cmp)
            vistool.update('history', model.real_img[0]*.5+.5)
            vistool.update('history', model.fake_imgs[0][0]*.5+.5)
            vistool.sync()
            if opt.save_model and idx%opt.save_freq == 0:
                model.save_networks('latest')
                model.save_networks('checkpoint_'+str(idx))
                pass
