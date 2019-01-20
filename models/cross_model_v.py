import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .networks import create_im2im, create_im2vec, create_vec2im, create_mixer, init_net,GANLoss
from utils.image_pool import ImagePool
import visdom
vis = visdom.Visdom(env='model')
debug = visdom.Visdom(env='debug')
import random
import numpy as np
from .base_model import BaseModel

def shuffle_channels(data):
    '''
    Shuffle the images. (dim=1)
    data : (batch_size, total, W, H) 
    '''
    data = list(torch.split(data, 1, 1))
    random.shuffle(data)
    data = torch.cat(data ,1) #Shuffle the texts
    return data

class CrossModelV(BaseModel):
    def __init__(self):
        super(CrossModelV, self).__init__()
        self.model_names = 'cross_model_v'

    def initialize(self, opt):
        super(CrossModelV, self).initialize(opt)
        self.netG = GModel()
        self.netD = DModel()
        self.netG.initialize(opt)
        self.netD.initialize(opt)

        self.criterionGAN = GANLoss(opt.use_lsgan)
        self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=opt.learn_rate, 
                #betas=(.5, 0.9)
                betas=(.5, 0.999)
                )
        self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=opt.learn_rate, 
                betas=(.5, 0.999)
                )
        self.pool = ImagePool(160)

        init_net(self)
        print(self)

    def set_input(self, texts, styles, target):
        self.texts = texts
        self.styles = styles
        self.real_img = target.unsqueeze(1)

    def forward(self):
        self.fake_imgs = self.netG(self.texts, self.styles)

    def backward_D(self):
        fake_all = self.fake_imgs
        real_all = self.real_img
        texts = self.texts
        styles = self.styles

        img = torch.cat((fake_all,real_all,texts, styles),1).detach()
        img = self.pool.query(img)
        tot = (img.size(1)-1)//3
        fake_all, real_all, texts, styles = torch.split(img,[tot,1,tot,tot],1)
        fake_all = fake_all.contiguous()
        real_all = real_all.contiguous()

        pred_fake = self.netD(fake_all.detach(), texts, styles)
        pred_real = self.netD(real_all.detach(), texts, styles)

        self.loss_fake = self.criterionGAN(pred_fake, False)
        self.loss_real = self.criterionGAN(pred_real, True)
        self.loss_D = (self.loss_fake + self.loss_real) * .5
        self.loss_D.backward()

    def backward_G(self):
        fake_all = self.fake_imgs
        pred_fake = self.netD(fake_all, self.texts, self.styles)
        self.loss_G = self.criterionGAN(pred_fake, True)
        self.loss_GSE = self.loss_G
        if hasattr(self.netG, 'score'):
            pred_basic = torch.softmax(pred_fake,1)
            self.loss_S = (pred_basic-self.netG.score).abs().mean()
            self.loss_GSE += self.loss_S
            vis.bar(torch.stack((pred_basic[0], self.netG.score[0]), 1).cpu().detach().numpy(), win='scores')
        self.loss_E = self.netG.extra_loss
        #self.loss_GSE += self.netG.extra_loss*1
        self.loss_GSE.backward()

    def optimize_parameters(self):
        self.forward()
        vis.images(self.fake_imgs[0].unsqueeze(1).cpu().detach().numpy()*.5+.5, win='data')
        vis.images(self.styles[0].unsqueeze(1).cpu().detach().numpy()*.5+.5, win='styles')
        vis.images(self.texts[0].unsqueeze(1).cpu().detach().numpy()*.5+.5, win='texts')
        vis.images(self.netG.vec_pred[0].unsqueeze(1).cpu().detach().numpy()*.5+.5, win='pred')
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


class GModel(nn.Module):
    def __init__(self):
        super(GModel, self).__init__()
        self.fastForward = True

    def initialize(self, opt):
        self.style_channels = opt.style_channels
        self.text_channels = opt.text_channels
        '''
        self.stylenet = create_im2vec(
                opt.im2vec_model,
                in_channels = 1,
                out_channels = self.style_channels,
                n_blocks = 4
                )
        '''
        self.stylenet_2 = create_im2vec(
                opt.im2vec_model,
                in_channels = 1,
                out_channels = self.style_channels,
                n_blocks = 4
                )
        self.textnet_2 = create_im2vec(
                opt.im2vec_model,
                in_channels = 1,
                out_channels = self.text_channels,
                n_blocks = 8
                )
        self.style_dropout = torch.nn.Dropout(0.5)
        self.gennet = create_vec2im(
                opt.vec2im_model,
                in_channels = self.style_channels+self.text_channels,
                out_channels = 1,
                n_blocks = 12
                )
        '''
        self.transnet = create_im2im(
                opt.transform_model,
                in_channels = 1,
                out_channels = 1,
                extra_channels = self.style_channels,
                n_blocks = 12
                )
        '''
        self.guessnet = create_im2vec(
                opt.im2vec_model,
                in_channels = 4,
                out_channels = 1,
                n_blocks = 1
                )

    def forward(self, texts, styles):
        bs, tot, W, H = texts.shape
        texts_ = texts.view(bs*tot, 1, W, H)
        styles_ = styles.view(bs*tot, 1, W, H)

        #styles_v_1 = self.stylenet(styles.view(bs*tot, 1, W, H)).view(bs, tot, self.style_channels).mean(1)
        styles_v_2 = self.stylenet_2(styles.view(bs*tot, 1, W, H)).view(bs, tot, self.style_channels).mean(1)
        #styles_v_1 = self.style_dropout(styles_v_1)
        #styles_v_1_ = styles_v_1.view(bs, 1, self.style_channels).expand(-1,tot,-1).contiguous().view(bs*tot, self.style_channels)
        texts_vs_2 = self.textnet_2(texts.view(bs*tot, 1, W, H)).view(bs, tot, self.text_channels)

        styles_vs_2 = styles_v_2.unsqueeze(1).expand(-1, tot, -1)

        texts_and_styles_vs = torch.cat([styles_vs_2, texts_vs_2], 2).view(bs*tot, self.style_channels+self.text_channels)

        data_v = self.gennet(texts_and_styles_vs).view(bs, tot, W, H)

        #data_i = self.transnet(texts_, styles_v_1_).view(bs, tot, W, H)
        #debug.images(data_i[0].unsqueeze(1).cpu().detach()*.5+.5, win='data_i')
        debug.images(data_v[0].unsqueeze(1).cpu().detach()*.5+.5, win='data_v')
        #debug.images(((data_i-data_v)/2)[0].unsqueeze(1).cpu().detach()*.5+.5, win='data__')
        t = ((data_v-data_v.mean(1).unsqueeze(1)))[0].unsqueeze(1)
        t = ((t - t.min())/(1e-8+t.max()-t.min())).cpu().detach()
        debug.images(t, win='data_a')
        #self.extra_loss = (data_v - data_i).abs().mean()
        self.extra_loss = None
        self.vec_pred = data_v
        data = data_v

        if self.fastForward:
            return data
        self.basic_preds = data
        data = data.view(bs*tot, 1, W, H)
        styles_rd = shuffle_channels(styles).view(bs*tot, 1, W, H)
        texts_rd = shuffle_channels(texts).view(bs*tot, 1, W, H)
        data = torch.cat([data, texts_, styles_rd, texts_rd], 1)
        score = self.guessnet(data).view(bs, tot)
        score = torch.softmax(score, 1)
        rank = torch.sort(score, 1, descending=True)[1]
        self.best_preds = torch.gather(self.basic_preds, 1, rank.view(bs, tot, 1, 1).expand(bs, tot, W, H))
        self.score = torch.gather(score, 1, rank)
        return self.best_preds
        '''
        bs, tot, W, H = texts.shape
        texts = texts.view(bs,tot,1,W*H).expand(-1,-1,tot,-1)
        styles = styles.view(bs,tot,W*H,1).expand(-1,-1,-1,tot)
        styles = torch.transpose(styles,-1,-2)
        texts = texts.contiguous().view(bs,tot*tot,W*H)
        styles = styles.contiguous().view(bs,tot*tot,W*H)
        data = torch.stack((texts, styles), 2)
        data = data.view(bs*tot*tot,2,W,H)
        data = self.transnet(data, None).view(bs, tot*tot, W, H)
        data = self.transnet_2(data, None).view(bs, W, H)
        return data
        '''

class DModel(nn.Module):
    def __init__(self):
        super(DModel, self).__init__()

    def initialize(self, opt):
        self.im2vec = create_im2vec(
                opt.im2vec_model,
                in_channels = 3,
                out_channels = 1,
                n_blocks = 3
                )

    def forward(self, targets, texts, styles):
        bs, tot, W, H = texts.shape
        texts = shuffle_channels(texts)
        styles = shuffle_channels(styles)
        _, tot_t, W, H = targets.shape
        targets = targets.view(bs*tot_t,1,W,H).expand(bs*tot_t,tot,W,H)
        texts = texts.view(bs, 1, tot, W, H).expand(bs, tot_t, tot, W, H).contiguous().view(bs*tot_t, tot, W, H)
        styles = styles.view(bs, 1, tot, W, H).expand(bs, tot_t, tot, W, H).contiguous().view(bs*tot_t, tot, W, H)
        data = torch.stack((targets, texts, styles), 2).view(bs*tot_t*tot,3,W,H)
        data = self.im2vec(data).view(bs, tot_t, tot).mean(2)*.5+.5
        return data
    '''
        bs, tot, W, H = texts.shape
        ddd = 1
        texts = texts.view(bs,tot,1,W*H).expand(-1,-1,ddd,-1)
        styles = styles.view(bs,tot,W*H,1).expand(-1,-1,-1,ddd)
        styles = torch.transpose(styles,-1,-2)
        texts = texts.contiguous().view(bs,tot*ddd,W*H)
        styles = styles.contiguous().view(bs,tot*ddd,W*H)
        target = target.view(bs,1,W*H).expand(bs,tot*ddd,W*H)
        data = torch.stack((target, texts, styles), 3)
        data = data.view(bs*tot*ddd,3,W,H)
        data = self.im2vec(data).view(bs, tot, ddd, 1).mean(2)*.5+.5
        return data
        '''
