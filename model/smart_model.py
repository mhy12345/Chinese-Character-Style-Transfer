import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools
import visdom
import numpy as np
vis = visdom.Visdom(env='model')

from .networks import create_transformer, create_encoder, init_net,GANLoss

class SmartModel(nn.Module):
    '''
    GAN & Pix2Pix
    '''
    def __init__(self):
        super(SmartModel, self).__init__()

    def initialize(self, opt):
        '''
        Define the model structure.
        '''
        self.netS_D = SModel()
        self.netS_D.initialize(opt)
        self.netG = GModel()
        self.netG.initialize(opt)
        self.netD = DModel()
        self.netD.initialize(opt)

        self.criterionGAN = GANLoss(opt.use_lsgan)
        self.criterionL1 = torch.nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG.parameters(), self.netS_D.parameters()),
                lr=opt.learn_rate, betas=(.5, 0.999))
        self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD.parameters(), self.netS_D.parameters()),
                lr=opt.learn_rate*10, betas=(.5, 0.999))

        init_net(self)

    def set_input(self, imgs_A, imgs_B):
        self.real_A = imgs_A
        self.real_B = imgs_B

    def forward(self):
        self.style_A_D = self.netS_D(self.real_A).mean(1)
        self.style_B_D = self.netS_D(self.real_B).mean(1)
        self.fake_B = self.netG(self.real_A, self.style_B_D)
        self.fake_A = self.netG(self.real_B, self.style_A_D)

    def backward_D(self):
        fake_all_A = torch.cat([self.fake_A, self.real_B], 1)
        fake_all_B = torch.cat([self.fake_B, self.real_A], 1)
        real_all_A = self.real_A
        real_all_B = self.real_B
        pred_fake_A = self.netD(fake_all_A.detach(), self.style_A_D)
        pred_fake_B = self.netD(fake_all_B.detach(), self.style_B_D)
        self.loss_A_fake = self.criterionGAN(pred_fake_A, False)
        self.loss_B_fake = self.criterionGAN(pred_fake_B, False)

        pred_real_A = self.netD(real_all_A.detach(), self.style_A_D)
        pred_real_B = self.netD(real_all_B.detach(), self.style_B_D)
        self.loss_A_real = self.criterionGAN(pred_real_A, True)
        self.loss_B_real = self.criterionGAN(pred_real_B, True)
        self.loss_D = (self.loss_A_fake + self.loss_B_fake + self.loss_A_real + self.loss_B_real) * 0.25
        self.loss_D.backward()


    def backward_G(self):
        fake_all_A = torch.cat([self.fake_A], 1)
        fake_all_B = torch.cat([self.fake_B], 1)
        real_all_A = self.real_A
        real_all_B = self.real_B
        pred_fake_A = self.netD(fake_all_A, self.style_A_D)
        pred_fake_B = self.netD(fake_all_B, self.style_B_D)
        self.loss_A_fake = self.criterionGAN(pred_fake_A, False)
        self.loss_B_fake = self.criterionGAN(pred_fake_B, False)
        self.loss_G_GAN = self.criterionGAN(pred_fake_A, True) + self.criterionGAN(pred_fake_B, True)

        self.loss_G = self.loss_G_GAN # + self.loss_G_L1 
        self.loss_G.backward()

    def optimize_parameters(self):
        choice = random.randint(0,1)
        self.forward()
        self.set_requires_grad(self.netD, True)
        '''
        self.set_requires_grad(self.netS_D, True)
        self.set_requires_grad(self.netG, False)
        self.set_requires_grad(self.netS_G, False)
        '''
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        '''
        self.set_requires_grad(self.netS_D, False)
        self.set_requires_grad(self.netG, True)
        self.set_requires_grad(self.netS_G, True)
        '''
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

class SModel(nn.Module):
    def __init__(self):
        super(SModel, self).__init__()

    def initialize(self,opt):
        self.dnet = create_encoder(
                opt.encoder_model,
                in_channels = 1,
                out_channels = opt.style_channels
                )
        self.style_channels = opt.style_channels

    def forward(self, imgs):
        bs, tot, w, h = imgs.shape
        data = torch.reshape(imgs, (bs*tot, 1, w, h))
        data = self.dnet(data)
        data = torch.reshape(data, (bs, tot, self.style_channels))
        return data

class DModel(nn.Module):
    def __init__(self):
        super(DModel, self).__init__()

    def initialize(self, opt):
        self.style_channels = opt.style_channels
        self.unet = create_transformer(
                opt.transform_model,
                in_channels = 1,
                out_channels = 10,
                extra_channels = opt.style_channels,
                n_blocks = 4
                )
        self.dnet = create_encoder(
                opt.encoder_model,
                in_channels = 10,
                out_channels = opt.style_channels
                )
        self.linear = nn.Linear(self.style_channels,1)

    def forward(self, imgs, style):
        bs, tot, w, h = imgs.shape
        style = style.unsqueeze(1).expand(bs, tot, -1).reshape(bs*tot, self.style_channels)
        data = torch.reshape(imgs, (bs*tot, 1, w, h))
        data = self.unet(data, style)
        data = self.dnet(data)
        data = self.linear(data)
        data = torch.reshape(data, (bs, tot))
        data = F.tanh(data)*.5+.5
        return data

class GModel(nn.Module):
    def __init__(self):
        super(GModel,self).__init__()

    def initialize(self, opt):
        self.style_channels = opt.style_channels
        self.unet = create_transformer(
                opt.transform_model,
                in_channels = 1,
                out_channels = 10,
                n_blocks = 6,
                extra_channels = opt.style_channels
                )

    def forward(self, imgs, style):
        bs, tot, w, h = imgs.shape
        style = style.unsqueeze(1).expand(bs, tot, -1).reshape(bs*tot, self.style_channels)
        data = torch.reshape(imgs, (bs*tot, 1, w, h))
        data = self.unet(data, style).sum(1)
        data = torch.reshape(data, (bs, tot, w, h))
        data = F.tanh(data)
        return data


