import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import create_transformer, create_encoder, init_net

class CrossModel(nn.Module):
    def __init__(self):
        super(CrossModel, self).__init__()

    def initialize(self, opt):
        self.netG = GModel()
        self.netD = DModel()
        self.netG.initialize(opt)
        self.netD.initialize(opt)

        self.criterionGAN = GANLoss(opt.use_lsgan)
        self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG.parameters(), self.netS_D.parameters()),
                lr=opt.learn_rate, betas=(.5, 0.999))
        self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD.parameters(), self.netS_D.parameters()),
                lr=opt.learn_rate, betas=(.5, 0.999))

        init_net(self)

    def set_input(self, texts, styles, target):
        self.texts = texts
        self.styles = styles
        self.real_img = target

    def forward(self):
        self.fake_img = self.netG(texts, styles)

    def backward_D(self):
        fake_all = self.fake_img
        real_all = self.real_img

        pred_fake = self.netD(fake_all.detach(), self.texts, self.styles)
        pred_real = self.netD(real_all.detach(), self.texts, self.styles)

        self.loss_fake = self.criterionGAN(pred_fake, False)
        self.loss_real = self.criterionGAN(pred_real, True)
        self.loss_D = (self.loss_fake + self.loss_real) * .5
        self.loss_D.backward()

    def backward_G(self):
        fake_all = self.fake_img
        pred_fake = self.netD(fake_all, self.texts, self.styles)
        self.loss_G = self.criterionGAN(pred_fake, True)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
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

class GModel(nn.Module):
    def __init__(self):
        super(GModel, self).__init__()
        self.trans = create_transformer(
                opt.transfrom_model,
                in_channels = 2,
                out_channels = 1,
                extra_channels = 0,
                n_blocks = 4
                )

    def initialize(self, opt):
        pass

    def forward(self, texts, styles):
        bs, tot, W, H = texts.shape
        #bs,tot,tot,W*H
        #bs,tot,tot,W*H
        texts = text.view(bs,tot,1,W*H).expand(-1,-1,tot,-1)
        styles = styles.view(bs,tot,W*H,1).expand(-1,-1,-1,tot)
        styles = torch.transpose(styles,-1,-2)
        ts = torch.stack((texts, styles), 3)
        ts = ts.view(bs*tot*tot,1,W,H)
        ts = self.transnet(ts, None).view(bs, tot*tot, W, H).sum(1)
        return ts

class DModel(nn.Module):
    def __init__(self):
        super(DModel, self).__init__()
        self.encoder = create_encoder(
                opt.transfrom_model,
                in_channels = 3,
                out_channels = 1
                )

    def initialize(self, opt):
        pass

    def forward(self, texts, styles, target):
        bs, tot, W, H = texts.shape
        #bs,tot,tot,W*H
        #bs,tot,tot,W*H
        texts = text.view(bs,tot,1,W*H).expand(-1,-1,tot,-1)
        styles = styles.view(bs,tot,W*H,1).expand(-1,-1,-1,tot)
        styles = torch.transpose(styles,-1,-2)
        target = target.view(bs,1,W*H).expand(bs,tot*tot,W*H)
        ts = torch.stack((targets, texts, styles), 3)
        ts = ts.view(bs*tot*tot,1,W,H)
        ts = self.encoder(ts, None).view(bs, tot*tot, W, H).sum(1)
        return ts
