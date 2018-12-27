import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from .networks import create_im2im, create_im2vec, init_net,GANLoss

class CrossModel(nn.Module):
    def __init__(self):
        super(CrossModel, self).__init__()
        self.model_names = 'cross_model'

    def initialize(self, opt):
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.netG = GModel()
        self.netD = DModel()
        self.netG.initialize(opt)
        self.netD.initialize(opt)

        self.criterionGAN = GANLoss(opt.use_lsgan)
        self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=opt.learn_rate, betas=(.5, 0.999))
        self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=opt.learn_rate, betas=(.5, 0.999))

        init_net(self)

    def set_input(self, texts, styles, target):
        self.texts = texts
        self.styles = styles
        self.real_img = target

    def forward(self):
        self.fake_img = self.netG(self.texts, self.styles)

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

    def save_networks(self, epoch):
        name = self.model_names
        save_filename = '%s_net_%s.pth' % (epoch, name)
        save_path = os.path.join(self.save_dir, save_filename)
        model = self
        torch.save(model.state_dict(), save_path)

    def load_networks(self, epoch):
        name = self.model_names
        load_filename = '%s_net_%s.pth' % (epoch, name)
        load_path = os.path.join(self.save_dir, load_filename)
        self.load_state_dict(torch.load(load_path))

class GModel(nn.Module):
    def __init__(self):
        super(GModel, self).__init__()

    def initialize(self, opt):
        self.transnet = create_im2im(
                opt.transform_model,
                in_channels = 2,
                out_channels = 1,
                extra_channels = 0,
                n_blocks = 8
                )
        self.transnet_2 = create_im2im(
                opt.transform_model,
                in_channels = 30,
                out_channels = 1,
                extra_channels = 0,
                n_blocks = 6
                )
        pass

    def forward(self, texts, styles):
        bs, tot, W, H = texts.shape
        ts = torch.stack((texts, styles), 2).view(bs*tot,2,W,H)
        ts = self.transnet(ts, None).view(bs, tot, W, H)
        ts = torch.cat([ts, texts, styles], 1)
        ts = self.transnet_2(ts, None).view(bs, W, H)
        return ts
        '''
        bs, tot, W, H = texts.shape
        texts = texts.view(bs,tot,1,W*H).expand(-1,-1,tot,-1)
        styles = styles.view(bs,tot,W*H,1).expand(-1,-1,-1,tot)
        styles = torch.transpose(styles,-1,-2)
        texts = texts.contiguous().view(bs,tot*tot,W*H)
        styles = styles.contiguous().view(bs,tot*tot,W*H)
        ts = torch.stack((texts, styles), 2)
        ts = ts.view(bs*tot*tot,2,W,H)
        ts = self.transnet(ts, None).view(bs, tot*tot, W, H)
        ts = self.transnet_2(ts, None).view(bs, W, H)
        return ts
        '''

class DModel(nn.Module):
    def __init__(self):
        super(DModel, self).__init__()

    def initialize(self, opt):
        self.im2vec = create_im2vec(
                opt.im2vec_model,
                in_channels = 3,
                out_channels = 1,
                n_blocks = 4
                )

    def forward(self, target, texts, styles):
        bs, tot, W, H = texts.shape
        target = target.view(bs,1,W,H).expand(bs,tot,W,H)
        ts = torch.stack((target, texts, styles), 2).view(bs*tot,3,W,H)
        ts = self.im2vec(ts).view(bs, tot, 1).mean(2)*.5+.5
        return ts
    '''
        bs, tot, W, H = texts.shape
        ddd = 1
        texts = texts.view(bs,tot,1,W*H).expand(-1,-1,ddd,-1)
        styles = styles.view(bs,tot,W*H,1).expand(-1,-1,-1,ddd)
        styles = torch.transpose(styles,-1,-2)
        texts = texts.contiguous().view(bs,tot*ddd,W*H)
        styles = styles.contiguous().view(bs,tot*ddd,W*H)
        target = target.view(bs,1,W*H).expand(bs,tot*ddd,W*H)
        ts = torch.stack((target, texts, styles), 3)
        ts = ts.view(bs*tot*ddd,3,W,H)
        ts = self.im2vec(ts).view(bs, tot, ddd, 1).mean(2)*.5+.5
        return ts
        '''
