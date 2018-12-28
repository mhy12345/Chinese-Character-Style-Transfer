import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from .networks import create_im2im, create_im2vec, init_net,GANLoss
from utils.image_pool import ImagePool
import visdom
vis = visdom.Visdom(env='model')
import random
import numpy as np

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
        self.pool = ImagePool(100)

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

        img = torch.cat((fake_all,real_all,texts, styles),1)
        img = self.pool.query(img)
        fake_all, real_all, texts, styles = torch.split(img,[10,1,10,10],1)
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
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        vis.images(self.fake_imgs[0].unsqueeze(1).cpu().detach().numpy()*.5+.5, win='ts')
        vis.images(self.styles[0].unsqueeze(1).cpu().detach().numpy()*.5+.5, win='styles')
        vis.images(self.texts[0].unsqueeze(1).cpu().detach().numpy()*.5+.5, win='texts')
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
        self.fastForward = True

    def initialize(self, opt):
        self.style_channels = opt.style_channels
        self.stylenet = create_im2vec(
                opt.im2vec_model,
                in_channels = 10,
                out_channels = self.style_channels,
                n_blocks = 6
                )
        self.transnet = create_im2im(
                opt.transform_model,
                in_channels = 1,
                out_channels = 1,
                extra_channels = self.style_channels,
                n_blocks = 8
                )
        self.transnet_2 = create_im2im(
                opt.transform_model,
                in_channels = 20,
                out_channels = 1,
                extra_channels = self.style_channels,
                n_blocks = 6
                )
        pass

    def forward(self, texts, styles):
        bs, tot, W, H = texts.shape
        ss = self.stylenet(styles)
        sss = ss.view(bs, 1, self.style_channels).expand(-1,tot,-1).contiguous().view(bs*tot, self.style_channels)
        #ts = torch.stack((texts, styles), 2).view(bs*tot,2,W,H)
        ts = texts.view(bs*tot,1,W,H)
        ts = self.transnet(ts, sss).view(bs, tot, W, H)
        if self.fastForward:
            return ts
            #return torch.split(ts, 1, 1)[random.randint(0,9)]
        '''
        mask = [1 for i in range(tot)]
        mask[random.randint(0,tot-1)] = 0
        mask = torch.tensor(np.array(mask).astype(np.float32)).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(bs, tot, W, H)

        mask2 = [random.randint(0,1) for i in range(tot)]
        if (random.randint(0,1) == 0):
            mask2[0] = 0
            mask2[5] = 0
        mask2 = torch.tensor(np.array(mask2).astype(np.float32)).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(bs, tot, W, H)
        ts = ts  * mask
        texts = texts * mask2
        styles = styles * mask2
        '''

        ts = torch.cat([ts, texts], 1)
        ts = self.transnet_2(ts, ss).view(bs, W, H)
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

    def forward(self, targets, texts, styles):
        bs, tot, W, H = texts.shape
        _, tot_t, W, H = targets.shape
        targets = targets.view(bs*tot_t,1,W,H).expand(bs*tot_t,tot,W,H)
        texts = texts.view(bs, 1, tot, W, H).expand(bs, tot_t, tot, W, H).contiguous().view(bs*tot_t, tot, W, H)
        styles = styles.view(bs, 1, tot, W, H).expand(bs, tot_t, tot, W, H).contiguous().view(bs*tot_t, tot, W, H)
        ts = torch.stack((targets, texts, styles), 2).view(bs*tot_t*tot,3,W,H)
        ts = self.im2vec(ts).view(bs, tot_t, tot).mean(2)*.5+.5
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
