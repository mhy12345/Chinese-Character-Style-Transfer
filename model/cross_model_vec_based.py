import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import create_im2im, create_im2vec, init_net,GANLoss, create_vec2im
import os
from utils.image_pool import ImagePool

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
        self.pool = ImagePool(40)

        self.criterionGAN = GANLoss(opt.use_lsgan)
        self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=opt.learn_rate
                )
        self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=opt.learn_rate
                )

        init_net(self)
        print(self)

    def set_input(self, texts, styles, target):
        self.texts = texts
        self.styles = styles
        self.real_img = target

    def forward(self):
        self.fake_img = self.netG(self.texts, self.styles)
        self.score = self.netG.score.std()

    def backward_D(self):
        fake_all = self.fake_img
        real_all = self.real_img
        texts = self.texts
        styles = self.styles
        #print(texts.shape,styles.shape,fake_all.shape,real_all.shape)
        '''
        img = torch.cat((torch.stack((fake_all,real_all),1),texts, styles),1)
        img = self.pool.query(img)
        fake_all, real_all, texts, styles = torch.split(img,[1,1,10,10],1)
        fake_all = fake_all.squeeze(1)
        real_all = real_all.squeeze(1)
        '''
        #print(texts.shape,styles.shape,fake_all.shape,real_all.shape)

        pred_fake = self.netD(fake_all.detach(), texts, styles)
        pred_real = self.netD(real_all.detach(), texts, styles)

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
        self.style_channels = opt.style_channels
        self.nfc = 128
        self.transnet = create_im2vec(
                opt.im2vec_model,
                in_channels = 2,
                out_channels = self.style_channels,
                )
        self.vec2vec = nn.Sequential(
                nn.Linear(self.style_channels, self.nfc),
                nn.LeakyReLU(.2),
                nn.Linear(self.nfc, self.nfc),
                nn.LeakyReLU(.2),
                nn.Linear(self.nfc, 1),
                nn.Tanh()
                )
        self.vec2im = create_vec2im(
                opt.vec2im_model,
                in_channels = self.style_channels,
                out_channels = 1
                )
        pass

    def forward(self, texts, styles):
        bs, tot, W, H = texts.shape
        ts = torch.stack((texts, styles), 2).view(bs*tot,2,W,H)
        ts = self.transnet(ts).view(bs, tot, self.style_channels)
        '''
        res = torch.split(ts, 1, 1)[0].view(bs, self.style_channels)
        return self.vec2im(res)
        '''
        options = ts
        vt = self.vec2vec(ts).view(bs, tot)
        self.score = vt
        mask = torch.argmax(vt, 1).unsqueeze(-1).unsqueeze(-1).expand(bs, 1, self.style_channels)
        res = torch.gather(options, 1, mask).squeeze(1)
        return self.vec2im(res)
        '''
        ddd = 3
        bs, tot, W, H = texts.shape
        texts = texts.view(bs,tot,1,W*H).expand(-1,-1,ddd,-1)
        styles = styles.view(bs,tot,W*H,1).expand(-1,-1,-1,ddd)
        styles = torch.transpose(styles,-1,-2)
        texts = texts.contiguous().view(bs,tot*ddd,W,H)
        styles = styles.contiguous().view(bs,tot*ddd,W,H)
        ts = torch.stack((texts, styles), 2)

        ts = ts.view(bs*ddd*tot,2,W,H)
        ts = self.transnet(ts, None).view(bs, tot*ddd, W, H)
        options = ts.view(bs, ddd*tot, W, H)
        print(ts.shape, texts.shape, styles.shape)
        ts = torch.stack((ts, texts, styles), 2).view(bs*tot*ddd,3,W,H)
        vt = self.im2vec(ts).view(bs, tot*ddd)
        mask = torch.argmax(vt, 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(bs, 1, W, H)
        res = torch.gather(options, 1, mask)
        return res.squeeze(1)
        '''

class DModel(nn.Module):
    def __init__(self):
        super(DModel, self).__init__()

    def initialize(self, opt):
        self.im2vec = create_im2vec(
                opt.im2vec_model,
                in_channels = 3,
                out_channels = 1,
                n_blocks = 2
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
