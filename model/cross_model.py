import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import create_im2im, create_im2vec, create_mixer, init_net,GANLoss
from utils.image_pool import ImagePool
import visdom
vis = visdom.Visdom(env='model')
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

class CrossModel(BaseModel):
    def __init__(self):
        super(CrossModel, self).__init__()
        self.model_names = 'cross_model'

    def initialize(self, opt):
        super(CrossModel, self).initialize(opt)
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

        #A trick to prevent mode collapse
        img = torch.cat((fake_all, real_all, texts, styles),1).detach()
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
        self.loss_G = self.criterionGAN(pred_fake, True) #Gan loss
        self.loss_GSE = self.loss_G
        if hasattr(self.netG, 'score'):
            pred_result = torch.softmax(pred_fake,1)
            self.loss_S = (pred_result-self.netG.score).abs().mean() #Selector loss
            self.loss_GSE += self.loss_S
            vis.bar(torch.stack((pred_result[0], self.netG.score[0]), 1).cpu().detach().numpy(), win='scores')
        self.loss_E = self.netG.extra_loss # Extra loss
        self.loss_GSE += self.loss_E
        self.loss_GSE.backward()

    def optimize_parameters(self):
        self.forward()
        vis.images(self.fake_imgs[0].unsqueeze(1).cpu().detach().numpy()*.5+.5, win='fake_images')
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


class GModel(nn.Module):
    def __init__(self):
        super(GModel, self).__init__()
        self.fastForward = False

    def initialize(self, opt):
        self.style_channels = opt.style_channels
        self.stylenet = create_im2vec(
                opt.im2vec_model,
                in_channels = 1,
                out_channels = self.style_channels,
                n_blocks = 2
                )
        self.style_dropout = torch.nn.Dropout(0.5)
        self.transnet = create_im2im(
                opt.transform_model,
                in_channels = 1,
                out_channels = 1,
                extra_channels = self.style_channels,
                n_blocks = 12
                )
        self.guessnet = create_im2vec(
                opt.im2vec_model,
                in_channels = 4,
                out_channels = 1,
                n_blocks = 1
                )

    def genFont(self, texts, styles_vec, style_count):
        '''
        Generate font using the given style vector
        '''
        texts = texts.unsqueeze(0) #1, tot, W, H
        bs, tot, W, H = texts.shape

        styles = torch.tensor(np.random.randn(bs, tot, W, H).astype(np.float32)).cuda() #I don't know styles, 

        ss =  styles_vec.unsqueeze(0) #1, sc
        sss = ss.view(bs, 1, self.style_channels).expand(-1,tot,-1).contiguous().view(bs*tot, self.style_channels)

        data = texts.view(bs*tot,1,W,H)
        data = self.transnet(data, sss).view(bs, tot, W, H)
        basic_preds = data

        data = data.view(bs*tot, 1, W, H)
        styles_ = styles.view(bs*tot, 1, W, H)
        texts_ = texts.view(bs*tot, 1, W, H)
        styles_rd = shuffle_channels(styles).view(bs*tot, 1, W, H)
        texts_rd = shuffle_channels(texts).view(bs*tot, 1, W, H)
        data = torch.cat([data, texts_, styles_rd, texts_rd], 1)
        score = self.guessnet(data).view(bs, tot)
        score = torch.softmax(score, 1)
        rank = torch.sort(score, 1, descending=True)[1]
        self.best_preds = torch.gather(basic_preds, 1, rank.view(bs, tot, 1, 1).expand(bs, tot, W, H))
        self.score = torch.gather(score, 1, rank)
        return self.best_preds

    def forward(self, texts, styles):
        bs, tot, W, H = texts.shape
        styles_v = self.stylenet(styles.view(bs*tot, 1, W, H)).view(bs, tot, self.style_channels).mean(1)
        styles_v = self.style_dropout(styles_v)

        styles_v_ = styles_v.view(bs, 1, self.style_channels).expand(-1,tot,-1).contiguous().view(bs*tot, self.style_channels)
        data = texts.view(bs*tot,1,W,H)
        data = self.transnet(data, styles_v_).view(bs, tot, W, H)

        basic_preds = data
        mean_pred = basic_preds.mean(1).unsqueeze(1)
        if self.fastForward:
            self.extra_loss = (data-mean_pred).abs().mean()
            t = (basic_preds-mean_pred)[0].unsqueeze(1)
            t = ((t - t.min())/(1e-8+t.max()-t.min())).cpu().detach()
            vis.images(t, win='diff_with_the_average')

        self.basic_preds = basic_preds
        if self.fastForward:
            return data

        data = data.view(bs*tot, 1, W, H)
        texts_ = texts.view(bs*tot, 1, W, H)
        styles_ = styles.view(bs*tot, 1, W, H)
        styles_rd = shuffle_channels(styles).view(bs*tot, 1, W, H)
        texts_rd = shuffle_channels(texts).view(bs*tot, 1, W, H)
        data = torch.cat([data, texts_, styles_rd, texts_rd], 1)
        score = self.guessnet(data).view(bs, tot)
        score = torch.softmax(score, 1)
        rank = torch.sort(score, 1, descending=True)[1]
        self.best_preds = torch.gather(self.basic_preds, 1, rank.view(bs, tot, 1, 1).expand(bs, tot, W, H))
        self.score = torch.gather(score, 1, rank)

        mean_pred = (self.best_preds * self.score.detach().unsqueeze(-1).unsqueeze(-1).expand(-1,-1,W,H)).sum(1).unsqueeze(1)
        self.extra_loss = (basic_preds-mean_pred).abs().mean()
        t = (basic_preds-mean_pred)[0].unsqueeze(1)
        t = ((t - t.min())/(1e-8+t.max()-t.min())).cpu().detach()
        vis.images(t, win='diff_with_the_average')
        return self.best_preds

class DModel(nn.Module):
    def __init__(self):
        super(DModel, self).__init__()

    def initialize(self, opt):
        self.im2vec = create_im2vec(
                opt.im2vec_model,
                in_channels = 3,
                out_channels = 1,
                n_blocks = 1
                )

    def forward(self, targets, texts, styles):
        bs, tot, W, H = texts.shape
        texts = shuffle_channels(texts)
        _, tot_t, W, H = targets.shape
        targets_ = targets.view(bs*tot_t,1,W,H).expand(bs*tot_t,tot,W,H)
        texts_ = texts.view(bs, 1, tot, W, H).expand(bs, tot_t, tot, W, H).contiguous().view(bs*tot_t, tot, W, H)
        styles_ = styles.view(bs, 1, tot, W, H).expand(bs, tot_t, tot, W, H).contiguous().view(bs*tot_t, tot, W, H)
        data = torch.stack((targets_, texts_, styles_), 2).view(bs*tot_t*tot,3,W,H)
        score = self.im2vec(data).view(bs, tot_t, tot).mean(2)*.5+.5

        if tot_t > 1:
            rank = torch.sort(score, 1, descending=True)[1]
            self.dis_preds = torch.gather(targets, 1, rank.view(bs, tot_t, 1, 1).expand(bs, tot_t, W, H))
            self.dis_preds[:,:,0,:] = 0 #Draw a line above the character
            vis.images(self.dis_preds[0].unsqueeze(1).cpu().detach().numpy()*.5+.5, win='dmodel_sorted')
        return score
