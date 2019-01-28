import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import create_im2im, create_im2vec, create_mixer, init_net,GANLoss, shuffle_channels
from utils.image_pool import ImagePool
import random
import numpy as np
from .base_model import BaseModel
import logging
import vistool
from criterions import find_criterion_using_name
logger = logging.getLogger(__name__)

class CrossModel(BaseModel):
    def __init__(self):
        super(CrossModel, self).__init__()
        self.model_names = 'cross_model'

    @staticmethod
    def modify_commandline_options(parser, is_train = True):
        if is_train:
            parser.add_argument('--style_dropout', type=float, default=.5, help='dropout ratio of style feature vector')
            parser.add_argument('--style_channels', type=int, default=32, help='size of style channels')
            parser.add_argument('--pool_size', type=int, default=150, help='size of image pool, which is used to prevent model collapse')
            parser.add_argument('--lambda_E', type=float, default=0.0, help='lambda of extra loss')
            parser.add_argument('--fast_forward', type=bool, default=False, help='do not train the selector')
            parser.add_argument('--opt_betas1', type=float, default=.5)
            parser.add_argument('--opt_betas2', type=float, default=.999)
            parser.add_argument('--g_model_transnet', type=str, default='resnet')
            parser.add_argument('--g_model_transnet_n_blocks', type=int, default=8)
            parser.add_argument('--d_model_n_blocks', type=int, default=1)
            parser.add_argument('--d_model_use_dropout', type=bool, default=False)
            parser.add_argument('--selector_criterion_method', type=str, default='l1')
        return parser

    def init_vistool(self, opt):
        self.vistool = vistool.VisTool(env=opt.name+'_model')
        self.vistool.register_data('fake_imgs', 'images')
        self.vistool.register_data('styles', 'images')
        self.vistool.register_data('texts', 'images')
        self.vistool.register_data('diff_with_average', 'images')
        self.vistool.register_data('dmodel_sorted', 'images')
        self.vistool.register_data('scores', 'array')
        self.vistool.register_data('dis_preds_L1_loss', 'scalar_ma')
        self.vistool.register_data('sel_preds_L1_loss', 'scalar_ma')
        self.vistool.register_data('rad_preds_L1_loss', 'scalar_ma')
        self.vistool.register_window('fake_imgs', 'images', source='fake_imgs')
        if not opt.fast_forward:
            self.vistool.register_window('scores', 'bar', source='scores')
        self.vistool.register_window('preds_L1_loss', 'lines', sources=['dis_preds_L1_loss', 'sel_preds_L1_loss', 'rad_preds_L1_loss'])

    def initialize(self, opt):
        super(CrossModel, self).initialize(opt)
        self.fastForward = opt.fast_forward
        self.netG = GModel()
        self.netD = DModel()
        self.netG.initialize(opt)
        self.netD.initialize(opt)
        self.criterionGAN = GANLoss(False)
        self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=opt.learn_rate, betas=(opt.opt_betas1, opt.opt_betas2))
        self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=opt.learn_rate, betas=(opt.opt_betas1, opt.opt_betas2))
        self.pool = ImagePool(opt.pool_size)
        self.lambda_E = opt.lambda_E
        self.criterionSelector = find_criterion_using_name(opt.selector_criterion_method)()

        init_net(self)
        path = opt.checkpoints_dir+'/'+self.model_names+'.txt'
        with open(path,'w') as f:
            f.write(str(self))
        logger.info("Model Structure has been written into %s"%path)

        self.init_vistool(opt)

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
        if not self.fastForward:
            #pred_result = torch.softmax(pred_fake,1)
            pred_result = pred_fake.detach()
            self.loss_S = (pred_result-self.netG.score).abs().mean() #Selector loss
            self.loss_GSE += self.loss_S
            self.vistool.update('scores', torch.stack((pred_result[0], self.netG.score[0]), 1))
        self.loss_E = self.netG.extra_loss # Extra loss
        self.loss_GSE += self.loss_E * self.lambda_E
        self.loss_GSE.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        if self.optm_d:
            self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        if self.optm_g:
            self.optimizer_G.step()

        self.vistool.update('dmodel_sorted', self.netD.dis_preds[0]*.5+.5)
        self.vistool.update('diff_with_average', self.netG.diff_with_average)
        self.vistool.update('dis_preds_L1_loss', self.criterionSelector(self.netD.dis_preds[:,0,:,:], self.real_img[:,0,:,:]).mean())
        self.vistool.update('sel_preds_L1_loss', self.criterionSelector(self.netG.best_preds[:,0,:,:], self.real_img[:,0,:,:]).mean())
        idx = random.randint(0, self.netG.best_preds.size(1)-1)
        self.vistool.update('rad_preds_L1_loss', self.criterionSelector(self.netG.best_preds[:,idx,:,:], self.real_img[:,0,:,:]).mean())
        self.vistool.update('fake_imgs', self.fake_imgs[0]*.5+.5)
        self.vistool.update('styles', self.styles[0]*.5+.5)
        self.vistool.update('texts', self.texts[0]*.5+.5)
        self.vistool.sync()


class GModel(nn.Module):
    def __init__(self):
        super(GModel, self).__init__()

    def initialize(self, opt):
        self.fastForward = opt.fast_forward
        self.style_channels = opt.style_channels
        self.stylenet = create_im2vec(
                'resnet',
                in_channels = 1,
                out_channels = self.style_channels,
                n_blocks = 4
                )
        self.style_dropout = torch.nn.Dropout(opt.style_dropout)
        self.transnet = create_im2im(
                opt.g_model_transnet,
                in_channels = 1,
                out_channels = 1,
                extra_channels = self.style_channels,
                n_blocks = opt.g_model_transnet_n_blocks
                )
        self.guessnet = create_im2vec(
                'resnet',
                in_channels = 4,
                out_channels = 1,
                n_blocks = 1
                )

    def forward_styles(self, styles):
        bs, tot, W, H = styles.shape
        styles_v = self.stylenet(styles.view(bs*tot, 1, W, H)).view(bs, tot, self.style_channels).mean(1)
        styles_v = self.style_dropout(styles_v)
        self.styles_v = styles_v.view(bs, 1, self.style_channels).expand(-1,tot,-1).contiguous().view(bs*tot, 1, self.style_channels)

    def forward_texts(self, texts, styles):
        bs, tot, W, H = texts.shape
        data = texts.view(bs*tot,1,W,H)
        data = self.transnet(data, self.styles_v).view(bs, tot, W, H)

        basic_preds = data
        self.basic_preds = basic_preds
        mean_pred = basic_preds.mean(1).unsqueeze(1)
        if self.fastForward:
            self.extra_loss = (data-mean_pred).abs().mean()
            t = (basic_preds-mean_pred)[0].unsqueeze(1)
            t = ((t - t.min())/(1e-8+t.max()-t.min())).cpu().detach()
            self.diff_with_average = t
            self.score = None
            self.best_preds = self.basic_preds
            return ;

        data = data.view(bs*tot, 1, W, H)
        texts_ = texts.view(bs*tot, 1, W, H)
        styles_ = styles.view(bs*tot, 1, W, H)
        styles_rd = shuffle_channels(styles, 1).view(bs*tot, 1, W, H)
        texts_rd = shuffle_channels(texts, 1).view(bs*tot, 1, W, H)
        data = torch.cat([data, texts_, styles_rd, texts_rd], 1)
        score = self.guessnet(data).view(bs, tot) *.5+.5
        #score = torch.softmax(score, 1)
        rank = torch.sort(score, 1, descending=True)[1]
        self.best_preds = torch.gather(self.basic_preds, 1, rank.view(bs, tot, 1, 1).expand(bs, tot, W, H))
        self.score = torch.gather(score, 1, rank)

        mean_pred = (self.best_preds * self.score.detach().unsqueeze(-1).unsqueeze(-1).expand(-1,-1,W,H)).sum(1).unsqueeze(1)
        self.extra_loss = (basic_preds-mean_pred).abs().mean()
        t = (basic_preds-mean_pred)[0].unsqueeze(1)
        t = ((t - t.min())/(1e-8+t.max()-t.min())).cpu().detach()
        self.diff_with_average = t

    def forward(self, texts, styles):
        self.forward_styles(styles)
        self.forward_texts(texts, styles)
        return self.best_preds

    def genFont(self, texts, styles, style_v):
        self.forward_styles(styles)
        self.style_v = style_v
        self.forward_texts(texts, styles)
        return self.best_preds

class DModel(nn.Module):
    def __init__(self):
        super(DModel, self).__init__()

    def initialize(self, opt):
        self.im2vec = create_im2vec(
                'resnet',
                in_channels = 3,
                out_channels = 1,
                n_blocks = opt.d_model_n_blocks,
                use_dropout = opt.d_model_use_dropout
                )

    def forward(self, targets, texts, styles):
        bs, tot, W, H = texts.shape
        texts = shuffle_channels(texts,2)
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
        return score
