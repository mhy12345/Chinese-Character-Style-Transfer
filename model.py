import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import random
import itertools
import functools

import visdom
import numpy as np
vis = visdom.Visdom(env='model')

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
        self.netS_G = SModel()
        self.netS_D = SModel()
        self.netG = GModel()
        self.netD = DModel()
        self.fake_img_pool = ImagePool(opt.pool_size)

        self.criterionGAN = GANLoss(opt.use_lsgan)
        self.criterionL1 = torch.nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG.parameters(), self.netS_G.parameters()),
                                            lr=opt.learn_rate, betas=(.5, 0.999))
        self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD.parameters(), self.netS_D.parameters()),
                                            lr=opt.learn_rate, betas=(.5, 0.999))

        init_net(self)

    def set_input(self, imgs_A, imgs_B):
        self.real_A = imgs_A
        self.real_B = imgs_B

    def forward(self):
        self.style_A_G = self.netS_G(self.real_A).mean(1)
        self.style_B_G = self.netS_G(self.real_B).mean(1)
        self.style_A_D = self.netS_D(self.real_A).mean(1)
        self.style_B_D = self.netS_D(self.real_B).mean(1)
        self.fake_B = self.netG(self.real_A, self.style_B_G)
        self.fake_A = self.netG(self.real_B, self.style_A_G)

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
        fake_all_A = torch.cat([self.fake_A, self.real_B], 1)
        fake_all_B = torch.cat([self.fake_B, self.real_A], 1)
        real_all_A = self.real_A
        real_all_B = self.real_B
        pred_fake_A = self.netD(fake_all_A, self.style_A_G)
        pred_fake_B = self.netD(fake_all_B, self.style_B_G)
        self.loss_A_fake = self.criterionGAN(pred_fake_A, False)
        self.loss_B_fake = self.criterionGAN(pred_fake_B, False)
        self.loss_G_GAN = self.criterionGAN(pred_fake_A, True) + self.criterionGAN(pred_fake_B, True)

        #self.loss_G_L1 = self.criterionL1(self.fake_img, self.real_img) * .05

        self.loss_G = self.loss_G_GAN # + self.loss_G_L1 
        self.loss_G.backward()

    def optimize_parameters(self):
        choice = random.randint(0,1)
        self.set_requires_grad(self.netD, True)
        self.set_requires_grad(self.netS_D, True)
        self.set_requires_grad(self.netG, False)
        self.set_requires_grad(self.netS_G, False)
        self.forward()
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netS_D, False)
        self.set_requires_grad(self.netG, True)
        self.set_requires_grad(self.netS_G, True)
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
        self.dnet = StyleEncoder(1)

    def forward(self, imgs):
        bs, tot, w, h = imgs.shape
        data = torch.reshape(imgs, (bs*tot, 1, w, h))
        data = self.dnet(data)
        data = torch.reshape(data, (bs, tot, 128))
        return data

class DModel(nn.Module):
    def __init__(self):
        super(DModel, self).__init__()
        self.unet = UnetGenerator(
                input_nc = 1,
                output_nc = 10,
                use_style = True,
                num_downs = 6
                )
        self.dnet = StyleEncoder(10)
        self.linear = nn.Linear(128,1)

    def forward(self, imgs, style):
        bs, tot, w, h = imgs.shape
        style = style.unsqueeze(1).expand(bs, tot, -1).reshape(bs*tot, 128)
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
        self.unet = UnetGenerator(
                input_nc = 1,
                output_nc = 10,
                use_style = True,
                num_downs = 6
                )

    def forward(self, imgs, style):
        bs, tot, w, h = imgs.shape
        style = style.unsqueeze(1).expand(bs, tot, -1).reshape(bs*tot, 128)
        data = torch.reshape(imgs, (bs*tot, 1, w, h))
        data = self.unet(data, style).sum(1)
        data = torch.reshape(data, (bs, tot, w, h))
        data = F.tanh(data)*.5+.5
        return data

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=16,
                    use_style = False,
                 norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, use_style = use_style)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input, style):
        return self.model(input, style)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
            use_style = False,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.use_style = use_style
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            self.down = downconv
            self.mid = submodule
            self.up = nn.Sequential(uprelu, upconv, nn.Tanh())
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc*(2 if use_style else 1), outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            self.up = nn.Sequential(uprelu, upconv, upnorm)
            self.down = nn.Sequential(downrelu, downconv)
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            self.down = nn.Sequential(downrelu, downconv, downnorm)
            self.up = nn.Sequential(uprelu, upconv, upnorm)
            self.mid = submodule


    def forward(self, x, s):
        xx = x
        if self.outermost:
            x = self.down(x)
            x = self.mid(x, s)
            x = self.up(x)
            return x
        elif self.innermost:
            x = self.down(x)
            if self.use_style:
                x = torch.cat([x,s.unsqueeze(-1).unsqueeze(-1)], 1)
            x = self.up(x)
            return torch.cat([x, xx], 1)
        else:
            x = self.down(x)
            x = self.mid(x, s)
            x = self.up(x)
            return torch.cat([xx, x], 1)

class StyleEncoder(nn.Module):
    def __init__(self, in_channels):
        super(StyleEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.InstanceNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.InstanceNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.InstanceNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn6 = nn.InstanceNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=2, stride=2, padding=0, bias=False)
        self.bn7 = nn.InstanceNorm2d(128)
        self.weight_init()

    def weight_init(self):
        nn.init.xavier_uniform(self.conv1.weight)
        nn.init.xavier_uniform(self.conv2.weight)
        nn.init.xavier_uniform(self.conv3.weight)
        nn.init.xavier_uniform(self.conv4.weight)
        nn.init.xavier_uniform(self.conv5.weight)
        nn.init.xavier_uniform(self.conv6.weight)

    def forward(self, input):
        x = F.leaky_relu(self.bn1(self.conv1(input)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn6(self.conv6(x)), negative_slope=0.2)
        output = self.conv7(x).squeeze(-1).squeeze(-1)
        return output

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images
