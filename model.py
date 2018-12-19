import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools

class SmartModel(nn.Module):
    def __init__(self):
        super(SmartModel,self).__init__()
        self.style_encoder = StyleEncoder(10)
        self.input_norm = nn.BatchNorm2d(10)
        self.unet = UnetGenerator(
                input_nc = 1,
                output_nc = 1,
                use_style = False,
                num_downs = 6
                )
        self.final = UnetGenerator(
                input_nc = 10,
                output_nc = 1,
                use_style = False,
                num_downs = 6
                )
        init_net(self)

    def forward(self, content_imgs, style_imgs):
        style_imgs = torch.split(style_imgs, 1, 1)
        data = []
        for style_img in style_imgs:
            data.append(self.style_encoder(style_img))
        data = torch.stack(data,1).mean(1)
        style = data

        content_imgs = self.input_norm(content_imgs)
        content_imgs = torch.split(content_imgs, 1, 1)
        data = []
        for content_img in content_imgs:
            data.append(self.unet(content_img, style))
        data = torch.stack(data,1).squeeze(2)
        data = self.final(data, None)*.5+.5
        return data.squeeze(1)


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=16,
            use_style = False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
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
        print("USE_BIAS: ",use_bias)
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
                x = torch.cat([x,s], 1)
            x = self.up(x)
            return torch.cat([x, xx], 1)
        else:
            x = self.down(x)
            x = self.mid(x, s)
            x = self.up(x)
            return torch.cat([xx, torch.rand_like(x)], 1)

class StyleEncoder(nn.Module):
    def __init__(self, style_batch):
        super(StyleEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=2, stride=2, padding=0, bias=False)
        self.bn7 = nn.BatchNorm2d(512)
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
        output = F.leaky_relu(self.bn7(self.conv7(x)), negative_slope=0.2)

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
