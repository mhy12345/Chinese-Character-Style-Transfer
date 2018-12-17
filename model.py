import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

class SmartModel(nn.Module):
    def __init__(self):
        super(SmartModel,self).__init__()
        self.style_encoder = StyleEncoder(10)
        self.unet = UnetGenerator(
                input_nc = 1,
                output_nc = 1,
                num_downs = 6
                )

    def forward(self, style_imgs, content_imgs):
        style_imgs = torch.split(style_imgs, 1, 1)
        data = []
        for style_img in style_imgs:
            data.append(self.style_encoder(style_img))
        data = torch.stack(data,1).mean(1)
        style = data

        content_imgs = torch.split(content_imgs, 1, 1)
        data = []
        for content_img in content_imgs:
            data.append(self.unet(content_img, style))
        data = torch.stack(data,1).mean(1)
        return data

class Hacker(nn.Module):
    def __init__(self):
        self.last = None
        pass
    
    def forward(self, x):
        return self.last

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
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
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
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
            self.up = nn.Sequential(uprelu, upconv, nn.Tanh())
            self.down = downconv
            self.mid = submodule
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc,
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
        if self.outermost:
            x = self.down(x)
            x = self.mid(x, s)
            x = self.up(x)
            return x
        elif self.innermost:
            xx = x
            x = self.down(x)
            x = torch.cat([x,s], 1)
            x = self.up(x)
            return torch.cat([x, xx], 1)
        else:
            xx = x
            x = self.down(x)
            x = self.mid(x, s)
            x = self.up(x)
            return torch.cat([x, xx], 1)

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
