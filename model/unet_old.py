import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, 
            use_style = False,
            num_downs = 6, ngf=16,
            norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(UNet, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, in_channels=None, submodule=None, norm_layer=norm_layer, innermost=True, use_style = use_style)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, in_channels=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, in_channels=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, in_channels=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, in_channels=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(out_channels, ngf, in_channels=in_channels, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input, style):
        return self.model(input, style)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, in_channels=None,
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
        if in_channels is None:
            in_channels = outer_nc
        downconv = nn.Conv2d(in_channels, inner_nc, kernel_size=4,
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
