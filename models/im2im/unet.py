import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from .resnet_block import ResnetBlock

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 extra_channels,
                 n_blocks = 6, 
                 n_downs = 6,
                 norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(Unet, self).__init__()
        self.extra_channels = extra_channels
        ngf = in_channels

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
                ngf * 8 + self.extra_channels, ngf * 8, ngf * 8,
                n_blocks = n_blocks,
                submodule=None, 
                norm_layer=norm_layer, 
                innermost=True)

        for i in range(n_downs - 4):
            unet_block = UnetSkipConnectionBlock(
                    ngf * 8 + self.extra_channels, ngf * 8, ngf * 8, 
                    n_blocks = n_blocks,
                    submodule=unet_block, 
                    norm_layer=norm_layer, 
                    use_dropout=use_dropout)

        unet_block = UnetSkipConnectionBlock(
                ngf * 4 + self.extra_channels, ngf * 8, ngf * 4, 
                n_blocks = n_blocks,
                submodule=unet_block, 
                norm_layer=norm_layer)

        unet_block = UnetSkipConnectionBlock(
                ngf * 2 + self.extra_channels, ngf * 4, ngf * 2, 
                n_blocks = n_blocks,
                submodule=unet_block, 
                norm_layer=norm_layer)

        unet_block = UnetSkipConnectionBlock(
                ngf + self.extra_channels, ngf * 2, out_channels, 
                n_blocks = n_blocks,
                submodule=unet_block, 
                outermost=True, 
                norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input, style):
        style = style.unsqueeze(-1).unsqueeze(-1)
        return self.model(input, style)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, 
            inner_nc, middle_nc, outer_nc, 
            n_blocks=4,
            submodule=None, 
            outermost=False, 
            innermost=False, 
            norm_layer=nn.InstanceNorm2d, 
            use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        downconv = nn.Conv2d(inner_nc, middle_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        
        downresnet = nn.Sequential(*[ResnetBlock(
            middle_nc,
            padding_type = 'reflect',
            norm_layer = norm_layer,
            use_dropout = use_dropout,
            use_bias = use_bias
            ) for _ in range(n_blocks)])
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)

        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(middle_nc*2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            self.down = nn.Sequential(downconv, downresnet)
            self.up = nn.Sequential(uprelu, upconv, nn.Tanh())
            self.submodule = submodule
        elif innermost:
            upconv = nn.ConvTranspose2d(middle_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            self.down = nn.Sequential(downrelu, downconv)
            self.up = nn.Sequential(uprelu, upconv, upnorm)
            self.submodule = None
        else:
            upconv = nn.ConvTranspose2d(middle_nc*2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            self.down = nn.Sequential(downrelu, downconv, downresnet, downnorm)
            self.up = nn.Sequential(uprelu, upconv, upnorm)
            self.submodule = submodule

    def forward(self, x, styles):
        data = torch.cat((x, styles.expand(-1,-1,x.size(-2),x.size(-1))),1)
        #print('DOWN>', data.shape)
        data = self.down(data)
        #print('CALL>', data.shape)
        if not self.innermost:
            data = self.submodule(data, styles)
        #print('UP>',data.shape)
        data = self.up(data)
        if not self.outermost:
            data = torch.cat([x, data], 1)
        #print('EXIT>',data.shape)
        return data
