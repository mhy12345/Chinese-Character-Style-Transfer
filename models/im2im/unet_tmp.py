import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 extra_channels,
                 num_downs=6, ngf=16,
                 norm_layer=nn.InstanceNorm2d, 
                 ):
        super(Unet, self).__init__()
        self.extra_channels = extra_channels

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
                ngf * 8, 
                ngf * 8, 
                in_channels=self.extra_channels, 
                submodule=None, 
                norm_layer=norm_layer, 
                innermost=True)

        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, 
                    in_channels=self.extra_channels, 
                    submodule=unet_block, 
                    norm_layer=norm_layer, 
                    )

        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, 
                in_channels=self.extra_channels, 
                submodule=unet_block, 
                norm_layer=norm_layer)

        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, 
                in_channels=self.extra_channels, 
                submodule=unet_block, 
                norm_layer=norm_layer)

        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf, 
                in_channels=self.extra_channels, 
                submodule=unet_block, 
                norm_layer=norm_layer)

        unet_block = UnetSkipConnectionBlock(ngf, out_channels, 
                in_channels=extra_channels, 
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
            inner_nc, outer_nc, 
            in_channels=None,
            submodule=None, 
            outermost=False, 
            innermost=False, 
            norm_layer=nn.InstanceNorm2d, 
            ):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if in_channels is None:
            in_channels = outer_nc
        downconv = nn.Conv2d(
                inner_ncin_channels, 
                inner_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            self.down = nn.Sequential(downconv)
            self.up = nn.Sequential(uprelu, upconv, nn.Tanh())
            self.submodule = submodule
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            self.down = nn.Sequential(downrelu, downconv)
            self.up = nn.Sequential(uprelu, upconv, upnorm)
            self.submodule = None
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            self.down = nn.Sequential(downrelu, downconv, downnorm)
            self.up = nn.Sequential(uprelu, upconv, upnorm)


    def forward(self, x, styles):
        style_x = torch.cat((x, styles.expand(-1,-1,x.size(-2),x.size(-1))),1)
        if self.outermost:
            return self.model(style_x)
        else:
            return torch.cat([x, self.model(style_x)], 1)
