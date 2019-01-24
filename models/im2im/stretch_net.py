import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from .stretch_block import StretchBlock

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class StretchNet(nn.Module):
    def __init__(self, in_channels, out_channels, 
            extra_channels,
            ngf=32, 
            norm_layer=nn.InstanceNorm2d, 
            use_dropout=False, 
            n_blocks=6, 
            padding_type='reflect'):
        assert(n_blocks >= 0)
        super(StretchNet, self).__init__()
        self.extra_channels = extra_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        models = []
        models.append(
                (nn.Sequential(
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(
                        in_channels+extra_channels, 
                        ngf, kernel_size=7, padding=0,
                        bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)
                    ), 1)
                )

        n_downsampling = 1
        for i in range(n_downsampling):
            mult = 4**i
            models.append(
                    (nn.Sequential(
                        nn.Conv2d(
                            ngf * mult + extra_channels,
                            ngf * mult * 4, kernel_size=3,
                            stride=2, padding=1, bias=use_bias
                            ),
                        norm_layer(ngf * mult * 4),
                        nn.ReLU(True)
                        ), 1)
                    )
            mult = 4**n_downsampling
        for i in range(n_blocks):
            models.append(
                    (
                    StretchBlock(
                        ngf * mult, 64//(2**n_downsampling), 64//(2**n_downsampling),
                        self.extra_channels,
                        ),0)
                    )
            pass

        for i in range(n_downsampling):
            mult = 4**(n_downsampling - i)
            models.append(
                    (nn.Sequential(
                        nn.ConvTranspose2d(ngf * mult + extra_channels, 
                            int(ngf * mult / 4),
                            kernel_size=3, stride=2,
                            padding=1, output_padding=1,
                            bias=use_bias),
                        norm_layer(int(ngf * mult / 4)),
                        nn.ReLU(True)
                        ), 1)
                    )
        models.append(
                (
                nn.Sequential(
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0),
                    nn.Tanh()
                    ), 2
                    )
                )
        self.models = models
        for i, (model,tag) in enumerate(self.models):
            setattr(self, 'model_'+str(i), model)

    def forward(self, data, style):
        if style is not None:
            bs, _, tf = style.shape
            style = style.view(bs, tf, 1, 1)
        for model,tag in self.models:
            _, _, W, H = data.shape
            if style is not None and tag == 1:
                data = torch.cat([data, style.expand(-1, -1, W, H)], 1)
            if tag == 1 or tag == 2:
                data = model(data)
            else:
                data = model(data, style)
        return data


