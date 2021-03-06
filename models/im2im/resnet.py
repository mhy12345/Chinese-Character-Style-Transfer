import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from .resnet_block import ResnetBlock

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class Resnet(nn.Module):
    def __init__(self, in_channels, out_channels, 
            extra_channels,
            ngf=32, 
            norm_layer=nn.InstanceNorm2d, 
            use_dropout=False, 
            n_blocks=6, 
            padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Resnet, self).__init__()
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
                    ), True)
                )

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            models.append(
                    (nn.Sequential(
                        nn.Conv2d(
                            ngf * mult + extra_channels,
                            ngf * mult * 2, kernel_size=3,
                            stride=2, padding=1, bias=use_bias
                            ),
                        norm_layer(ngf * mult * 2),
                        nn.ReLU(True)
                        ), True)
                    )
            mult = 2**n_downsampling
        for i in range(n_blocks):
            models.append(
                    (
                    ResnetBlock(
                        ngf * mult, 
                        padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),False)
                    )
            pass

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            models.append(
                    (nn.Sequential(
                        nn.ConvTranspose2d(ngf * mult + extra_channels, 
                            int(ngf * mult / 2),
                            kernel_size=3, stride=2,
                            padding=1, output_padding=1,
                            bias=use_bias),
                        norm_layer(int(ngf * mult / 2)),
                        nn.ReLU(True)
                        ), True)
                    )
        models.append(
                (
                nn.Sequential(
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0),
                    nn.Tanh()
                    ),False)
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
            if style is not None and tag:
                data = torch.cat([data, style.expand(-1, -1, W, H)], 1)
            data = model(data)
        return data


