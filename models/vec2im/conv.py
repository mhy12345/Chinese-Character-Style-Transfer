import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            ngf = 32,
            norm_layer = nn.InstanceNorm2d,
            n_blocks = None
            ):
        super(Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        models = []
        models = models + [
                    nn.ConvTranspose2d(
                        in_channels,
                        ngf*(2**5),
                        kernel_size=3, 
                        stride=2, 
                        output_padding=1, 
                        padding=1, 
                        bias=True),
                    nn.LeakyReLU(0.2)
                    ]
        for i in range(4):
            models = models + [
                        nn.ConvTranspose2d(
                            ngf*2**(5-i),
                            ngf*2**(4-i),
                            kernel_size=3, 
                            stride=2, 
                            output_padding=1, 
                            padding=1, 
                            bias=True),
                        ]
            if i>=2 and i!=5:
                models += [nn.BatchNorm2d( ngf*2**(4-i), ngf*2**(4-i))]
            if i!=5:
                models += [ nn.LeakyReLU(0.2) ]
        models = models + [
                    nn.ConvTranspose2d(
                        ngf*2,
                        out_channels,
                        kernel_size=3, 
                        stride=2, 
                        output_padding=1, 
                        padding=1, 
                        bias=True),
                    nn.Tanh(),
                    ]
        self.model = nn.Sequential(*models)
    
    def forward(self, x):
        x = self.model(x.unsqueeze(-1).unsqueeze(-1))
        return x
