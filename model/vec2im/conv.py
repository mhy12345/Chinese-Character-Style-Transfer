import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            ngf = 32,
            norm_layer = nn.InstanceNorm2d
            ):
        super(Conv, self).__init__()
        models = []
        for i in range(6):
            models = models + [
                        nn.ConvTranspose2d(
                            min(in_channels, in_channels*16//4**(i)),
                            min(in_channels, in_channels*16//4**(i+1)), 
                            kernel_size=3, 
                            stride=2, 
                            output_padding=1, 
                            padding=1, 
                            bias=True),
                        ]
            if i>=2 and i!=5:
                models += [nn.InstanceNorm2d(in_channels//2**(i+1))]
            if i!=5:
                models += [ nn.LeakyReLU(0.2) ]
        models += [nn.Tanh()]
        self.model = nn.Sequential(*models)
    
    def forward(self, x):
        x = self.model(x.unsqueeze(-1).unsqueeze(-1))
        return x
