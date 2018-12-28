import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, in_channels):
        super(Conv, self).__init__()
        self.A_channels = in_channels//2
        self.B_channels = in_channels - self.A_channels
        if in_channels > 1:
            self.partA = Conv(self.A_channels)
            self.partB = Conv(self.B_channels)
            block = []
            block += [nn.ReflectionPad2d(1)]
            block += [
                    nn.Conv2d(2, 2, kernel_size=3, padding=p, bias=use_bias),
                    nn.InstanceNorm2d(2),
                    nn.ReLU(True)]
            block += [
                    nn.ReflectionPad2d(1)
                    ]
            block += [
                    nn.Conv2d(2, 1, kernel_size=3, padding=p, bias=use_bias),
                    ]
            self.mixer = nn.Sequential(*block)

    def forward(self, data, styles, texts):
        bs, tot, W, H = data.shape
        if tot == 1:
            return data
        data = torch.split(data, [totA, totB], 1)
        resA = self.partA(data[0])
        resB = self.partB(data[1])
        resAB = torch.cat([resA, resB], 1)
        return self.mixer(resAB)
