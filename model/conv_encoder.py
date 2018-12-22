import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.InstanceNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.InstanceNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.InstanceNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn6 = nn.InstanceNorm2d(128)
        self.conv7 = nn.Conv2d(128, out_channels, kernel_size=2, stride=2, padding=0, bias=False)

    def forward(self, input):
        x = F.leaky_relu(self.bn1(self.conv1(input)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn6(self.conv6(x)), negative_slope=0.2)
        output = self.conv7(x).squeeze(-1).squeeze(-1)
        return output
