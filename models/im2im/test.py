import torch

from stretch_block import StretchBlock
from stretch_net import StretchNet

bs, tot, H, W = 10,1,64,64

sc = 120

data = torch.randn(bs, tot, H, W)
ext = torch.randn(bs, tot, sc)

model = StretchNet(tot, 1, sc)
model(data, ext)
