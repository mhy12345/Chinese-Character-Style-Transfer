import torch
import torch.nn as nn
from torch.nn import init

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel,self).__init__()
        use_bias = True
        self.downconv_1 = nn.Conv2d(1, 8, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        self.downconv_2 = nn.Conv2d(8, 16, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)

        self.upconv_2 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.upconv_1 = nn.ConvTranspose2d(16,1,kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.tanh = nn.Tanh()
        init_weights(self)

    def forward(self, content_imgs, style_imgs):
        content_imgs = torch.split(content_imgs, 1, 1)
        level_0 = content_imgs[-1];
        level_1 = self.downconv_1(level_0)
        level_2 = self.downconv_2(level_1)
        level_1 = torch.cat([level_1, self.upconv_2(level_2)] , 1)
        data = self.upconv_1(level_1)
        data = self.tanh(data)*.5+.5
        return data.squeeze(1)

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net
