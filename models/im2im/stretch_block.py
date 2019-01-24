import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

# Define a resnet block
class StretchBlock(nn.Module):
    def __init__(self, dim, H_in, W_in,
            style_channels,
            norm_layer = nn.InstanceNorm2d):
        super(StretchBlock, self).__init__()
        self.H_in = H_in
        self.W_in = W_in
        self.style_channels = style_channels
        self.identical = self.zoom_in_block(dim, stride_w = 1, stride_h = 1, norm_layer = norm_layer)
        self.vert_zoom_in = self.zoom_in_block(dim, stride_w = 1, stride_h = 3, norm_layer = norm_layer)
        self.vert_zoom_out = self.zoom_out_block(dim, stride_w = 1, stride_h = 2, norm_layer = norm_layer)
        self.hori_zoom_in = self.zoom_in_block(dim, stride_w = 3, stride_h = 1, norm_layer = norm_layer)
        self.hori_zoom_out = self.zoom_out_block(dim, stride_w = 2, stride_h = 1, norm_layer = norm_layer)

        self.final_block = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim*5 + self.style_channels,dim, kernel_size=3, bias=True),
                norm_layer(dim),
                nn.LeakyReLU(.2)
                )

    def zoom_out_block(self, dim, stride_w, stride_h, norm_layer, use_dropout = False):
        H_in = self.H_in
        W_in = self.W_in
        H_out = (H_in - 1)*stride_h + 1*(3-1) + 1
        W_out = (W_in - 1)*stride_w + 1*(3-1) + 1
        pad_top, pad_left = (H_in - H_out)//2, (W_in - W_out)//2
        pad_bot, pad_right = H_in - H_out- pad_top, W_in - W_out - pad_left
        conv_block = []
        conv_block += [
                nn.ConvTranspose2d(dim, dim, 
                    stride = (stride_h, stride_w),
                    kernel_size=3, padding=0, bias=True),
                ]
        conv_block += [
                nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bot)),
                norm_layer(dim)]
        return nn.Sequential(*conv_block)

    '''
        H_out = (H_in + 2*padding - dilation * (kernel_size - 1) - 1) / stride_h + 1
        W_out = (W_in + 2*padding - dilation * (kernel_size - 1) - 1) / stride_w + 1
    '''
    def zoom_in_block(self, dim, stride_w, stride_h, norm_layer, use_dropout = False):
        H_in = self.H_in
        W_in = self.W_in
        H_out = (H_in+1 - 1 * (3 - 1) ) // stride_h + 1
        W_out = (W_in+1 - 1 * (3 - 1) ) // stride_w + 1
        pad_top, pad_left = (H_in - H_out)//2, (W_in - W_out)//2
        pad_bot, pad_right = H_in - H_out- pad_top, W_in - W_out - pad_left
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [
                nn.Conv2d(dim, dim, 
                    stride = (stride_h, stride_w),
                    kernel_size=3, padding=0, bias=True),
                ]
        conv_block += [
                norm_layer(dim),
                nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bot)),
                ]
        return nn.Sequential(*conv_block)

    def forward(self, x, ext_vec):
        bs, tot, H, W = x.shape
        a = x + self.identical(x)
        b = x + self.vert_zoom_in(x)
        c = x + self.vert_zoom_out(x)
        d = x + self.hori_zoom_in(x)
        e = x + self.hori_zoom_out(x)
        data = torch.cat((a,b,c,d,e, ext_vec.expand(-1,-1,H,W)),1)
        data = x + self.final_block(data)
        return data
