import torch
import torch.nn as nn
import torch.nn.functional as F

class Resnet(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            ngf = 32,
            use_bias = True,
            norm_layer=nn.InstanceNorm2d, 
            use_dropout=False, 
            n_blocks=2, 
            padding_type='reflect'
            ):
        super(Resnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        models = []
        models_ = []
        models = models + [
                    nn.ConvTranspose2d(
                        in_channels,
                        ngf*(2**5),
                        kernel_size=4, 
                        stride=2, 
                        output_padding=0, 
                        padding=1, 
                        bias=True),
                    nn.LeakyReLU(0.2)
                    ]
        models += [
                ResnetBlock(
                    ngf * 2**5, 
                    padding_type=padding_type, 
                    norm_layer=norm_layer, 
                    use_dropout=use_dropout, 
                    use_bias=use_bias
                    )
                for _ in range(n_blocks)
            ]
        for i in range(4):
            models = models + [
                    nn.ConvTranspose2d(
                        ngf*2**(5-i),
                        ngf*2**(4-i),
                        kernel_size=4, 
                        stride=2, 
                        output_padding=0, 
                        padding=1, 
                        bias=True),
                    ]
            models += [nn.InstanceNorm2d(ngf*2**(4-i))]
            models += [ nn.LeakyReLU(0.2) ]
            if i>=2:
                models += [
                        ResnetBlock(
                            ngf * 2**(4-i), 
                            padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
                        for _ in range(n_blocks)
                        ]
        #models += [nn.InstanceNorm2d(ngf*2)]
        models += [
            nn.ConvTranspose2d(
                ngf*2,
                out_channels,
                kernel_size=4, 
                stride=2, 
                output_padding=0, 
                padding=1, 
                bias=True),
            ]
        models_ += [
            nn.Tanh(),
            ]
        self.model = nn.Sequential(*models)
        self.model_ = nn.Sequential(*models_)

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.model(x)
        #print(x)
        #t = x.view(5,10,64,32,32)[:,:,:10,:10,:10]
        #print((t - t.mean(1).unsqueeze(1))[0])
        x = self.model_(x)
        #print(x.shape)
        #t = x.view(5,10,1,64,64)[:,:,:1,:10,:10]
        #print((t - t.mean(1).unsqueeze(1))[0])
        return x

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        conv_block += [norm_layer(dim)]
        conv_block += [nn.LeakyReLU(.2)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        conv_block += [norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
