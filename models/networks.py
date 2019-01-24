import importlib
import torch
import torch.nn as nn
from torch.nn import init
import logging
logger = logging.getLogger(__name__)

def create_model(name, *args, **kwargs):
    path = 'models.'+name
    modules = importlib.import_module(path)
    model = getattr(modules, name.capitalize())
    if model is None:
        logger.warn("There should be a modul named %s\n"%path)
    model = model(*args, **kwargs)
    return model

def create_layer(cls, name, args, kwargs):
    model_filename = 'models.'+cls+'.'+name
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = name.replace('_', '')
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    model = model(*args, **kwargs)
    init_net(model)
    return model

def create_im2im(name, *args, **kwargs):
    return create_layer('im2im', name, args, kwargs)

def create_im2vec(name, *args, **kwargs):
    return create_layer('im2vec', name, args, kwargs)

def create_vec2im(name, *args, **kwargs):
    return create_layer('vec2im', name, args, kwargs)

def create_mixer(name, *args, **kwargs):
    return create_layer('mixer', name, args, kwargs)

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

    logger.info('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, 
        #init_type='kaiming', 
        init_type='xavier', 
        init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

def shuffle_channels(data, tot):
    '''
    将data向量基于dim=1轴向后旋转位移tot位（用于打乱顺序）
    '''
    data = torch.cat((data[:,tot:,:], data[:,:tot,:]),1)
    return data
