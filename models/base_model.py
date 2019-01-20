import os
import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.model_names = '__base__'

    def initialize(self, opt):
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save_networks(self, epoch):
        name = self.model_names
        save_filename = '%s_net_%s.pth' % (epoch, name)
        save_path = os.path.join(self.save_dir, save_filename)
        model = self
        torch.save(model.state_dict(), save_path)

    def load_networks(self, epoch):
        name = self.model_names
        load_filename = '%s_net_%s.pth' % (epoch, name)
        load_path = os.path.join(self.save_dir, load_filename)
        self.load_state_dict(torch.load(load_path))
