import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F


class SIM_VGG(nn.Module):
    def __init__(self, base_model_path):
        super(SIM_VGG, self).__init__()
        self.model_path = base_model_path

    def init_modules(self):
        vgg = models.vgg16()
        print("Loading pretrained weights from %s" % (self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})
        self.features = nn.Sequential(*list(vgg.features._modules.values())[:])
        #self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        #x = self.gap(x)
        return x

