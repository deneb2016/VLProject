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


class FeatureExtractor(nn.Module):
    def __init__(self, base_model_path):
        super(FeatureExtractor, self).__init__()
        self.model_path = base_model_path
        vgg = models.vgg16()
        print("Loading pretrained weights from %s" % (self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})
        self.base = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        self.extra_conv = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, attention=None):
        N, C, H, W = x.size()
        c5_feature_map = self.base(x)
        c6_feature_map = F.relu(self.extra_conv(c5_feature_map), inplace=True)

        fac = 1
        if attention is not None:
            c6_feature_map = c6_feature_map * attention.view(N, 1, 20, 20)
            fac = attention.view(N, 20 * 20).sum(1)
            fac = 400 / fac
            fac = fac.view(N, 1)
        feature_vector = self.gap(c6_feature_map).view(-1, 1024) * fac
        return feature_vector
