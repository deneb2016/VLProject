import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, base_model_path):
        super(FeatureExtractor, self).__init__()
        self.model_path = base_model_path

        vgg = models.vgg16()
        print("Loading pretrained weights from %s" % (self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        self.base = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        self.fc = nn.Sequential(*list(vgg.classifier._modules.values())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fix the layers before conv3:
        for layer in range(len(self.base)):
            for p in self.base[layer].parameters(): p.requires_grad = False

    def forward(self, im_data):
        base_feat = self.base(im_data)
        gap_feat = self.gap(base_feat)
        #fc_feat = self.fc(base_feat.view(base_feat.size(0), -1))

        return gap_feat, base_feat