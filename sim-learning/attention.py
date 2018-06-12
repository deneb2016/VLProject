import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFeature(nn.Module):
    def __init__(self, base_model_path):
        super(AttentionFeature, self).__init__()
        self.model_path = base_model_path
        vgg = models.vgg16()
        print("Loading pretrained weights from %s" % (self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})
        self.base = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        self.gap = nn.AdaptiveAvgPool2d(1)

    # input: x(N, C, H, W) image
    # output: 512 dimension mean feature vector
    def get_mean_feature(self, x):
        c5_feature_map = self.base(x)
        feature_vector = self.gap(c5_feature_map).view(-1, 512)
        mean_feature = feature_vector.mean(0)
        return mean_feature

    # input: x(N, C, H, W) image, mean_feature(N, 512) dimension mean feature vectors
    # output: N * H * W attention
    def make_attention(self, x, mean_feature):
        N, C, H, W = x.size()
        c5_feature_map = self.base(x)
        norm1 = torch.norm(c5_feature_map, p=2, dim=1, keepdim=True)
        norm2 = torch.norm(mean_feature, p=2, dim=1, keepdim=True)
        c5_feature_map = c5_feature_map / (norm1 + 0.0000001)
        mean_feature = mean_feature / (norm2 + 0.0000001)
        attention = c5_feature_map * mean_feature.view(N, 512, 1, 1)
        attention = torch.sum(attention, 1).view(N, -1)

        mini, _ = torch.min(attention, dim=1, keepdim=True)
        attention = attention - mini
        maxi, _ = torch.max(attention, dim=1, keepdim=True)
        attention = attention / (maxi + 0.0000001)
        attention = attention.view(N, 20, 20)

        hard_attention = torch.ge(attention, 0.5).float()
        return hard_attention
