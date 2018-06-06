import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding_VGG(nn.Module):
    def __init__(self, base_model_path):
        super(Embedding_VGG, self).__init__()
        self.model_path = base_model_path

    def init_modules(self):
        vgg = models.vgg16()
        print("Loading pretrained weights from %s" % (self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        self.base = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fix the layers before conv3:
        for layer in range(len(self.base)):
            for p in self.base[layer].parameters(): p.requires_grad = False

    def forward(self, im_data, query_img):
        base_feat = self.base(im_data)
        pooled_feat = self.gap(base_feat)
        avg_feat = pooled_feat.mean(0)

        #feat_vectors = base_feat.permute(0, 2, 3, 1).contiguous().view(-1, 512)



        return avg_feat, self.base(query_img)
