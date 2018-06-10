import os
import numpy as np
import argparse
import pdb
import time

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler

from dataset.tdet_dataset import TDetDataset, tdet_collate
from model import FeatureExtractor
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Retrieval A')
    parser.add_argument('--nw', dest='num_workers', help='number of worker to load data', default=2, type=int)
    parser.add_argument('--bs', help='batch size', default=1, type=int)
    parser.add_argument('--img_scale', type=int, default=320)

    args = parser.parse_args()
    return args


# input: feature map(N * C * H * W)
# output: text embedding(C)
def make_text_embedding(feature_map):
    N = feature_map.size(0)
    C = feature_map.size(1)
    gap = feature_map.view(N, C, -1).mean(2)
    ret = gap.mean(0)
    return ret


# input: feature map(N * C * H * W), text embedding(C)
# output: attention(N * H * W)
def make_attention(feature_map, text_embedding):
    norm1 = torch.norm(feature_map, p=2, dim=1, keepdim=True)
    norm2 = torch.norm(text_embedding)
    feature_map = feature_map / norm1
    attention_feature = text_embedding.view(1, 512, 1, 1) / norm2
    attention = feature_map * attention_feature
    attention = torch.sum(attention, 1)

    shape = attention.size()
    softmax_attention = torch.exp(attention)
    #attention = (F.softmax(attention.view(1, -1))).view(shape)
    attention -= attention.min()
    attention = attention / attention.max()

    hard_attention = attention.clone()
    hard_attention[hard_attention.lt(0.5)] = 0
   # print(attention)

    return hard_attention


def eval():
    args = parse_args()

    print('Called with args:')
    print(args)
    np.random.seed(4)
    torch.manual_seed(2017)
    torch.cuda.manual_seed(1086)

    train_dataset = TDetDataset(dataset_name='coco2017train', training=True, img_scale=args.img_scale)
    val_dataset = TDetDataset(dataset_name='coco2017val', training=False, img_scale=args.img_scale)
    #val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, num_workers=args.num_workers, collate_fn=tdet_collate, pin_memory=True)

    feature_extractor = FeatureExtractor('./data/pretrained_model/vgg16_caffe.pth')
    feature_extractor.eval()
    feature_extractor.cuda()

    attention_features = [None for i in range(80)]
    CLS = 10
    K = 30
    for cls in range(CLS, CLS + 1):
        print(cls)
        attention_images = []
        for i in np.random.choice(range(train_dataset.len_per_cls(cls)), K, replace=False):
            data, gt_boxes, gt_categories, image_level_label, height_scale, width_scale, raw_img, id = train_dataset.get_from_cls(cls, i)
            attention_images.append(data.numpy())
        attention_images = np.array(attention_images)
        attention_images = torch.FloatTensor(attention_images)
        attention_images = attention_images.cuda()
        attention_images = Variable(attention_images)

        gap_features, _ = feature_extractor(attention_images)
        gap_features = gap_features.view(K, -1)
        attention_features[cls] = gap_features.mean(0).data

    print('attention feature extract finish')

    extracted_features = []
    target_class_idx = []
    attentions = []
    for i in range(len(val_dataset)):
        im_data, gt_boxes, _, weak_labels, hscales, wscales, raw_img, im_id = val_dataset[i]
        target_class_idx.append(i)
        print(i)
        im_data = Variable(im_data.unsqueeze(0).cuda())
        gap_features, feature_map = feature_extractor(im_data)
        feature_map = feature_map.data
        #plt.imshow(raw_img)
        #plt.show()

        attention, softmax_attention, hard_attention = attend(feature_map, attention_features[CLS])
       # print(attention)

        attentions.append(hard_attention.squeeze())
        # plt.imshow(attention[0].cpu().numpy())
        # plt.show()
        attended_feature_map = feature_map #* hard_attention
        # print(feature_map)
        # print(attention)
        # print(attended_feature_map)
        attended_gap = feature_extractor.gap(attended_feature_map).view(512)
        extracted_features.append(attended_gap)
        #print(attended_gap)

    for feat_idx, img_idx in enumerate(target_class_idx):

        if val_dataset[img_idx][3][CLS] < 1:
            continue
        sim = torch.zeros(len(val_dataset))
        for t_feat_idx, t_img_idx in enumerate(target_class_idx):
            if feat_idx == t_feat_idx:
                continue
            sim[t_img_idx] = F.cosine_similarity(extracted_features[feat_idx].unsqueeze(0), extracted_features[t_feat_idx].unsqueeze(0)).data[0]

        _, sorted_indices = torch.sort(sim, dim=0, descending=True)
        source_img = val_dataset[img_idx][6]
        source_data = val_dataset[img_idx][0]
        attention = attentions[img_idx]
        attended_image = (source_data.cuda() + torch.FloatTensor([102.9801, 115.9465, 122.7717]).unsqueeze(1).unsqueeze(2).cuda()) * F.upsample_bilinear(Variable(attention).unsqueeze(0).unsqueeze(0), (320, 320)).squeeze().data
        attended_image = attended_image.permute(1, 2, 0).cpu().clamp(max=250)
        print(attended_image, attended_image.max(), attended_image.min())
        attended_image = attended_image.numpy().astype(np.uint8)
        print('a')
        plt.imshow(source_img)
        plt.show()
        print('c')
        plt.imshow(attended_image)
        plt.show()
        for i in range(1):
            target_img = val_dataset[sorted_indices[i]][6]
            print('b')
            plt.imshow(target_img)
            plt.show()


if __name__ == '__main__':
    eval()