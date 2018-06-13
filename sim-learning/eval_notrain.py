import numpy as np
import torch
import pickle
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
from model import FeatureExtractor, SIM_VGG
from attention import AttentionFeature
from matplotlib import pyplot as plt
from dataset.tdet_dataset import TDetDataset, tdet_collate
import cv2
import time
import torch.nn.functional as F
import torch.optim as optim
import os
import datetime

IM_SIZE = 224
sim_model = SIM_VGG('../../data/pretrained_model/vgg16_caffe.pth', IM_SIZE)
attention_model = AttentionFeature(None, IM_SIZE)

train_data = TDetDataset(dataset_name='coco2017train', training=False, img_scale=IM_SIZE)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=20, num_workers=0, shuffle=True, collate_fn=tdet_collate, pin_memory=True)

val_data = TDetDataset(dataset_name='coco2017val', training=False, img_scale=IM_SIZE)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, num_workers=0, shuffle=False, collate_fn=tdet_collate, pin_memory=True)

sim_model.eval()
attention_model.eval()

sim_model.cuda()
attention_model.cuda()


def eval(args):

    # # Generate text-embeddings to give attention
    # if args.attention:
    #     attention_features = [None for i in range(80)]
    #     CLS = 80
    #     K = args.emb_K
    #     for cls in range(0, CLS):
    #         attention_images = []
    #         for i in np.random.choice(range(train_data.len_pos_img(cls)), K, replace=False):
    #             data, gt_boxes, gt_categories, image_level_label, height_scale, width_scale, raw_img, id = train_data.get_pos_img(cls, i)
    #             attention_images.append(data.numpy())
    #         attention_images = np.asarray(attention_images)
    #         attention_images = torch.FloatTensor(attention_images)
    #         attention_images = attention_images.cuda()
    #         attention_images = Variable(attention_images)
    #         features = attention_model.get_mean_feature(attention_images)
    #         attention_features[cls] = features.data.clone()
    #
    # print("Finish generation embedding for {}classes.".format(CLS))
    # start = time.time()
    # attended_features = torch.zeros((80, len(val_data), 512)).cuda()
    # for index, data in enumerate(val_loader):
    #     img, _, _, image_level_label, _, _, raw_img, img_id = data
    #     query_img = Variable(img).cuda()
    #
    #     if args.attention:
    #         attention_base_feat = attention_model.get_base_feature(query_img)
    #     _, sim_base_feat = sim_model(query_img)
    #
    #     for cls in range(80):
    #         if args.attention:
    #             attention = attention_model.make_attention(query_img, Variable(attention_features[cls].unsqueeze(0)), attention_base_feat)
    #         else:
    #             attention = None
    #         here, _ = sim_model(query_img, attention, sim_base_feat)
    #         attended_features[cls, index, :] = here.data
    #
    #     if index % 100 == 0:
    #         print(index, time.time() - start)
    #         start = time.time()
    #
    # torch.save(attended_features, './features_notrain_noattention.pth')

    attended_features = torch.load('./features_notrain_noattention.pth')
    top_k = np.array([10, 30, 100, 300, 500], dtype=float)
    tot_recall_sum = np.array([0, 0, 0, 0, 0], dtype=float)
    tot_precision_sum = np.array([0, 0, 0, 0, 0], dtype=float)
    tot_query = 0
    for cls in range(16, 80):
        cls_recall_sum = np.array([0, 0, 0, 0, 0], dtype=float)
        cls_precision_sum = np.array([0, 0, 0, 0, 0], dtype=float)

        tot_cnt = val_data.len_pos_img(cls)
        tot_query += tot_cnt
        this_cls_features = attended_features[cls]
        for i in range(len(val_data)):
            if cls not in val_data.get_label(i):
                continue

            #print(i)
            distance = this_cls_features - this_cls_features[i].clone()
            distance = torch.norm(distance, p=2, dim=1)
            distance[i] = 999999999
            _, sorted_indices = torch.sort(distance)

            query_img = val_data.get_raw_img(i)
            # plt.imshow(query_img)
            # plt.show()

            cor_cnt = 0
            recall = []
            precision = []

            for j in range(500):
                #print(i, j)
                #target_img = val_data.get_raw_img(sorted_indices[j])
                # plt.imshow(target_img)
                # plt.show()
                if cls in val_data.get_label(sorted_indices[j]):
                    cor_cnt += 1

                if j + 1 in top_k:
                    recall.append(cor_cnt / (tot_cnt - 1))
                    precision.append(cor_cnt / (j + 1))

            recall = np.array(recall)
            precision = np.array(precision)

            cls_recall_sum += recall
            cls_precision_sum += precision

            tot_recall_sum += recall
            tot_precision_sum += precision
            if i % 1000 == 0:
                print(i)

        cls_recall_sum = cls_recall_sum / tot_cnt
        cls_precision_sum = cls_precision_sum / tot_cnt
        print(cls, cls_recall_sum, cls_precision_sum)
        print('------------------------------------------------')

    tot_recall_sum = tot_recall_sum / tot_query
    tot_precision_sum = tot_precision_sum / tot_query
    print(tot_recall_sum)
    print(tot_precision_sum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='retrieval eval')
    parser.add_argument('--emb_K', help='K used in text embedding', default=30)
    parser.add_argument('--attention', default=True)

    args = parser.parse_args()

    eval(args)

