import numpy as np
import torch
import pickle
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
from model import FeatureExtractor
from attention import AttentionFeature
from matplotlib import pyplot as plt
from dataset.tdet_dataset import TDetDataset, tdet_collate
import cv2
import torch.nn.functional as F
import torch.optim as optim
import datetime

pre_labels = pickle.load(open("labels.pickle", "rb"))
sim_model = FeatureExtractor('../../data/pretrained_model/vgg16_caffe.pth')
attention_model = AttentionFeature('../../data/pretrained_model/vgg16_caffe.pth')
train_data = TDetDataset(dataset_name='coco2017train', training=False, img_scale=320)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, num_workers=0, shuffle=True, collate_fn=tdet_collate, pin_memory=True)


sim_model.train()
attention_model.eval()

sim_model.cuda()
attention_model.cuda()

criterion = torch.nn.TripletMarginLoss()
optimizer = optim.SGD(sim_model.parameters(), lr=0.001, momentum=0.9)


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

    attention -= attention.min()
    attention = attention / attention.max()
    hard_attention = attention
    mask = torch.ge(hard_attention, 0.5).float()
    hard_attention = hard_attention * mask
    #print(hard_attention)
    return hard_attention


def train(args, epoch):
    print("Training {}th epoch...".format(epoch))
    cum_loss = 0.0
    total_cnt = 0
    cnt = 0

    # Generate text-embeddings to give attention
    attention_features = [None for i in range(80)]
    CLS = 80
    K = args.emb_K
    for cls in range(0, CLS):
        attention_images = []
        for i in np.random.choice(range(train_data.len_pos_img(cls)), K, replace=False):
            data, gt_boxes, gt_categories, image_level_label, height_scale, width_scale, raw_img, id = train_data.get_pos_img(cls, i)
            attention_images.append(data.numpy())
        attention_images = np.asarray(attention_images)
        attention_images = torch.FloatTensor(attention_images)
        attention_images = attention_images.cuda()
        attention_images = Variable(attention_images)
        features = attention_model.get_mean_feature(attention_images)
        attention_features[cls] = features.data.clone()

    print("Finish generation embedding for {}classes.".format(CLS))
    # Train
    for index, data in enumerate(train_loader):
        img, _, _, image_level_label, _, _, raw_img, img_id = data
        N, C, H, W = img.size()
        img = Variable(img).cuda()
        label = image_level_label.cpu().numpy()

        mean_features = torch.zeros((N, 512)).cuda()
        pos_img = torch.zeros((N, 3, H, W)).cuda()
        neg_img = torch.zeros((N, 3, H, W)).cuda()

        for i in range(img.size(0)):
            pos_cls = np.random.choice(np.where(label[i, :] == 1)[0], 1)[0]
            here_pos_img, _, _, _, _, _, _, _ = train_data.get_pos_img(pos_cls, np.random.randint(train_data.len_pos_img(pos_cls)))
            here_neg_img, _, _, _, _, _, _, _ = train_data.get_neg_img(pos_cls, np.random.randint(train_data.len_neg_img(pos_cls)))
            pos_img[i, :] = here_pos_img[:]
            neg_img[i, :] = here_neg_img[:]
            mean_features[i, :] = attention_features[pos_cls]

        pos_img = Variable(pos_img)
        neg_img = Variable(neg_img)
        mean_features = Variable(mean_features)

        attention = Variable(attention_model.make_attention(img, mean_features).data)
        pos_attention = Variable(attention_model.make_attention(pos_img, mean_features).data)
        neg_attention = Variable(attention_model.make_attention(neg_img, mean_features).data)

        output = sim_model(img, attention)
        pos_output = sim_model(pos_img, pos_attention)
        neg_output = sim_model(neg_img, neg_attention)

        # for i in range(img.size(0)):
        #     plt.imshow(raw_img[i])
        #     plt.show()
        #     plt.imshow(attention[i, :].data.cpu().numpy())
        #     plt.show()

        loss = criterion(output, pos_output, neg_output)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(sim_model.parameters(), 10.0)
        optimizer.step()

        cum_loss += loss.data[0]
        cnt += 1
        total_cnt += 1


        if index % 100 == 0:
            cum_loss = cum_loss / cnt
            print(cum_loss)
            cnt = 0
            cum_loss = 0

    print("Finished training {}th epoch, total count:{} pairs.".format(epoch, total_cnt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Similarity training')
    parser.add_argument('--emb_K', help='K used in text embedding', default=30)
    parser.add_argument('--epoch', default=10)
    args = parser.parse_args()
    for epoch in range(1, args.epoch + 1):
        train(args, epoch)

