import numpy as np
import torch
import pickle
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
from model import SIM_VGG
from matplotlib import pyplot as plt
from dataset.tdet_dataset import TDetDataset, tdet_collate
import cv2
import torch.nn.functional as F
import torch.optim as optim
import datetime

pre_labels = pickle.load(open("labels.pickle", "rb"))
net = SIM_VGG('/ssd/pretrained/vgg16_caffe.pth')
net.init_modules()
net = net.cuda()
train_data = TDetDataset(dataset_name='coco2017train', training=False, img_scale=224)
train_loader = DataLoader(train_data)
dataset_len = len(train_data)

now = datetime.datetime.now()
nowTime = now.strftime('%H:%M:%S')

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, input, positive_pair, negative_pair):
        pos_sim = F.cosine_similarity(input, positive_pair).sum()
        neg_sim = F.cosine_similarity(input, negative_pair).sum()
        zero = Variable(torch.cuda.FloatTensor([0.0] * pos_sim.size()[0]))
        alpha = Variable(torch.cuda.FloatTensor([0.3] * pos_sim.size()[0]))
        ranking_loss = torch.max(zero, alpha - pos_sim + neg_sim)
        ptemp = pos_sim.data[0]
        ntemp = neg_sim.data[0]
        print("Similarity pos: {}, neg: {}".format(ptemp, ntemp))
        return ranking_loss


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
    """
    shape = attention.size()
    softmax_attention = torch.exp(attention)
    attention -= attention.min()
    attention = attention / attention.max()

    hard_attention = attention.clone()
    """
    hard_attention = attention
    #print(hard_attention)
    #mask = torch.ge(hard_attention, 0.5).float()
    #hard_attention = hard_attention * mask
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
        for i in np.random.choice(range(train_data.len_per_cls(cls)), K, replace=False):
            data, gt_boxes, gt_categories, image_level_label, height_scale, width_scale, raw_img, id = train_data.get_from_cls(cls, i)
            attention_images.append(data.numpy())
        attention_images = np.asarray(attention_images)
        attention_images = torch.FloatTensor(attention_images)
        attention_images = attention_images.cuda()
        attention_images = Variable(attention_images)
        features = net(attention_images)
        gap_features = F.adaptive_avg_pool2d(features, 1)
        gap_features = gap_features.view(K, -1)
        attention_features[cls] = gap_features.mean(0).data

    print("Finish generation embedding for {}classes.".format(CLS))
    # Train
    for index, data in enumerate(train_loader):
        img, _, _, image_level_label, _, _, _, img_id = data
        img = Variable(img).cuda()
        label = image_level_label.cpu().numpy()

        # Train mode
        # 1: to train every label per a single image
        # 2: to train for only one randomly selected label per a single image
        # default mode 2

        # Select one class to give attention
        # Randomly select a positive sample containing the class
        # Avoid selecting the original image
        pos_classes = np.where(label[0] == 1)[0]
        pos_class_sel = np.random.choice(pos_classes.size, 1, replace=False)[0]
        pos_sel = index
        while index == pos_sel:
            pos_sel = np.random.choice(train_data.len_per_cls(pos_class_sel), 1, replace=False)[0]
        pos_img, _, _, pos_label, _, _, _, pos_id = train_data.get_from_cls(pos_class_sel, pos_sel)
        pos_img = pos_img.unsqueeze(0)
        pos_img = Variable(pos_img).cuda()

        # Randomly choose a class not included in the image
        # Randomly choose a negative sample
        # Check if the image contains positive class
        while True:
            neg_classes = np.where(label[0] == 0)[0]
            neg_class_sel = np.random.choice(neg_classes.size, 1, replace=False)[0]
            neg_sel = np.random.choice(train_data.len_per_cls(neg_class_sel), 1, replace=False)[0]
            neg_img, _, _, neg_label, _, _, _, neg_id  = train_data.get_from_cls(neg_class_sel, neg_sel)
            if not neg_label[pos_class_sel] == 1:
                break;
        neg_img = neg_img.unsqueeze(0)
        neg_img = Variable(neg_img).cuda()
        output = net(img)
        pos_output = net(pos_img)
        neg_output = net(neg_img)

        # Give attention to the feature map
        # Use attention_features[cls] to give attention
        output = make_attention(output, Variable(attention_features[pos_class_sel]))
        pos_output = make_attention(pos_output, Variable(attention_features[pos_class_sel]))
        neg_output = make_attention(neg_output, Variable(attention_features[pos_class_sel]))
        output = output.view(-1, 7*7)
        pos_output = pos_output.view(-1, 7*7)
        neg_output = neg_output.view(-1, 7*7)

        # criterion = nn.CosineEmbeddingLoss()
        # criterion = nn.TripletMarginLoss(margin=1, p=2)
        criterion = TripletLoss()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
        # psim = F.cosine_similarity(output, pos_output.view(-1, 512)).sum()
        # nsim = F.cosine_similarity(output, neg_output.view(-1, 512)).sum()
        # print("Similarity pos: {}, neg: {}".format(psim.data[0], nsim.data[0]))
        #loss = criterion(output, pos_output.view(-1, 512), neg_output.view(-1, 512))
        # py = Variable(torch.cuda.FloatTensor([1.0] * output.size()[0]))
        # ny = Variable(torch.cuda.FloatTensor([-1.0] * output.size()[0]))
        # loss1 = criterion(output, pos_output.view(-1, 512), py)
        # loss2 = criterion(output, neg_output.view(-1, 512), ny)
        # loss = loss1 + loss2
        loss = criterion(output, pos_output, neg_output)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(net.parameters(), 0.25)
        optimizer.step()

        cum_loss += loss.data[0]
        cnt += 1
        total_cnt += 1

        if index % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] cnt:{} \tLoss: {:.6f}'.format(
                # epoch, index, len(train_data), 100. * index / len(train_data), loss.data[0]))
                epoch, index, len(train_data), 100. * index / len(train_data), cnt, cum_loss/cnt))
            cum_loss = 0
            cnt = 0
        if index % 20000 == 0:
            torch.save(net.state_dict(), nowTime + ':ep1:' + str(index) + '.pt')
    torch.save(net.state_dict(), nowTime+':lr0.005:ep'+str(epoch)+'.pt')
    print("Finished training {}th epoch, total count:{} pairs.".format(epoch, total_cnt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Similarity training')
    parser.add_argument('--emb_K', help='K used in text embedding', default=30)
    parser.add_argument('--epoch', default=10)
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--lr', type=float, default=0.005)
    args = parser.parse_args()
    for epoch in range(1, args.epoch + 1):
        train(args, epoch)

