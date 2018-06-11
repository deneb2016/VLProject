import numpy as np
import torch
import pickle
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
train_data = TDetDataset(dataset_names=['coco2017train'], training=False, img_scale=224)
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


criterion = TripletLoss()
# criterion = nn.CosineEmbeddingLoss()
# criterion = nn.TripletMarginLoss(margin=1, p=2)
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)


def train(epoch):
    print("Training {}th epoch...".format(epoch))
    cum_loss = 0.0
    total_cnt = 0
    cnt = 0
    for index, data in enumerate(train_loader):
        img, _, _, image_level_label, _, _, _, img_id, _ = data
        img = Variable(img).cuda()
        label = image_level_label.cpu().numpy()
        positive_labels = []
        negative_labels = []
        for idx, pre_label in pre_labels.items():
            eq = np.equal(label, pre_label)
            indicator = np.prod(eq.astype(int))
            if indicator == 1:
                positive_labels.append(idx)
            else:
                negative_labels.append(idx)
        # print("{}/{}: {} positive samples and {} negative samples are found.".format(index, dataset_len, len(positive_labels), len(negative_labels)))
        pos_sel = np.random.choice(positive_labels, 1, replace=False)[0]
        neg_sel = np.random.choice(negative_labels, 1, replace=False)[0]
        if index == pos_sel:
            continue
        # print("Randomly selected positive and negative samples are: {}, {}".format(pos_sel, neg_sel))
        pos_img, _, _, pos_label, _, _, _, pos_id, _ = train_data[pos_sel]
        # pos_img = pos_img.unsqueeze(0)
        # pos_img = Variable(pos_img).cuda()
        neg_img, _, _, neg_label, _, _, _, neg_id, _ = train_data[neg_sel]
        print("img_id pair{}, {}, {}".format(img_id.numpy()[0], pos_id, neg_id))
        print(image_level_label)
        print(pos_label)
        print(neg_label)
    #     neg_img = neg_img.unsqueeze(0)
    #     neg_img = Variable(neg_img).cuda()
    #     output = net(img)
    #     pos_output = net(pos_img)
    #     neg_output = net(neg_img)
    #
    #     output = output.view(-1, 512)
    #     # psim = F.cosine_similarity(output, pos_output.view(-1, 512)).sum()
    #     # nsim = F.cosine_similarity(output, neg_output.view(-1, 512)).sum()
    #     # print("Similarity pos: {}, neg: {}".format(psim.data[0], nsim.data[0]))
    #     loss = criterion(output, pos_output.view(-1, 512), neg_output.view(-1, 512))
    #     # py = Variable(torch.cuda.FloatTensor([1.0] * output.size()[0]))
    #     # ny = Variable(torch.cuda.FloatTensor([-1.0] * output.size()[0]))
    #     # loss1 = criterion(output, pos_output.view(-1, 512), py)
    #     # loss2 = criterion(output, neg_output.view(-1, 512), ny)
    #     # loss = loss1 + loss2
    #     optimizer.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm(net.parameters(), 0.25)
    #     optimizer.step()
    #
    #     cum_loss += loss.data[0]
    #     cnt += 1
    #     total_cnt += 1
    #
    #     if index % 20 == 0:
    #         print('Train Epoch: {} [{}/{} ({:.0f}%)] cnt:{} \tLoss: {:.6f}'.format(
    #             # epoch, index, len(train_data), 100. * index / len(train_data), loss.data[0]))
    #             epoch, index, len(train_data), 100. * index / len(train_data), cnt, cum_loss/cnt))
    #         cum_loss = 0
    #         cnt = 0
    #     if index % 20000 == 0:
    #         torch.save(net.state_dict(), nowTime + ':ep1:' + str(index) + '.pt')
    # torch.save(net.state_dict(), nowTime+':lr0.005:ep'+str(epoch)+'.pt')
    # print("Finished training {}th epoch, total count:{} pairs.".format(epoch, total_cnt))


if __name__ == '__main__':
    for epoch in range(1, 11):
        train(epoch)
