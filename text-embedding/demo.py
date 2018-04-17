import numpy as np
import torch
from torch.autograd import Variable
from voc_data import VOCDetection
from model import Embedding_VGG
from matplotlib import pyplot as plt
import cv2
import torch.nn.functional as F
import itertools
num_search = 1
dataset = VOCDetection('./data/VOCdevkit2007', [('2007', 'test'), ('2007', 'trainval'), ('2012', 'trainval')])
embedding_net = Embedding_VGG('./data/pretrained_model/vgg16_caffe.pth')
embedding_net.init_modules()
embedding_net = embedding_net.cuda()


def preprocess(im):
    # rgb -> bgr
    im = im[:, :, ::-1]
    im = im.astype(np.float32, copy=False)
    im -= np.array([[[102.9801, 115.9465, 122.7717]]])
    return im


for cls in range(20):
    tot_cnt = dataset.cls_len(cls)
    query_img_raw, gt_boxes, h, w, im_id = dataset.pull_item(cls, 0)
    query_img_raw = cv2.resize(query_img_raw, (224, 224), interpolation=cv2.INTER_LINEAR)

    query_img = preprocess(query_img_raw)
    query_img = torch.FloatTensor(query_img)
    query_img = query_img.permute(2, 0, 1).contiguous()
    query_img = query_img.unsqueeze(0).cuda()

    search_img = []
    for i in range(1, min(num_search + 1, dataset.cls_len(cls))):
        raw_im, gt_boxes, h, w, im_id = dataset.pull_item(1, i)
        raw_im = cv2.resize(raw_im, (224, 224), interpolation=cv2.INTER_LINEAR)
        im_data = preprocess(raw_im)
        search_img.append(im_data)

    search_img = np.array(search_img)
    search_img = torch.FloatTensor(search_img)
    search_img = search_img.permute(0, 3, 1, 2).contiguous()
    search_img = search_img.cuda()
    avg_feat, query_feat_map = embedding_net(Variable(search_img), Variable(query_img))
    avg_feat = avg_feat.squeeze()
    print(avg_feat)
    heat_map = np.zeros((14, 14))
    for i, j in itertools.product(range(14), range(14)):
        #affinity = F.cosine_similarity(avg_feat.unsqueeze(0), query_feat_map[:, :, i, j])
        affinity = torch.sqrt(torch.pow(avg_feat - query_feat_map[0, :, i, j], 2).sum())
        heat_map[i, j] = affinity.data[0]
    #print(heat_map)
    #print(torch.from_numpy(heat_map))
    # plt.imshow(heat_map)
    # plt.show()
    # plt.imshow(query_img_raw)
    # plt.show()

    scam = heat_map
    scam = scam - np.min(scam)
    scam = scam / np.max(scam)
    scam = np.uint8(255 * scam)
    query_img_raw = query_img_raw[:, :, ::-1]
    sh = cv2.applyColorMap(cv2.resize(scam, (query_img_raw.shape[1], query_img_raw.shape[0])), cv2.COLORMAP_JET)
    sr = sh * 0.5 + query_img_raw * 0.5
    cv2.imwrite('./haha/n1orm%d.jpg' % cls, sr)


