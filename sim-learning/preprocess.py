import numpy as np
import torch
import pickle
from torch.autograd import Variable
from matplotlib import pyplot as plt
from dataset.tdet_dataset import TDetDataset, tdet_collate
from torch.utils.data import DataLoader
import cv2
import torch.nn.functional as F

dataset = TDetDataset(dataset_names=['coco2017train'], training=False, img_scale=224)

def extract_and_save_labels():
    labels = {}
    dataloader = DataLoader(dataset)
    for idx, data in enumerate(dataloader):
        _, _, _, image_level_label, _, _, _, _, _ = data
        labels[idx] = image_level_label.cpu().numpy()
    pickle.dump(labels, open('labels.pickle', 'wb'))
    print("Finished writing id-label pairs")
    return 0


if __name__ == '__main__':
    #data, gt_boxes, gt_categories, image_level_label, height_scale, width_scale, raw_img, id, loader_index = dataset[1]
    extract_and_save_labels()
