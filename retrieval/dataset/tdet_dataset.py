import torch.utils.data as data
import torch

from scipy.misc import imread
import numpy as np
import cv2

from dataset.coco_loader import COCOLoader
from dataset.voc_loader import VOCLoader


class TDetDataset(data.Dataset):
    def __init__(self, dataset_names, training, img_scale, keep_aspect_ratio=False):
        self.training = training
        self.img_scale = img_scale
        self.keep_aspect_ratio = keep_aspect_ratio
        self._dataset_loaders = []

        for name in dataset_names:
            if name == 'coco60_train':
                self._dataset_loaders.append(COCOLoader('./data/coco/annotations/coco60_train_reindex_21413_61353.json', './data/coco/images/train2017/', '../repo/proposals/MCG_COCO'))
            elif name == 'coco60_val':
                self._dataset_loaders.append(COCOLoader('./data/coco/annotations/coco60_val_reindex_900_2575.json', './data/coco/images/val2017/', '../repo/proposals/MCG_COCO'))
            elif name == 'coco_voc_val':
                self._dataset_loaders.append(COCOLoader('./data/coco/annotations/voc20_val_reindex_740_2844.json', './data/coco/images/val2017/', '../repo/proposals/MCG_COCO'))
            elif name == 'voc07_trainval':
                self._dataset_loaders.append(VOCLoader('./data/VOCdevkit2007', [('2007', 'trainval')], '../repo/proposals/MCG_VOC'))
            elif name == 'voc07_test':
                self._dataset_loaders.append(VOCLoader('./data/VOCdevkit2007', [('2007', 'test')], '../repo/proposals/MCG_VOC'))
            else:
                print('@@@@@@@@@@ undefined dataset @@@@@@@@@@@')

    def __getitem__(self, index):
        im, gt_boxes, gt_categories, proposals, id, loader_index = self.get_raw_data(index)
        raw_img = im.copy()

        # rgb -> bgr
        im = im[:, :, ::-1]

        # random flip
        if self.training and np.random.rand() > 0.5:
            im = im[:, ::-1, :]
            raw_img = raw_img[:, ::-1, :].copy()

            flipped_gt_boxes = gt_boxes.copy()
            flipped_gt_boxes[:, 0] = im.shape[1] - gt_boxes[:, 2]
            flipped_gt_boxes[:, 2] = im.shape[1] - gt_boxes[:, 0]
            gt_boxes = flipped_gt_boxes

            flipped_proposals = proposals.copy()
            flipped_proposals[:, 0] = im.shape[1] - proposals[:, 2]
            flipped_proposals[:, 2] = im.shape[1] - proposals[:, 0]
            proposals = flipped_proposals

        # cast to float type and mean subtraction
        im = im.astype(np.float32, copy=False)
        im -= np.array([[[102.9801, 115.9465, 122.7717]]])

        # image rescale
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])

        if self.keep_aspect_ratio:
            im_scale = self.img_scale / float(im_size_min)
            im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            height_scale = im_scale
            width_scale = im_scale
        else:
            height_scale, width_scale = self.img_scale / im_shape[0], self.img_scale / im_shape[1]
            im = cv2.resize(im, (self.img_scale, self.img_scale), interpolation=cv2.INTER_LINEAR)
        gt_boxes[:, 0] *= width_scale
        gt_boxes[:, 1] *= height_scale
        gt_boxes[:, 2] *= width_scale
        gt_boxes[:, 3] *= height_scale

        proposals[:, 0] *= width_scale
        proposals[:, 1] *= height_scale
        proposals[:, 2] *= width_scale
        proposals[:, 3] *= height_scale


        # to tensor
        data = torch.from_numpy(im)
        data = data.permute(2, 0, 1).contiguous()
        gt_boxes = torch.from_numpy(gt_boxes)
        proposals = torch.from_numpy(proposals)
        gt_categories = torch.from_numpy(gt_categories)

        image_level_label = torch.zeros(80)
        for label in gt_categories:
            image_level_label[label] = 1.0
        return data, gt_boxes, gt_categories, proposals, image_level_label, height_scale, width_scale, raw_img, id, loader_index

    def get_raw_data(self, index):
        here = None
        loader_index = 0
        for loader in self._dataset_loaders:
            if index < len(loader):
                here = loader.items[index]
                break
            else:
                index -= len(loader)
                loader_index += 1

        assert here is not None
        im = imread(here['img_full_path'])

        # gray to rgb
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)

        gt_boxes = here['boxes'].copy()
        gt_categories = here['categories'].copy()
        proposals = np.load(here['proposal_path']).astype(np.float32)
        id = here['id']
        return im, gt_boxes, gt_categories, proposals, id, loader_index

    def __len__(self):
        tot_len = 0
        for loader in self._dataset_loaders:
            tot_len += len(loader)
        return tot_len


def tdet_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    imgs = []
    gt_boxes = []
    gt_categories = []
    proposals = []
    image_level_labels = []
    height_scales = []
    width_scales = []
    raw_imgs = []
    ids = []
    loader_indices = []

    for sample in batch:
        imgs.append(sample[0])
        gt_boxes.append(sample[1])
        gt_categories.append(sample[2])
        proposals.append(sample[3])
        image_level_labels.append(sample[4])
        height_scales.append(sample[5])
        width_scales.append(sample[6])
        raw_imgs.append(sample[7])
        ids.append(sample[8])
        loader_indices.append(sample[9])

    return torch.stack(imgs, 0), gt_boxes, gt_categories, proposals, torch.stack(image_level_labels, 0), height_scales, width_scales, raw_imgs, ids, loader_indices
