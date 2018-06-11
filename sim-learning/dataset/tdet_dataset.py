import torch.utils.data as data
import torch

from scipy.misc import imread
import numpy as np
import cv2

from dataset.coco_loader import COCOLoader
from dataset.voc_loader import VOCLoader


class TDetDataset(data.Dataset):
    def __init__(self, dataset_name, training, img_scale, keep_aspect_ratio=False):
        self.training = training
        self.img_scale = img_scale
        self.keep_aspect_ratio = keep_aspect_ratio
        self.per_class = [[] for i in range(80)]

        if dataset_name == 'coco2017train':
            self._dataset_loader = COCOLoader('../../data/coco/annotations/instances_train2017.json', '../../data/coco/images/train2017/')
        elif dataset_name == 'coco2017val':
            self._dataset_loader = COCOLoader('../../data/coco/annotations/instances_val2017.json', '../../data/coco/images/val2017/')

        else:
            print('@@@@@@@@@@ undefined dataset @@@@@@@@@@@')

        for i in range(len(self._dataset_loader)):
            here = self._dataset_loader.items[i]
            for cls in here['categories']:
                if len(self.per_class[cls]) > 0 and self.per_class[cls][-1] == i:
                    continue
                else:
                    self.per_class[cls].append(i)

    def __getitem__(self, index):
        im, gt_boxes, gt_categories, id = self.get_raw_data(index)
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

        # to tensor
        data = torch.from_numpy(im)
        data = data.permute(2, 0, 1).contiguous()
        gt_boxes = torch.from_numpy(gt_boxes)
        gt_categories = torch.from_numpy(gt_categories)

        image_level_label = torch.zeros(80)
        for label in gt_categories:
            image_level_label[label] = 1.0
        return data, gt_boxes, gt_categories, image_level_label, height_scale, width_scale, raw_img, id

    def len_per_cls(self, cls):
        return len(self.per_class[cls])

    def get_from_cls(self, cls, index):
        return self.__getitem__(self.per_class[cls][index])

    def get_raw_data(self, index):
        here = self._dataset_loader.items[index]

        assert here is not None
        im = imread(here['img_full_path'])

        # gray to rgb
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)

        gt_boxes = here['boxes'].copy()
        gt_categories = here['categories'].copy()
        id = here['id']
        return im, gt_boxes, gt_categories, id

    def __len__(self):
        return len(self._dataset_loader)


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
    image_level_labels = []
    height_scales = []
    width_scales = []
    raw_imgs = []
    ids = []

    for sample in batch:
        imgs.append(sample[0])
        gt_boxes.append(sample[1])
        gt_categories.append(sample[2])
        image_level_labels.append(sample[3])
        height_scales.append(sample[4])
        width_scales.append(sample[5])
        raw_imgs.append(sample[6])
        ids.append(sample[7])

    return torch.stack(imgs, 0), gt_boxes, gt_categories, torch.stack(image_level_labels, 0), height_scales, width_scales, raw_imgs, ids
