import torch
import os
import numpy as np
import cv2
from scipy.misc import imread


class WebDataLoader:
    def __init__(self, img_scale):
        self.index_to_classname = []
        self.img_scale = img_scale
        for line in open('./dataset/class_name.txt').readlines():
            self.index_to_classname.append(line.rstrip())

    def get_images(self, cls):
        dirname = '../../data/web/googlecoco/%s/' % self.index_to_classname[cls]
        filenames = os.listdir(dirname)
        ret = []
        for fn in filenames:
            im = imread(dirname + fn)
            im = self.preprocess(im)
            if im is None:
                continue
            ret.append(im)

        print(cls, len(ret))
        ret = np.array(ret)
        ret = torch.from_numpy(ret)
        ret = ret.permute(0, 3, 1, 2).contiguous()

        return ret

    def preprocess(self, im):
        #print(im.shape)
        if len(im.shape) != 3 or im.shape[2] != 3:
            return None
        # rgb -> bgr
        im = im[:, :, ::-1]
        im = im.astype(np.float32, copy=False)
        im -= np.array([[[102.9801, 115.9465, 122.7717]]])

        # image rescale
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])

        height_scale, width_scale = self.img_scale / im_shape[0], self.img_scale / im_shape[1]
        im = cv2.resize(im, (self.img_scale, self.img_scale), interpolation=cv2.INTER_LINEAR)

        return im