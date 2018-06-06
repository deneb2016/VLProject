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

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Retrieval A')
    parser.add_argument('--nw', dest='num_workers', help='number of worker to load data', default=2, type=int)
    parser.add_argument('--bs', help='batch size', default=2, type=int)
    parser.add_argument('--img_scale', type=int, default=320)

    args = parser.parse_args()
    return args


def eval():
    args = parse_args()

    print('Called with args:')
    print(args)
    np.random.seed(4)
    torch.manual_seed(2017)
    torch.cuda.manual_seed(1086)

    output_dir = args.save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    val_dataset = TDetDataset(dataset_names=['coco2017_val'], training=False, img_scale=args.img_scale, keep_aspect_ratio=args.keep_aspect)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, num_workers=args.num_workers, collate_fn=tdet_collate, pin_memory=True)

    feature_extractor = FeatureExtractor('./data/pretrained_model/vgg16_caffe.pth')
    output_file_name = os.path.join(output_dir, 'eval_{}_{}_{}.txt'.format(checkpoint['net'], checkpoint['session'], checkpoint['iterations']))
    output_file = open(output_file_name, 'w')
    output_file.write(str(args))
    output_file.write('\n')

    model.cuda()
    model.eval()
    source_acc = validate(model, source_val_dataloader)
    print('@@@@@@@@@ source acc @@@@@@@@@@@')
    output_file.write('@@@@@@@@@ source acc @@@@@@@@@@\n')
    for i in range(80):
        print(source_acc[i])
        output_file.write('%.4f\n' % source_acc[i])
    print('source mean %f' % source_acc[20:].mean())

    target_acc = validate(model, target_val_dataloader)
    print('@@@@@@@@@ target acc @@@@@@@@@@@')
    output_file.write('@@@@@@@@@ target acc @@@@@@@@@@\n')
    for i in range(80):
        print(target_acc[i])
        output_file.write('%.4f\n' % target_acc[i])

    print('target mean %f' % target_acc[:20].mean())

if __name__ == '__main__':
    eval()