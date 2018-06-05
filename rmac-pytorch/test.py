import numpy as np
import torch
from torch.autograd import Variable
from model import RMAC_VGG
from matplotlib import pyplot as plt
import cv2
import torch.nn.functional as F
from sklearn.preprocessing import normalize as sknormalize
from sklearn.decomposition import PCA

net = RMAC_VGG('/ssd/pretrained/vgg16_caffe.pth')
net.init_modules()
net = net.cuda()

#
# def run_feature_processing_pipeline(features, d=128, whiten=True, copy=False, params=None):
#     """
#     Given a set of feature vectors, process them with PCA/whitening and return the transformed features.
#     If the params argument is not provided, the transformation is fitted to the data.
#     :param ndarray features:
#         image features for transformation with samples on the rows and features on the columns
#     :param int d:
#         dimension of final features
#     :param bool whiten:
#         flag to indicate whether features should be whitened
#     :param bool copy:
#         flag to indicate whether features should be copied for transformed in place
#     :param dict params:
#         a dict of transformation parameters; if present they will be used to transform the features
#     :returns ndarray: transformed features
#     :returns dict: transform parameters
#     """
#     # Normalize
#     features = normalize(features, copy=copy)
#
#     # Whiten and reduce dimension
#     if params:
#         pca = params['pca']
#         features = pca.transform(features)
#     else:
#         pca = PCA(n_components=d, whiten=whiten, copy=copy)
#         features = pca.fit_transform(features)
#         params = {'pca': pca}
#
#     # Normalize
#     features = normalize(features, copy=copy)
#
#     return features, params


def normalize(x, copy=False):
    """
    A helper function that wraps the function of the same name in sklearn.
    This helper handles the case of a single column vector.
    """
    if type(x) == np.ndarray and len(x.shape) == 1:
        return np.squeeze(sknormalize(x.reshape(1, -1), copy=copy))
    else:
        return sknormalize(x, copy=copy)


def preprocess(im):
    # rgb -> bgr
    # im = im[:, :, ::-1]
    im = im.astype(np.float32, copy=False)
    im -= np.array([[[102.9801, 115.9465, 122.7717]]])
    return im


if __name__ == '__main__':
    raw_img = cv2.imread('./sample.jpg', -1) # BGR
    raw_img = cv2.resize(raw_img, (224, 157), interpolation=cv2.INTER_LINEAR)
    query_img = preprocess(raw_img)
    query_img = torch.FloatTensor(query_img)
    print(query_img)
    query_img = query_img.permute(2, 0, 1).contiguous()
    print(query_img)
    query_img = query_img.unsqueeze(0).cuda()
    output = net(Variable(query_img))
    print(output)

