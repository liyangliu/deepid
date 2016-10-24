#! /usr/bin/python
'''visualize net'''

import sys
sys.path.append('./python')
import caffe
from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.cmap'] = 'gray'
# init
caffe.set_device(0)
caffe.set_mode_gpu()
model_def = 'models/deepid/resnet/kps/train_val.prototxt'
model_weights = 'models/deepid/resnet/kps/resnet_p56x56_f0_k5_s3_b1x1x1x1/resnet_Celeb_1703_iter_4215.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)
net.forward()

def vis_square(data, file_name):
    data = (data - data.min()) / (data.max() - data.min())
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))
               + ((0, 0),) * (data.ndim - 3))
    data = np.pad(data, padding, mode='constant', constant_values=1)
    data = data.reshape((n, n) + data.shape[1:]).\
        transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data)
    plt.axis('off')
    plt.imsave(file_name, data)

i = 0
save_root = 'models/deepid/test_kps/'
feat = net.blobs['conv3_1_sum'].data[i]
vis_square(feat, save_root + 'feat.jpg')
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1), save_root + 'filters.jpg')
