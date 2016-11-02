#! /usr/bin/python
'''visualize net'''

import sys
sys.path.append('./python')
import caffe
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# plt.rcParams['figure.figsize'] = (10, 10)
# plt.rcParams['image.cmap'] = 'gray'
# init
caffe.set_device(0)
caffe.set_mode_gpu()
model_def = 'examples/mnist/mnist_example/mnist_train_test.prototxt'
model_weights = 'examples/mnist/mnist_example/mnist_train_iter_10000.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)
center = net.params['center_loss'][0].data[...]

colors = ['#f31d29', '#b7509f', '#3853a4', '#6cbb50', '#ff8302', '#a84f41', '#7a3fd7', '#231f20', '#00fb0a', '#ffff48']

test_num = 10000
batch_size = 100
feat_num = center.shape[1]
feat_mat = np.zeros((test_num, feat_num))
iter_num = test_num / batch_size

acc = 0.
c = []
for i in range(iter_num):
  if i%10 == 0:
    print i
  net.forward()
  feat = net.blobs['ip1'].data[...].reshape((batch_size, feat_num))
  feat_mat[i * batch_size : (i + 1) * batch_size, :] = feat
  label = net.blobs['label'].data[...]
  for j in range(batch_size):
    c.append(colors[int(label[j])])
  cnt = 0
  for j in range(batch_size):
    dis_m = feat[j] - center
    dis = np.sum(dis_m * dis_m, axis=1)
    if np.argmin(dis) == label[j]:
      cnt += 1
  acc += cnt
acc /= test_num
print 'accuracy:', acc

# np.save('examples/deepid/test_kps/feat.npy', feat_mat)
# feat_mat = np.load('examples/deepid/test_kps/feat.npy')
# np.save('examples/deepid/test_kps/color.npy', c)
# label = np.load('examples/deepid/test_kps/label.npy')

# (u, s, v) = np.linalg.svd(feat_mat)
# print 'norm: ', np.sum(v[0] * v[0])
# pca_num = 3
# feat_pca = np.dot(feat_mat, v.T[:, :pca_num])
# center_pca = np.dot(center, v.T[:, :pca_num])

plt.figure()
plt.scatter(feat_mat[:, 0], feat_mat[:, 1], c=c)
plt.scatter(center[:, 0], center[:, 1], c=colors, s=80)
plt.axis('off')
plt.savefig('examples/deepid/test_kps/result/result_iter_%d.jpg'%test_num)
