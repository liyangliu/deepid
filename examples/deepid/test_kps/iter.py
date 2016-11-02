import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('./python')
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('examples/mnist/mnist_example/mnist_solver.prototxt')

def get_batch_color(net, batch_size, colors):
    c = []
    batch_label = net.blobs['label'].data[...]
    for i in range(batch_size):
        c.append(colors[int(batch_label[i])])
    return c

def get_batch_feat(net, batch_size, feat_blob, feat_num):
    return net.blobs[feat_blob].data[...].reshape((batch_size, feat_num))

colors = ['#f31d29', '#b7509f', '#3853a4', '#6cbb50', '#ff8302', '#a84f41', '#7a3fd7', '#231f20', '#00fb0a', '#ffff48']
test_num = 10000
batch_size = 100
iter_num = test_num / batch_size
train_iter = 10000
snap_shot = 100
feat_num = solver.net.params['center_loss'][0].data[...].shape[1]

for i in range(train_iter/snap_shot):
    solver.test_nets[0].share_with(solver.net)
    center = solver.net.params['center_loss'][0].data[...]
    c = []
    feat_mat = np.zeros((test_num, feat_num))
    for j in range(iter_num):
        solver.test_nets[0].forward()
        c.extend(get_batch_color(solver.test_nets[0], batch_size, colors))
        feat_mat[j * batch_size : (j + 1) * batch_size, :] = get_batch_feat\
            (solver.test_nets[0], batch_size, 'ip1', feat_num)
    plt.figure()
    plt.scatter(feat_mat[:, 0], feat_mat[:, 1], c=c)
    plt.scatter(center[:, 0], center[:, 1], c=colors, s=80)
    plt.axis('off')
    plt.savefig('examples/deepid/test_kps/result/result_iter_%d.jpg'%(i*snap_shot))
    plt.close()

    solver.step(snap_shot)
