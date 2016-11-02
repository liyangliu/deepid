import numpy as np
# import Image
import sys
sys.path.append('./python')
import caffe

# init
caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('examples/mnist/mnist_example/mnist_solver.prototxt')

solver.step(1)

  # img = solver.net.blobs['data'].data[i, :, :, :]
  # img = np.transpose(img, [1, 2, 0])
  # img = Image.fromarray(np.uint8(img))
  # conv = solver.net.blobs['conv3_1_sum'].data[i, :, :, :]
  # conv_diff = solver.net.blobs['conv3_1_sum'].diff[i, :, :, :]
  # kps_label = solver.net.blobs['kps'].data[i, :, :, :]
  # kps_w = solver.net.params['kpsg_conv5_1_1'][0].data
  # kps_b = solver.net.params['kpsg_conv5_1_1'][1].data
  # kps = solver.net.blobs['kpsg'].data[i, :, :, :]
  # kps_diff = solver.net.blobs['kpsg'].diff[i, :, :, :]

x = solver.net.blobs['ip1'].data[...]
label = solver.net.blobs['label'].data[...]
c = solver.net.params['center_loss'][0].data[...]
l = solver.net.blobs['center_loss'].data[...]
M = 64
N = 10
K = 2
a = 0.2
# cy = np.zeros((M, K))
# for j in range(M):
    # cy[j] = c[int(label[j])]

lj = np.zeros((N,))
loss = 0
for i in range(M):
    xi = x[i]
    ximc = xi - c
    dis = np.sqrt(np.sum(ximc * ximc, axis=1))
    dism1 = dis - 1
    fc_diff = np.zeros(K)
    for j in range(N):
        if j == label[i]:
            lj[j] = 1
            pa = dism1[j] + a
        else:
            lj[j] = -1
            pa = -dism1[j] + a
        if pa > 0:
            loss += pa
            fc_diff += lj[j] * ximc[j] / dis[j]
    fc_diff /= M
    print 'diff tru:', fc_diff
    print 'diff est:', solver.net.blobs['ip1'].diff[i], '\n'
print 'loss tru:', loss/M
print 'loss est:', l
