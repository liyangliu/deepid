import numpy as np
import Image
import sys
sys.path.append('./python')
import caffe

# init
caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('models/deepid/resnet/solver_kps.prototxt')

solver.step(5)

for i in range(3):
  img = solver.net.blobs['data'].data[i, :, :, :]
  img = np.transpose(img, [1, 2, 0])
  img = Image.fromarray(np.uint8(img))
  img.save('models/deepid/test_kps/result%d.jpg'%(i+1))
  conv = solver.net.blobs['conv2_1_sum'].data[i, :, :, :]
  conv_diff = solver.net.blobs['conv2_1_sum'].diff[i, :, :, :]
  kps_label = solver.net.blobs['kps'].data[i, :, :, :]
  #print '===================================================\n'
  #print kps_label
  kps = solver.net.blobs['kps2'].data[i, :, :, :]
  kps_diff = solver.net.blobs['kps2'].diff[i, :, :, :]
  np.save('models/deepid/test_kps/conv_%d.npy'%(i+1), conv)
  np.save('models/deepid/test_kps/conv_diff_%d.npy'%(i+1), conv_diff)
  np.save('models/deepid/test_kps/kps_%d.npy'%(i+1), kps)
  np.save('models/deepid/test_kps/kps_diff_%d.npy'%(i+1), kps_diff)
