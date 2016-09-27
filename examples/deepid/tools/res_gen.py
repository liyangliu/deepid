#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def parse_args():
    """Parse input arguments
    """

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--solver',
                        help=('Output solver.prototxt file'),
                        default='solver.prototxt')
    parser.add_argument('--train_val',
                        help=('Output train_val.prototxt file'),
                        default = 'train_val.prototxt')
    parser.add_argument('--block_number', nargs='*',
                        help=('Block number for each stage.'),
                        default=[1, 1, 1])
    parser.add_argument('--type', type=int,
                        help=('0 for deploy.prototxt, 1 for train_val.prototxt.'),
                        default=1)

    args = parser.parse_args()
    return args

def generate_data_layer(data_root, data_name, batch_size):
    data_layer_str = '''name: "ResNet"
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    mean_file: "%s/%s_mean.binaryproto"
  }
  data_param {
    source: "%s/%s_train_lmdb"
    batch_size: %d
    backend: LMDB
  }
}
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mirror: false
    mean_file: "%s/%s_mean.binaryproto"
  }
  data_param {
    source: "%s/%s_val_lmdb"
    batch_size: %d
    backend: LMDB
  }
}
'''%(data_root, data_name, data_root, data_name, batch_size, data_root, data_name, data_root, data_name, batch_size)
    return data_layer_str

def generate_conv_layer(kernel_size, kernel_num, stride, pad, layer_name, bottom, top, filler="xavier"):
    conv_layer_str = '''layer {
  name: "%s"
  type: "Convolution"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: %d
    pad: %d
    kernel_size: %d
    stride: %d
    weight_filler {
      type: "%s"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
'''%(layer_name, bottom, top, kernel_num, pad, kernel_size, stride, filler)
    return conv_layer_str

def generate_bn_layer(layer_name, bottom, top):
    bn_layer_str = '''layer {
  name: "%s"
  type: "BatchNorm"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
     lr_mult: 0
     decay_mult: 0
  }
  param {
     lr_mult: 0
     decay_mult: 0
  }
  batch_norm_param {
    use_global_stats: false
  }
  include {
    phase: TRAIN
  }
}
layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "BatchNorm"
  batch_norm_param {
    use_global_stats: true
  }
  include {
    phase: TEST
  }
}
'''%(layer_name, bottom, top, layer_name, bottom, top)
    return bn_layer_str

def generate_scale_layer(layer_name, bottom, top):
    scale_layer_str = '''layer {
  name: "%s"
  type: "Scale"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 1
    decay_mult: 0
  }
  param {
     lr_mult: 1
     decay_mult: 0
  }
  scale_param {
    bias_term: true
  }
}
'''%(layer_name, bottom, top)
    return scale_layer_str

def generate_conv_bn_scale_layer(kernel_size, kernel_num, stride, pad, layer_name, bottom, top, filler="xavier"):
    return generate_conv_layer(kernel_size, kernel_num, stride, pad, layer_name, bottom, top, filler) +\
    generate_bn_layer(layer_name+'_bn', top, top) + generate_scale_layer(layer_name+'_scale', top, top)

def generate_pooling_layer(kernel_size, stride, pool_type, layer_name, bottom, top):
    pool_layer_str = '''layer {
  name: "%s"
  type: "Pooling"
  bottom: "%s"
  top: "%s"
  pooling_param {
    pool: %s
    kernel_size: %d
    stride: %d
  }
}
'''%(layer_name, bottom, top, pool_type, kernel_size, stride)
    return pool_layer_str

def generate_fc_layer(num_output, layer_name, bottom, top, filler="msra"):
    fc_layer_str = '''layer {
  name: "%s"
  type: "InnerProduct"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
     num_output: %d
     weight_filler {
       type: "%s"
       std: 0.001
     }
     bias_filler {
       type: "constant"
       value: 0
     }
  }
}
'''%(layer_name, bottom, top, num_output, filler)
    return fc_layer_str

def generate_eltwise_layer(layer_name, bottom_1, bottom_2, top, op_type="SUM"):
    eltwise_layer_str = '''layer {
  name: "%s"
  type: "Eltwise"
  bottom: "%s"
  bottom: "%s"
  top: "%s"
  eltwise_param {
    operation: %s
  }
}
'''%(layer_name, bottom_1, bottom_2, top, op_type)
    return eltwise_layer_str

def generate_activation_layer(layer_name, bottom, top, act_type="ReLU"):
    act_layer_str = '''layer {
  name: "%s"
  type: "%s"
  bottom: "%s"
  top: "%s"
}
'''%(layer_name, act_type, bottom, top)
    return act_layer_str

def generate_dropout_layer(bottom, drop_ratio=0.5):
    drop_layer_str = '''layer {
  name: "dropout"
  type: "Dropout"
  bottom: "%s"
  top: "%s"
  dropout_param {
    dropout_ratio: %0.1f
  }
}
'''%(bottom, bottom, drop_ratio)
    return drop_layer_str

def generate_softmax_loss(bottom):
    softmax_loss_str = '''layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "%s"
  bottom: "label"
  top: "loss"
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "%s"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }
}
'''%(bottom, bottom)
    return softmax_loss_str

def get_layer_name(stage, block, layer, relu_sum):
    layer_name = 'conv%d_%d_%d'%(stage+1, block, layer)
    if relu_sum == 0:
      return layer_name
    elif relu_sum == 1:
      return layer_name+'_relu'
    elif relu_sum == 2:
      return layer_name[:-1]+'sum'

def generate_train_val_stage(stage, kernel_num, last_top, network_str, args):
    b = 1

    l = 1
    conv_layer_name = get_layer_name(stage, b, l, 0)
    relu_layer_name = get_layer_name(stage, b, l, 1)
    network_str += generate_conv_bn_scale_layer(3, kernel_num, 2, 1, conv_layer_name, last_top, conv_layer_name)
    network_str += generate_activation_layer(relu_layer_name, conv_layer_name, conv_layer_name, 'ReLU')

    l = 2
    conv_layer_name_bottom = conv_layer_name
    conv_layer_name = get_layer_name(stage, b, l, 0)
    relu_layer_name = get_layer_name(stage, b, l, 1)
    network_str += generate_conv_bn_scale_layer(3, kernel_num, 1, 1, conv_layer_name, conv_layer_name_bottom, conv_layer_name)

    l = 3
    conv_layer_name_bottom = conv_layer_name
    conv_layer_name = get_layer_name(stage, b, l, 0)
    sum_layer_name = get_layer_name(stage, b, l, 2)
    network_str += generate_conv_layer(1, kernel_num, 2, 0, conv_layer_name, last_top, conv_layer_name)
    network_str += generate_eltwise_layer(sum_layer_name, conv_layer_name_bottom, conv_layer_name, sum_layer_name, 'SUM')
    network_str += generate_activation_layer(relu_layer_name, sum_layer_name, sum_layer_name, 'ReLU')
    last_top = sum_layer_name

    for b in xrange(2, args.block_number[stage-1]+1):
        l = 1
        conv_layer_name = get_layer_name(stage, b, l, 0)
        relu_layer_name = get_layer_name(stage, b, l, 1)
        network_str += generate_conv_bn_scale_layer(3, kernel_num, 1, 1, conv_layer_name, last_top, conv_layer_name)
        network_str += generate_activation_layer(relu_layer_name, conv_layer_name, conv_layer_name, 'ReLU')

        l = 2
        conv_layer_name_bottom = conv_layer_name
        conv_layer_name = get_layer_name(stage, b, l, 0)
        relu_layer_name = get_layer_name(stage, b, l, 1)
        sum_layer_name = get_layer_name(stage, b, l, 2)
        network_str += generate_conv_bn_scale_layer(3, kernel_num, 1, 1, conv_layer_name, conv_layer_name_bottom, conv_layer_name)

        network_str += generate_eltwise_layer(sum_layer_name, last_top, conv_layer_name, sum_layer_name, 'SUM')
        network_str += generate_activation_layer(relu_layer_name, sum_layer_name, sum_layer_name, 'ReLU')
        last_top = sum_layer_name
    return network_str, last_top

def generate_train_val(num_class, args, data_root, data_name, batch_size):
    network_str = generate_data_layer(data_root, data_name, batch_size)
    '''before stage'''
    last_top = 'data'

    kernel_num = 20
    network_str += generate_conv_bn_scale_layer(3, kernel_num, 2, 1, 'conv1', last_top, 'conv1')
    network_str += generate_activation_layer('conv1_relu', 'conv1', 'conv1', 'ReLU')
    # network_str += generate_pooling_layer(3, 2, 'MAX', 'pool1', 'conv1', 'pool1')
    '''stage 1'''
    last_top = 'conv1'
    for b in xrange(1, args.block_number[0]+1):
        network_str += generate_conv_bn_scale_layer(3, kernel_num, 1, 1, 'conv2_%d_1'%b, last_top, 'conv2_%d_1'%b)
        network_str += generate_activation_layer('conv2_%d_1_relu'%b, 'conv2_%d_1'%b, 'conv2_%d_1'%b, 'ReLU')

        network_str += generate_conv_bn_scale_layer(3, kernel_num, 1, 1, 'conv2_%d_2'%b, 'conv2_%d_1'%b, 'conv2_%d_2'%b)

        network_str += generate_eltwise_layer('conv2_%d_sum'%b, last_top, 'conv2_%d_2'%b, 'conv2_%d_sum'%b, 'SUM')
        network_str += generate_activation_layer('conv2_%d_2_relu'%b, 'conv2_%d_sum'%b, 'conv2_%d_sum'%b, 'ReLU')
        last_top = 'conv2_%d_sum'%b

    for s in range(1, len(args.block_number)):
        kernel_num *= 2
        network_str, last_top = generate_train_val_stage(s + 1, kernel_num, last_top, network_str, args)

    # network_str += generate_pooling_layer(7, 1, 'AVE', 'pool2', last_top, 'pool2')
    network_str += generate_fc_layer(kernel_num * 2, 'feature', last_top, 'feature', 'gaussian')
    network_str += generate_dropout_layer('feature')
    network_str += generate_fc_layer(num_class, 'fc', 'feature', 'fc', 'gaussian')
    network_str += generate_softmax_loss('fc')
    return network_str

def generate_solver(train_val, batch_size, num_imgs):
    solver_str = '''net: "%s"
test_iter: %d
test_interval: 1000
test_initialization: true
display: 100
base_lr: 0.1
lr_policy: "step"
stepsize: 10000
gamma: 0.5
max_iter: 80000
momentum: 0.9
weight_decay: 0.0001
snapshot: 80000
snapshot_prefix: "models/deepid/resnet/resnet"
solver_mode: GPU'''%(train_val, num_imgs//(batch_size*4) + 1, )
    return solver_str

def main():
    args = parse_args()
    data_root = 'data/CASIA'
    data_name = 'CASIA'
    batch_size = 256
    num_imgs = 133342
    num_class = 9727
    train_val = 'models/deepid/resnet/train_val.prototxt'
    solver = 'models/deepid/resnet/solver.prototxt'
    solver_str = generate_solver(train_val, batch_size, num_imgs)
    network_str = generate_train_val(num_class, args, data_root, data_name, batch_size)
    fp = open(solver, 'w')
    fp.write(solver_str)
    fp.close()
    fp = open(train_val, 'w')
    fp.write(network_str)
    fp.close()

if __name__ == '__main__':
    main()
