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
                        default=[1, 1, 1, 1])
    parser.add_argument('--type', type=int,
                        help=('0 for deploy.prototxt, 1 for train_val.prototxt.'),
                        default=1)

    args = parser.parse_args()
    return args

def generate_data_layer(data_root, data_name, batch_size, pix):
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
    mean_file: "%s/%s_%dx%d_mean.binaryproto"
  }
  data_param {
    source: "%s/%s_%dx%d_train_lmdb"
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
    mean_file: "%s/%s_%dx%d_mean.binaryproto"
  }
  data_param {
    source: "%s/%s_%dx%d_val_lmdb"
    batch_size: %d
    backend: LMDB
  }
}
'''%(data_root, data_name, pix, pix, data_root, data_name, pix, pix, batch_size,
     data_root, data_name, pix, pix, data_root, data_name, pix, pix, batch_size)
    return data_layer_str

def generate_data_kps_layer(data_root, data_name, batch_size, pix):
    data_kps_layer_str = '''layer {
  name: "kps"
  type: "Data"
  top: "kps"
  include {
    phase: TRAIN
  }
  data_param {
    source: "%s/%s_%dx%d_train_kps_lmdb"
    batch_size: %d
    backend: LMDB
  }
}
layer {
  name: "kps"
  type: "Data"
  top: "kps"
  include {
    phase: TEST
  }
  data_param {
    source: "%s/%s_%dx%d_val_kps_lmdb"
    batch_size: %d
    backend: LMDB
  }
}
'''%(data_root, data_name, pix, pix, batch_size,
     data_root, data_name, pix, pix, batch_size)
    return data_kps_layer_str

def generate_conv_layer(kernel_size, kernel_num, stride, pad, layer_name, bottom, top, filler="msra"):
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

def generate_conv_bn_scale_layer(kernel_size, kernel_num, stride, pad, layer_name, bottom, top, filler="msra"):
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

def generate_kps_layer(loc, height, width, scale, layer_name, bottom, top):
    kps_layer_str = '''layer {
  name: "%s"
  type: "Kps"
  bottom: "%s"
  bottom: "kps"
  top: "%s"
  kps_param {
    kps_id: %d
    kps_height: %d
    kps_width: %d
    kps_scale: %f
  }
}
'''%(layer_name, bottom, top, loc, height, width, scale)
    return kps_layer_str

def generate_concat_layer(bottoms):
    concat_layer_str = '''layer {
  name: "concat"
  type: "Concat"
  top: "feature"'''
    for i in range(len(bottoms)):
        concat_layer_str += '''
  bottom: "%s"'''%bottoms[i]
    concat_layer_str += '''
  concat_param {
    axis: 3
  }
}
'''
    return concat_layer_str

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

def get_layer_name_kps(loc, stage, block, layer, relu_sum):
    layer_name = 'kps_%d_conv%d_%d_%d'%(loc + 1, stage+1, block, layer)
    if relu_sum == 0:
      return layer_name
    elif relu_sum == 1:
      return layer_name+'_relu'
    elif relu_sum == 2:
      return layer_name[:-1]+'sum'

def generate_train_val_1st_stage(kernel_num, last_top, network_str, args):
    for b in xrange(1, args.block_number[0]+1):
        l = 1
        conv_layer_name = get_layer_name(1, b, l, 0)
        relu_layer_name = get_layer_name(1, b, l, 1)
        network_str += generate_conv_bn_scale_layer(3, kernel_num, 1, 1, conv_layer_name, last_top, conv_layer_name)
        network_str += generate_activation_layer(relu_layer_name, conv_layer_name, conv_layer_name, 'ReLU')

        l = 2
        conv_layer_name_bottom = conv_layer_name
        conv_layer_name = get_layer_name(1, b, l, 0)
        relu_layer_name = get_layer_name(1, b, l, 1)
        sum_layer_name = get_layer_name(1, b, l, 2)
        network_str += generate_conv_bn_scale_layer(3, kernel_num, 1, 1, conv_layer_name, conv_layer_name_bottom, conv_layer_name)

        network_str += generate_eltwise_layer(sum_layer_name, last_top, conv_layer_name, sum_layer_name, 'SUM')
        network_str += generate_activation_layer(relu_layer_name, sum_layer_name, sum_layer_name, 'ReLU')
        last_top = sum_layer_name
    return network_str, last_top

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

def generate_kps_1st_stage(stage, loc, kernel_num, last_top, network_str, args):

    b = 1

    l = 1
    conv_layer_name = get_layer_name_kps(loc, stage, b, l, 0)
    relu_layer_name = get_layer_name_kps(loc, stage, b, l, 1)
    network_str += generate_conv_bn_scale_layer(3, kernel_num, 1, 1, conv_layer_name, last_top, conv_layer_name)
    network_str += generate_activation_layer(relu_layer_name, conv_layer_name, conv_layer_name, 'ReLU')

    l = 2
    conv_layer_name_bottom = conv_layer_name
    conv_layer_name = get_layer_name_kps(loc, stage, b, l, 0)
    relu_layer_name = get_layer_name_kps(loc, stage, b, l, 1)
    network_str += generate_conv_bn_scale_layer(3, kernel_num, 1, 1, conv_layer_name, conv_layer_name_bottom, conv_layer_name)

    l = 3
    conv_layer_name_bottom = conv_layer_name
    conv_layer_name = get_layer_name_kps(loc, stage, b, l, 0)
    sum_layer_name = get_layer_name_kps(loc, stage, b, l, 2)
    network_str += generate_conv_layer(3, kernel_num, 1, 1, conv_layer_name, last_top, conv_layer_name)
    network_str += generate_eltwise_layer(sum_layer_name, conv_layer_name_bottom, conv_layer_name, sum_layer_name, 'SUM')
    network_str += generate_activation_layer(relu_layer_name, sum_layer_name, sum_layer_name, 'ReLU')
    last_top = sum_layer_name

    for b in xrange(2, args.block_number[stage-1]+1):
        l = 1
        conv_layer_name = get_layer_name_kps(loc, stage, b, l, 0)
        relu_layer_name = get_layer_name_kps(loc, stage, b, l, 1)
        network_str += generate_conv_bn_scale_layer(3, kernel_num, 1, 1, conv_layer_name, last_top, conv_layer_name)
        network_str += generate_activation_layer(relu_layer_name, conv_layer_name, conv_layer_name, 'ReLU')

        l = 2
        conv_layer_name_bottom = conv_layer_name
        conv_layer_name = get_layer_name_kps(loc, stage, b, l, 0)
        relu_layer_name = get_layer_name_kps(loc, stage, b, l, 1)
        sum_layer_name = get_layer_name_kps(loc, stage, b, l, 2)
        network_str += generate_conv_bn_scale_layer(3, kernel_num, 1, 1, conv_layer_name, conv_layer_name_bottom, conv_layer_name)

        network_str += generate_eltwise_layer(sum_layer_name, last_top, conv_layer_name, sum_layer_name, 'SUM')
        network_str += generate_activation_layer(relu_layer_name, sum_layer_name, sum_layer_name, 'ReLU')
        last_top = sum_layer_name
    return network_str, last_top

def generate_kps_stage(stage, loc, kernel_num, last_top, network_str, args):

    b = 1

    l = 1
    conv_layer_name = get_layer_name_kps(loc, stage, b, l, 0)
    relu_layer_name = get_layer_name_kps(loc, stage, b, l, 1)
    network_str += generate_conv_bn_scale_layer(3, kernel_num, 2, 1, conv_layer_name, last_top, conv_layer_name)
    network_str += generate_activation_layer(relu_layer_name, conv_layer_name, conv_layer_name, 'ReLU')

    l = 2
    conv_layer_name_bottom = conv_layer_name
    conv_layer_name = get_layer_name_kps(loc, stage, b, l, 0)
    relu_layer_name = get_layer_name_kps(loc, stage, b, l, 1)
    network_str += generate_conv_bn_scale_layer(3, kernel_num, 1, 1, conv_layer_name, conv_layer_name_bottom, conv_layer_name)

    l = 3
    conv_layer_name_bottom = conv_layer_name
    conv_layer_name = get_layer_name_kps(loc, stage, b, l, 0)
    sum_layer_name = get_layer_name_kps(loc, stage, b, l, 2)
    network_str += generate_conv_layer(1, kernel_num, 2, 0, conv_layer_name, last_top, conv_layer_name)
    network_str += generate_eltwise_layer(sum_layer_name, conv_layer_name_bottom, conv_layer_name, sum_layer_name, 'SUM')
    network_str += generate_activation_layer(relu_layer_name, sum_layer_name, sum_layer_name, 'ReLU')
    last_top = sum_layer_name

    for b in xrange(2, args.block_number[stage-1]+1):
        l = 1
        conv_layer_name = get_layer_name_kps(loc, stage, b, l, 0)
        relu_layer_name = get_layer_name_kps(loc, stage, b, l, 1)
        network_str += generate_conv_bn_scale_layer(3, kernel_num, 1, 1, conv_layer_name, last_top, conv_layer_name)
        network_str += generate_activation_layer(relu_layer_name, conv_layer_name, conv_layer_name, 'ReLU')

        l = 2
        conv_layer_name_bottom = conv_layer_name
        conv_layer_name = get_layer_name_kps(loc, stage, b, l, 0)
        relu_layer_name = get_layer_name_kps(loc, stage, b, l, 1)
        sum_layer_name = get_layer_name_kps(loc, stage, b, l, 2)
        network_str += generate_conv_bn_scale_layer(3, kernel_num, 1, 1, conv_layer_name, conv_layer_name_bottom, conv_layer_name)

        network_str += generate_eltwise_layer(sum_layer_name, last_top, conv_layer_name, sum_layer_name, 'SUM')
        network_str += generate_activation_layer(relu_layer_name, sum_layer_name, sum_layer_name, 'ReLU')
        last_top = sum_layer_name
    return network_str, last_top


def generate_train_val(num_classes, args, data_root, data_name, batch_size, pix, num_points, has_full, begin_stage):
    network_str = generate_data_layer(data_root, data_name, batch_size, pix)
    if num_points > 0:
        network_str += generate_data_kps_layer(data_root, data_name, batch_size, pix)
    last_top = 'data'

    '''before stage'''
    kernel_num = 20
    scale = float(1)
    network_str += generate_conv_bn_scale_layer(3, kernel_num, 2, 1, 'conv1', last_top, 'conv1')
    network_str += generate_activation_layer('conv1_relu', 'conv1', 'conv1', 'ReLU')
    last_top = 'conv1'

    last_tops = []
    scales = []
    kernel_nums = []

    '''stage 1'''
    network_str, last_top = generate_train_val_1st_stage(kernel_num, last_top, network_str, args)
    last_tops.append(last_top)
    scale /= 2
    scales.append(scale)
    kernel_nums.append(kernel_num)

    bottoms = []

    '''stage 2...'''
    if has_full:
        for s in range(1, len(args.block_number)):
            kernel_num *= 2
            kernel_nums.append(kernel_num)
            network_str, last_top = generate_train_val_stage(s + 1, kernel_num, last_top, network_str, args)
            last_tops.append(last_top)
            scale /= 2
            scales.append(scale)
        scale = scales[-1]
        kernel_size_fc = pix*scale
        if kernel_size_fc != int(kernel_size_fc):
            kernel_size_fc = kernel_size_fc + 1
        kernel_num_fc = kernel_nums[-1] * 2
        last_top = last_tops[-1]
        conv_full_name = 'fc_full'
        network_str += generate_conv_bn_scale_layer(kernel_size_fc, kernel_num_fc, 1, 0, conv_full_name, last_top, conv_full_name)
        network_str += generate_activation_layer('relu_full', conv_full_name, conv_full_name, 'ReLU')
        bottoms.append('fc_full')

        kps_last_top = last_tops[begin_stage - 1]
        kps_scale = scales[begin_stage - 1]
        kps_height = pix * kps_scale / 2
        kps_width = pix * kps_scale / 2
        for i in range(num_points):
            kps_layer_name = "kps%d"%(i+1)
            network_str += generate_kps_layer(i, kps_height, kps_width, kps_scale, kps_layer_name, kps_last_top, kps_layer_name)
            last_top = kps_layer_name
            kernel_num = kernel_nums[begin_stage]
            network_str, last_top = generate_kps_1st_stage(begin_stage + 1, i, kernel_num, last_top, network_str, args)
            for s in range(begin_stage + 1, len(args.block_number)):
                kernel_num *= 2
                network_str, last_top = generate_kps_stage(s + 1, i, kernel_num, last_top, network_str, args)
            conv_kps_name = 'fc_kps_%d'%(i+1)
            network_str += generate_conv_bn_scale_layer(kernel_size_fc, kernel_num_fc, 1, 0, conv_kps_name, last_top, conv_kps_name)
            network_str += generate_activation_layer('relu_kps_%d'%(i+1), conv_kps_name, conv_kps_name, 'ReLU')
            bottoms.append(conv_kps_name)
    else:
        for s in range(1, begin_stage):
            kernel_num *= 2
            kernel_nums.append(kernel_num)
            network_str, last_top = generate_train_val_stage(s + 1, kernel_num, last_top, network_str, args)
            last_tops.append(last_top)
            scale /= 2
            scales.append(scale)

        kps_last_top = last_top
        kps_scale = scale
        kps_height = pix * kps_scale / 2
        kps_width = pix * kps_scale / 2
        for i in range(num_points):
            kernel_num = kernel_nums[-1] * 2
            kps_layer_name = "kps%d"%(i+1)
            network_str += generate_kps_layer(i, kps_height, kps_width, kps_scale, kps_layer_name, kps_last_top, kps_layer_name)
            last_top = kps_layer_name
            network_str, last_top = generate_kps_1st_stage(begin_stage + 1, i, kernel_num, last_top, network_str, args)
            scale /= 2
            for s in range(begin_stage + 1, len(args.block_number)):
                kernel_num *= 2
                network_str, last_top = generate_kps_stage(s + 1, i, kernel_num, last_top, network_str, args)
                scale /= 2
            kernel_size_fc = pix * scale
            if kernel_size_fc != int(kernel_size_fc):
                kernel_size_fc = kernel_size_fc + 1
            kernel_num_fc = kernel_num * 2
            conv_kps_name = 'fc_kps_%d'%(i+1)
            network_str += generate_conv_bn_scale_layer(kernel_size_fc, kernel_num_fc, 1, 0, conv_kps_name, last_top, conv_kps_name)
            network_str += generate_activation_layer('relu_kps_%d'%(i+1), conv_kps_name, conv_kps_name, 'ReLU')
            bottoms.append(conv_kps_name)

    '''after stage'''
    network_str += generate_concat_layer(bottoms)
    # network_str += generate_dropout_layer('feature')
    network_str += generate_fc_layer(num_classes, 'fc', 'feature', 'fc', 'gaussian')
    network_str += generate_softmax_loss('fc')
    return network_str

def generate_solver(train_val, batch_size, num_epochs, num_imgs_train, num_imgs_val, pix, has_full, has_kps, begin_stage, args):
    test_iter = num_imgs_val//(batch_size*4)+1
    max_iter = num_epochs*(num_imgs_train)//(batch_size*4)+1
    stepsize = max_iter//4
    snap_shot = max_iter//2
    solver_str = '''net: "%s"
test_iter: %d
test_interval: 1000
test_initialization: true
display: 100
base_lr: 0.1
lr_policy: "step"
stepsize: %d
gamma: 0.1
max_iter: %d
momentum: 0.9
weight_decay: 0.0001
snapshot: %d
snapshot_prefix: "models/deepid/resnet/resnet_kps_p%dx%d_f%d_k%d_s%d_b'''%(train_val, test_iter, stepsize, max_iter, snap_shot, pix, pix, has_full, has_kps, begin_stage)
    for i in range(len(args.block_number)):
        solver_str += '%d'%args.block_number[i]
        if i < len(args.block_number) - 1:
            solver_str += 'x'
    solver_str += '''"
solver_mode: GPU'''
    return solver_str

def main():
    args = parse_args()
    data_root = 'data/Celeb'
    data_name = 'Celeb'
    pix = 56
    num_points = 5
    has_full = 0
    begin_stage = 3
    batch_size = 256
    num_imgs_train = 495008
    num_imgs_val = 61510
    num_epochs = 20
    num_classes = 1650
    train_val = 'models/deepid/resnet/train_val_kps.prototxt'
    solver = 'models/deepid/resnet/solver_kps.prototxt'
    solver_str = generate_solver(train_val, batch_size, num_epochs, num_imgs_train, num_imgs_val, pix, has_full, num_points>0, begin_stage, args)
    network_str = generate_train_val(num_classes, args, data_root, data_name, batch_size, pix, num_points, has_full, begin_stage - 1)
    fp = open(solver, 'w')
    fp.write(solver_str)
    fp.close()
    fp = open(train_val, 'w')
    fp.write(network_str)
    fp.close()

if __name__ == '__main__':
    main()
