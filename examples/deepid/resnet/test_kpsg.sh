#!/usr/bin/env sh
set -e

./build/tools/caffe test --model=models/deepid/resnet/kpsg/train_val.prototxt \
    --weights=models/deepid/resnet/kpsg/resnet_p56x56_g5_d1_f0_k5_s3_b1x1x1x1/resnet_LFW_iter_35674.caffemodel \
    --iterations=2377 -gpu=all
