#!/usr/bin/env sh
set -e

./build/tools/caffe test --model=models/deepid/deepid/train_val.prototxt \
    --weights=models/deepid/deepid/deepid_LFW_iter_35674.caffemodel \
    --iterations=2377 -gpu=all
