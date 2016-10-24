#!/usr/bin/env sh
set -e

./build/tools/caffe time --model=models/deepid/resnet/kps/train_val.prototxt -gpu=all
