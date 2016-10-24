#!/usr/bin/env sh
set -e

./build/tools/caffe time --model=models/deepid/resnet/kpsg/train_val.prototxt -gpu=all
