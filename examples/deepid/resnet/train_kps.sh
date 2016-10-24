#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=models/deepid/resnet/kps/solver.prototxt -gpu=all
