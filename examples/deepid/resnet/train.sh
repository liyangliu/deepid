#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=models/deepid/resnet/solver_kps.prototxt -gpu=all
