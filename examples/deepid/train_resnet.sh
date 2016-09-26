#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=models/deepid/resnet_solver.prototxt -gpu=all
