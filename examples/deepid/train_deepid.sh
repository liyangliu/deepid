#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=models/deepid/deepid/deepid_solver.prototxt -gpu=all
