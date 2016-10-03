#!/usr/bin/env sh
# Create the CASIA train set mean binaryproto
set -e

DATA=data/YTF/YTF_56x56_train_lmdb
TOOLS=build/tools
MEANFILE=data/YTF/YTF_56x56_mean.binaryproto

echo "Computing train set mean..."

GLOG_logtostderr=1 $TOOLS/compute_image_mean \
    $DATA \
    $MEANFILE

echo "Done."
