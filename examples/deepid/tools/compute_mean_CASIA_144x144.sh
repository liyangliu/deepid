#!/usr/bin/env sh
# Create the CASIA train set mean binaryproto
set -e

DATA=data/CASIA/224x224/CASIA_train_lmdb
TOOLS=build/tools
MEANFILE=data/CASIA/224x224/CASIA_mean.binaryproto

echo "Computing train set mean..."

GLOG_logtostderr=1 $TOOLS/compute_image_mean \
    $DATA \
    $MEANFILE

echo "Done."
