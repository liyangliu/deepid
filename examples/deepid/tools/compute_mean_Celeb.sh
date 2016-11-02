#!/usr/bin/env sh
# Create the Celeb train set mean binaryproto
set -e
SIZEH=112
SIZEW=96
DATASET=CFW

DATA=data/${DATASET}/${DATASET}_${SIZEH}x${SIZEW}_train_lmdb
TOOLS=build/tools
MEANFILE=data/${DATASET}/${DATASET}_${SIZEH}x${SIZEW}_mean.binaryproto

echo "Computing train set mean..."

GLOG_logtostderr=1 $TOOLS/compute_image_mean \
    $DATA \
    $MEANFILE

echo "Done."
