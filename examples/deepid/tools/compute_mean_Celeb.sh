#!/usr/bin/env sh
# Create the Celeb train set mean binaryproto
set -e
SIZE=56
DATASET=LFW

DATA=data/${DATASET}/${DATASET}_${SIZE}x${SIZE}_rep_train_lmdb
TOOLS=build/tools
MEANFILE=data/${DATASET}/${DATASET}_${SIZE}x${SIZE}_rep_mean.binaryproto

echo "Computing train set mean..."

GLOG_logtostderr=1 $TOOLS/compute_image_mean \
    $DATA \
    $MEANFILE

echo "Done."
