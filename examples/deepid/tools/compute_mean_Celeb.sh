#!/usr/bin/env sh
# Create the Celeb train set mean binaryproto
set -e
SIZE=56

DATA=data/Celeb/Celeb_${SIZE}x${SIZE}_train_lmdb
TOOLS=build/tools
MEANFILE=data/Celeb/Celeb_${SIZE}x${SIZE}_mean.binaryproto

echo "Computing train set mean..."

GLOG_logtostderr=1 $TOOLS/compute_image_mean \
    $DATA \
    $MEANFILE

echo "Done."
