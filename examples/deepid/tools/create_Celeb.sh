#!/usr/bin/env sh
# Create the Celeb lmdb inputs
# N.B. set the path to the Celeb train + val data dirs
set -e

# EXAMPLE=examples/deepid
DATASET=CFW
DATA=data/${DATASET}
TOOLS=build/tools

# DATA_ROOT=data/${DATASET}/
DATA_ROOT=/
TRAIN_TXT_PATH=data/${DATASET}/train.txt
VAL_TXT_PATH=data/${DATASET}/val.txt

#DATASET=${DATASET}_1703

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
HEIGHT=112
WIDTH=96

if $RESIZE; then
  RESIZE_HEIGHT=$HEIGHT
  RESIZE_WIDTH=$WIDTH
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_Celeb to the path" \
       "where the Celeb training data is stored."
  exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_Celeb.sh to the path" \
       "where the Celeb validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA_ROOT \
    $TRAIN_TXT_PATH \
    $DATA/${DATASET}_${RESIZE_HEIGHT}x${RESIZE_WIDTH}_train_lmdb \
    # $DATA/${DATASET}_${RESIZE_HEIGHT}x${RESIZE_WIDTH}_train_kps_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA_ROOT \
    $VAL_TXT_PATH \
    $DATA/${DATASET}_${RESIZE_HEIGHT}x${RESIZE_WIDTH}_val_lmdb \
    # $DATA/${DATASET}_${RESIZE_HEIGHT}x${RESIZE_WIDTH}_val_kps_lmdb

echo "Done."
