#!/usr/bin/env sh
# Create the CASIA lmdb inputs
# N.B. set the path to the CASIA train + val data dirs
set -e

# EXAMPLE=examples/deepid
DATA=data/CASIA/144x144
TOOLS=build/tools

DATA_ROOT=/home/lly/grq/deepface/CASIA_CROP_JD_144x144/
TRAIN_TXT_PATH=/home/lly/grq/deepface
VAL_TXT_PATH=/home/lly/grq/deepface

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=false

if $RESIZE; then
  RESIZE_HEIGHT=55
  RESIZE_WIDTH=55
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_CASIA.sh to the path" \
       "where the CASIA training data is stored."
  exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_CASIA.sh to the path" \
       "where the CASIA validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA_ROOT \
    $TRAIN_TXT_PATH/train.txt \
    $DATA/CASIA_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA_ROOT \
    $VAL_TXT_PATH/val.txt \
    $DATA/CASIA_val_lmdb

echo "Done."
