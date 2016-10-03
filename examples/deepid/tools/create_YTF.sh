#!/usr/bin/env sh
# Create the CASIA lmdb inputs
# N.B. set the path to the CASIA train + val data dirs
set -e

# EXAMPLE=examples/deepid
DATA=data/YTF
TOOLS=build/tools

DATA_ROOT=data/YTF/
TRAIN_TXT_PATH=data/YTF/aux
VAL_TXT_PATH=data/YTF/aux

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true

if $RESIZE; then
  RESIZE_HEIGHT=56
  RESIZE_WIDTH=56
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_YTF.shto the path" \
       "where the YTF training data is stored."
  exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_YTF.sh to the path" \
       "where the YTF validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA_ROOT \
    $TRAIN_TXT_PATH/train.txt \
    $DATA/YTF_56x56_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA_ROOT \
    $VAL_TXT_PATH/val.txt \
    $DATA/YTF_56x56_val_lmdb

echo "Done."
