#!/usr/bin/env sh
set -e

LOG_FILE=models/deepid/solver.log
TRAIN_LOSS=models/deepid/train_loss
TEST_LOSS=models/deepid/test_loss
TEST_ACC=models/deepid/test_acc

GLOG_logtostderr=1 cat \
    $LOG_FILE \
    | grep "Train net output" \
    | awk '{print $11}' \
    > $TRAIN_LOSS

GLOG_logtostderr=1 cat \
    $LOG_FILE \
    | grep "Test net output #1" \
    | awk '{print $11}' \
    > $TEST_LOSS

GLOG_logtostderr=1 cat \
    $LOG_FILE \
    | grep "Test net output #0" \
    | awk '{print $11}' \
    > $TEST_ACC

echo "Done."
