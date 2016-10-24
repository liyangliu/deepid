#!/usr/bin/env sh
set -e

#LOG_FILE=models/deepid/resnet/result/solver.log
#TRAIN_LOSS=models/deepid/resnet/result/train_loss
#TEST_LOSS=models/deepid/resnet/result/test_loss
#TEST_ACC=models/deepid/resnet/result/test_acc

LOG_FILE=models/deepid/resnet/kpsg/result_cor/solver.log
TRAIN_LOSS=models/deepid/resnet/kpsg/result_cor/train_loss.txt
TEST_LOSS=models/deepid/resnet/kpsg/result_cor/test_loss.txt
TEST_ACC=models/deepid/resnet/kpsg/result_cor/test_acc.txt

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
