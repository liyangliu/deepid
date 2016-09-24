#!/usr/bin/env sh
set -e

LOG_FILE=models/deepid/result/solver.log
BLOB=models/deepid/result/blob
SHAPE=models/deepid/result/shape

GLOG_logtostderr=1 cat \
    $LOG_FILE \
    | grep " -> " \
    | awk '{print $5 $6 $7}' \
    > $BLOB
GLOG_logtostderr=1 cat \
    $LOG_FILE \
    | grep "Top shape: " \
    | awk '{print $7, $8, $9, $10}' \
    > $SHAPE
paste $BLOB $SHAPE

echo "Done."
