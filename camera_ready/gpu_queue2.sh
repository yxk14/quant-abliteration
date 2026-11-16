#!/usr/bin/env bash
# Second queue: resume E4 (interleaves with E5 under the lock), then refresh E0 and the report.
cd "$(dirname "$0")/.."
PY=~/qa-env/bin/python; L=camera_ready/logs
run() { echo "=== $(date '+%F %T') $1" | tee -a $L/queue.log; flock /tmp/gpu.lock $PY "${@:2}" >> $L/$1.log 2>&1; echo "    exit $? $(date '+%T')" | tee -a $L/queue.log; }
run e4 camera_ready/e4/run_e4.py
$PY camera_ready/e0/run_e0.py >> $L/e0.log 2>&1
$PY camera_ready/lib/make_report.py >> $L/report.log 2>&1
echo "=== $(date '+%F %T') queue2 finished" | tee -a $L/queue.log
