#!/usr/bin/env bash
# Fourth queue (2026-10-24 evening). Waits for gpu_queue.sh to print "queue finished"
# (E5 judge + E5.4), then: E5 analysis (CPU), the optional gemma-2-9b second-judge
# pass over the full corpus (E4, GPU, ~90 min), its analysis, and the report.
cd "$(dirname "$0")/.."
PY=~/qa-env/bin/python; L=camera_ready/logs
run() { echo "=== $(date '+%F %T') $1" | tee -a $L/queue.log; flock /tmp/gpu.lock $PY "${@:2}" >> $L/$1.log 2>&1; echo "    exit $? $(date '+%T')" | tee -a $L/queue.log; }
until grep -q "queue finished" $L/queue.log; do sleep 60; done
$PY camera_ready/e5/analyze.py >> $L/e5_analyze.log 2>&1
run e4_second_judge camera_ready/e4/second_judge.py
$PY camera_ready/e4/second_judge_analyze.py >> $L/e4_second_judge_analyze.log 2>&1
$PY camera_ready/lib/make_report.py >> $L/report.log 2>&1
echo "=== $(date '+%F %T') queue4 finished" | tee -a $L/queue.log
