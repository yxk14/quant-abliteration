#!/usr/bin/env bash
# Serial GPU queue for the camera-ready stages. Each stage takes the shared GPU
# lock on its own so other sessions can interleave. Every stage is resumable.
cd "$(dirname "$0")/.."
PY=~/qa-env/bin/python
L=camera_ready/logs
run() { echo "=== $(date '+%F %T') $1" | tee -a $L/queue.log; flock /tmp/gpu.lock $PY "${@:2}" >> $L/$1.log 2>&1; echo "    exit $? $(date '+%T')" | tee -a $L/queue.log; }
run e1_translate   camera_ready/e1/translate.py
run e4             camera_ready/e4/run_e4.py
run e5_gen_abl     camera_ready/e5/run_generation.py --subject abl --stages 0 2 3
run e5_gen_base    camera_ready/e5/run_generation.py --subject base --stages 0 2
run e5_judge       camera_ready/e5/judge_gens.py
run e5_e54         camera_ready/e5/run_e54.py
echo "=== $(date '+%F %T') queue finished" | tee -a $L/queue.log
