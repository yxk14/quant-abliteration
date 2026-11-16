#!/usr/bin/env bash
# Re-run every camera-ready stage end to end. Resumes: every GPU stage skips
# configurations whose output already exists with the right row count, and the
# gated analyses run only when the owner's label files are present.
#   bash camera_ready/run_all.sh            # everything
#   bash camera_ready/run_all.sh cpu        # only the CPU stages (seconds)
set -u
cd "$(dirname "$0")/.."
PY=~/qa-env/bin/python
L=camera_ready/logs; mkdir -p $L
gpu() { echo "=== $(date '+%F %T') $1"; flock /tmp/gpu.lock $PY "${@:2}" 2>&1 | tee -a $L/$1.log | grep -v -i "warning\|Loading weights"; }
cpu() { echo "=== $(date '+%F %T') $*"; $PY "$@"; }

cpu camera_ready/lib/s1_check.py || { echo "S1 failed: stop"; exit 1; }
cpu camera_ready/e0/run_e0.py
cpu camera_ready/e3/run_e3_offline.py
cpu camera_ready/e2/make_packet.py
[ -f camera_ready/e1/gateB_packet_untranslated.json ] || cpu camera_ready/e1/make_packet.py
if [ "${1:-}" != "cpu" ]; then
  gpu e3_subclassifier camera_ready/e3/run_subclassifier.py
  gpu e1_translate     camera_ready/e1/translate.py
  gpu e4               camera_ready/e4/run_e4.py
  cpu camera_ready/e0/run_e0.py            # now with the candidate judges' C0 verdicts
  gpu e4_second_judge  camera_ready/e4/second_judge.py   # optional: only if E4 flags a candidate (kappa >= 0.6); ~90 min
  gpu e5_gen_abl       camera_ready/e5/run_generation.py --subject abl --stages 0 2 3
  gpu e5_gen_base      camera_ready/e5/run_generation.py --subject base --stages 0 2
  gpu e5_judge         camera_ready/e5/judge_gens.py
  gpu e5_e54           camera_ready/e5/run_e54.py
fi
[ -f camera_ready/e3/subclassifier_preds.json ] && cpu camera_ready/e3/finish_e3.py
cpu camera_ready/e3/gateA_analyze.py && [ -f camera_ready/e3/gateA_analysis.json ] && cpu camera_ready/e3/finish_e3.py
cpu camera_ready/e1/analyze.py
cpu camera_ready/e2/analyze.py
[ -d camera_ready/e4/second_judge ] && cpu camera_ready/e4/second_judge_analyze.py
[ -d camera_ready/e5/gen ] && cpu camera_ready/e5/analyze.py
echo "=== done $(date '+%F %T')"
