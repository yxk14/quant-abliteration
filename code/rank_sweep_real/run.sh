#!/usr/bin/env bash
# Real-direction rank sweep for the ARTMAN '26 camera-ready.
# Replaces the synthetic Table 8 with a measurement on real weights.
#
#   ./run.sh dry                      # CPU only, no model, ~1 min. Do this first.
#   ./run.sh real hf                     # the actual 5090 run, ~20 min
#                                        # (hf = AdvBench+HarmBench from your HF cache;
#                                        #  or give a local prompt file instead)
#
# The GPU is shared with another Claude session. GPU stages take
# /tmp/gpu.lock via flock and wait their turn; CPU stages run regardless.
# Recommended way to launch the real run so it survives a closed terminal:
#   nohup ./run.sh real hf > logs/run.out 2>&1 &
#   tail -f logs/run.out
#
# Everything lands in out/. Nothing here generates text or calls a judge model.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TAG="${TAG:-qwen7b}"
NPROMPT="${NPROMPT:-512}"
KS="${KS:-1,2,4,8,16,32,64,128,256}"
SCHEMES="${SCHEMES:-int4,nf4,fp8,int8}"
LAYERS="${LAYERS:-}"          # empty = every layer
MODE="${1:-dry}"

mkdir -p out logs data

run() { echo; echo "=== $* ==="; "$@"; }

# GPU sharing rule: another Claude session shares this RTX 5090. Every GPU
# stage goes through a lock so jobs take turns instead of crashing. CPU stages
# never take the lock. If the lock is held, we wait (and say so).
GPU_LOCK="${GPU_LOCK:-/tmp/gpu.lock}"
gpu() {
  echo; echo "=== [gpu, lock $GPU_LOCK] $* ==="
  if ! flock -n "$GPU_LOCK" true 2>/dev/null; then
    echo "    lock held by another GPU job -- waiting for it to finish..."
  fi
  flock "$GPU_LOCK" "$@"
}

case "$MODE" in
dry)
  run python3 src/selftest.py
  run python3 src/mock_sweep.py --out out/mock_sweep.csv
  run python3 src/make_outputs.py --csv out/mock_sweep.csv --tag MOCK --outdir out
  echo
  echo "Dry run OK. out/*_MOCK.* are SYNTHETIC placeholders -- do not put them"
  echo "in the paper. Now run:   ./run.sh real hf"
  ;;

real)
  HARMFUL="${2:-}"
  if [ -z "$HARMFUL" ]; then
    echo "usage: ./run.sh real hf                     (AdvBench+HarmBench from HF cache)" >&2
    echo "       ./run.sh real /path/to/prompts.{txt,csv,jsonl}" >&2
    exit 1
  fi
  if [ "$HARMFUL" != "hf" ] && [ ! -f "$HARMFUL" ]; then
    echo "ERROR: prompt file not found: $HARMFUL" >&2
    echo "       use 'hf' to read AdvBench+HarmBench from your HF cache" >&2
    exit 1
  fi
  echo "=== nvidia-smi (what is on the GPU right now; we never kill anything) ==="
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv 2>/dev/null || true
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true
  python3 - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    sys.exit("ERROR: no CUDA device visible. Run this on the 5090, not over SSH "
             "to a CPU box.")
print(f"GPU: {torch.cuda.get_device_name(0)}  "
      f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
PY

  run python3 src/selftest.py
  run python3 src/prep_data.py --from "$HARMFUL" --n "$NPROMPT" --outdir data

  gpu python3 src/extract_directions.py \
      --model "$MODEL" \
      --harmful  data/harmful.txt \
      --harmless data/harmless.txt \
      --n-prompts "$NPROMPT" \
      ${LAYERS:+--layers "$LAYERS"} \
      --out "out/directions_${TAG}.npz" 2>&1 | tee "logs/extract_${TAG}.log"

  gpu python3 src/rank_sweep.py \
      --model "$MODEL" \
      --dirs "out/directions_${TAG}.npz" \
      --ks "$KS" --schemes "$SCHEMES" \
      ${LAYERS:+--layers "$LAYERS"} \
      --out "out/rank_sweep_${TAG}.csv" 2>&1 | tee "logs/sweep_${TAG}.log"

  # control arm: alpha=1 at every rank. SNR should be FLAT in k.
  gpu python3 src/rank_sweep.py \
      --model "$MODEL" \
      --dirs "out/directions_${TAG}.npz" \
      --ks "$KS" --schemes int4 --no-norm-match \
      ${LAYERS:+--layers "$LAYERS"} \
      --out "out/rank_sweep_${TAG}_control.csv" 2>&1 | tee "logs/control_${TAG}.log"

  run python3 src/make_outputs.py \
      --csv "out/rank_sweep_${TAG}.csv" --tag "$TAG" --outdir out

  echo
  echo "Done. Send back:"
  echo "   out/rank_sweep_${TAG}.csv"
  echo "   out/rank_sweep_${TAG}_control.csv"
  echo "   out/directions_${TAG}_meta.json"
  echo "   out/summary_${TAG}.txt"
  echo "(the .npz is large and not needed)"
  ;;

*)
  echo "usage: ./run.sh [dry|real] [harmful_prompts_file]" >&2; exit 1 ;;
esac
