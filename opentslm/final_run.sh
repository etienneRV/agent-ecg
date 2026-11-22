#!/usr/bin/env bash
# Run ecg v2 (c2) or ecg-full v2 (a2) end to end on one GPU (GPU.md section 6): export, smoke test, train, score, compare.
#
#     tmux new -s c2 'opentslm/final_run.sh c2'      # and, on a second GPU: opentslm/final_run.sh a2
#
# Needs `uv sync`, `uv pip install -r opentslm/requirements-gpu.txt` and `agent-ecg prepare` first.
# The smoke test picks the largest batch that fits: the default 8 x 2, else 4 x 4 (the same effective batch),
# else 4 x 4 with gradient checkpointing; a run that still runs out of memory restarts with checkpointing.
# Everything lands in runs/<name>/ and runs/<name>.final.log; a finished run is never overwritten.
# DRY_RUN=1 prints the commands instead of running them.
set -uo pipefail
cd "$(dirname "$0")/.."
# A fresh tmux session may not have uv on its PATH yet (the installer puts it in ~/.local/bin).
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

RUN=${1:-}
case "$RUN" in
  c2)
    NAME=ecg-counts
    DATA=data/opentslm-none-counts
    EXPORT=(--context none --count-channels)
    TRAIN=(--tasks outcome)  # everything else as ecg
    ORIGINAL=results/opentslm/ecg-only/predictions.test.parquet
    BASELINE=results/baselines/signals.test.parquet
    ;;
  a2)
    NAME=full-multi-a2
    DATA=data/opentslm-a2
    EXPORT=(--head-thought-tokens 512 --thought-tokens 1536)
    TRAIN=(--epochs 1 --resolved-share 0.17 --lora-dropout 0.1 --eval-every 250 --val-checkpoints 4000)
    ORIGINAL=results/opentslm/full-multi/predictions.test.parquet
    BASELINE=results/baselines/issue_signals_thoughts.test.parquet
    ;;
  *)
    echo "usage: opentslm/final_run.sh c2|a2" >&2
    exit 2
    ;;
esac

R=runs/$NAME
LOG=runs/$NAME.final.log
mkdir -p runs
say() { echo "[$NAME $(date -u +%H:%M) UTC] $*" | tee -a "$LOG"; }
run() {  # a command, logged; DRY_RUN=1 only prints it
  say "\$ $*"
  [ "${DRY_RUN:-0}" = 1 ] && return 0
  "$@"
}
train() {  # the training command, logged in full; DRY_RUN=1 only prints it
  local cmd=(uv run --no-sync python opentslm/train.py --name "$NAME" --data "$DATA"
    --init-from OpenTSLM/llama-3.2-1b-ecg-sp "${TRAIN[@]}" "$@")
  say "\$ ${cmd[*]}"
  [ "${DRY_RUN:-0}" = 1 ] && return 0
  "${cmd[@]}"
}

if [ -e "$R/best_model.pt" ] && [ "${DRY_RUN:-0}" != 1 ]; then
  say "$R/best_model.pt exists: this run already trained; move it away to start again"
  exit 1
fi
[ -e data/checkpoints.parquet ] || { say "no data/checkpoints.parquet: run 'uv run --no-sync agent-ecg prepare' first"; exit 1; }
if [ "${DRY_RUN:-0}" != 1 ]; then  # fail now, not mid-smoke-test, if the GPU packages or the GPU are missing
  uv run --no-sync python -c "import torch, opentslm; assert torch.cuda.is_available(), 'no CUDA GPU visible'" \
    || { say "GPU setup incomplete: 'uv pip install -r opentslm/requirements-gpu.txt' on a machine with a CUDA GPU"; exit 1; }
fi

# 1. The export (skipped when it already exists with these settings).
if [ ! -e "$DATA/meta.json" ]; then
  run uv run --no-sync agent-ecg opentslm-export "${EXPORT[@]}" --out "$DATA" || { say "export failed"; exit 1; }
fi

# 2. Smoke tests: the largest batch whose peak memory leaves 20 % of the GPU free for the longest batches.
TOTAL_GB=$( (nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo 81920) \
  | head -1 | awk '{printf "%.0f", $1 / 1024}')
LIMIT=$(awk -v t="$TOTAL_GB" 'BEGIN {printf "%.0f", t * 0.8}')
FLAGS=(--batch-size 4 --grad-accum 4 --gradient-checkpointing)
for TRY in "" "--batch-size 4 --grad-accum 4"; do
  say "smoke test with ${TRY:-the default batch (8 x 2)}; limit ${LIMIT} of ${TOTAL_GB} GB"
  if [ "${DRY_RUN:-0}" = 1 ]; then
    train --smoke $TRY
    FLAGS=($TRY)
    break
  fi
  train --smoke $TRY 2>&1 | tee "runs/$NAME-smoke.log"
  status=${PIPESTATUS[0]}
  peak=$(tr '\r' '\n' < "runs/$NAME-smoke.log" | grep -o 'peak GPU memory [0-9.]*' | awk '{print $4}' | sort -n | tail -1)
  say "smoke exit $status, peak ${peak:-?} GB"
  if [ "$status" = 0 ] && awk -v p="${peak:-999}" -v l="$LIMIT" 'BEGIN {exit !(p <= l)}'; then
    FLAGS=($TRY)
    break
  fi
done

# 3. Training (a run that runs out of memory restarts once with gradient checkpointing).
if [ ${#FLAGS[@]} -eq 0 ]; then DESC="the default batch (8 x 2)"; else DESC="${FLAGS[*]}"; fi
say "training with $DESC"
if [ "${DRY_RUN:-0}" = 1 ]; then
  train ${FLAGS[@]+"${FLAGS[@]}"}
else
  train ${FLAGS[@]+"${FLAGS[@]}"} 2>&1 | tee "runs/$NAME.log"
  status=${PIPESTATUS[0]}
  if [ "$status" != 0 ] && tr '\r' '\n' < "runs/$NAME.log" | grep -qi 'out of memory'; then
    say "out of GPU memory: restarting with gradient checkpointing"
    train --overwrite --batch-size 4 --grad-accum 4 --gradient-checkpointing 2>&1 | tee -a "runs/$NAME.log"
    status=${PIPESTATUS[0]}
  fi
  [ "$status" = 0 ] || { say "training failed (exit $status); see runs/$NAME.log"; exit 1; }
fi

# 4. Score pass/fail on validation and test, calibrated on validation (GPU.md section 4).
run uv run --no-sync python opentslm/predict.py "$R" --split validation --tasks outcome || { say "validation predictions failed"; exit 1; }
run uv run --no-sync python opentslm/predict.py "$R" --split test --tasks outcome \
  --threshold-from "$R/predictions.validation.parquet" --calibrate-from "$R/predictions.validation.parquet" \
  || { say "test predictions failed"; exit 1; }
run uv run --no-sync agent-ecg evaluate "$R/predictions.test.parquet" --json "$R/report.test.json"

# 5. The comparisons the decision rule uses: against the original run, and against the benchmark.
for OTHER in "$ORIGINAL" "$BASELINE"; do
  OUT="$R/compare_vs_$(basename "$(dirname "$OTHER")")_$(basename "$OTHER" .test.parquet).test.txt"
  if [ "${DRY_RUN:-0}" = 1 ]; then
    run uv run --no-sync agent-ecg compare "$R/predictions.test.parquet" "$OTHER"
  else
    uv run --no-sync agent-ecg compare "$R/predictions.test.parquet" "$OTHER" | tee "$OUT" | tee -a "$LOG"
  fi
done
say "done: $R/report.test.json and $R/compare_vs_*.test.txt; judge with GPU.md section 6"
