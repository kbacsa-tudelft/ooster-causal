#!/usr/bin/env bash
# Sampling-period sensitivity variant of run_limburg_flood_eval.sh: identical pipeline and
# hyperparameters, except build_dataset.sh is told to resample onto a 5-minute grid instead of
# auto-detecting the coarsest native cadence - mirrors run_zeeland_flood_eval_5min.sh, which found a
# real precision/F1 improvement (not just recall) from this on Zeeland once trained to convergence;
# this run answers whether that effect generalizes to a different region's station network, or was
# Zeeland-specific. Writes to its own DATA_DIR/RUN_DIR so the original 15min/auto Limburg run isn't
# touched and the two can be compared directly (AC@k, precision/recall, held_out/val_mse) the same
# way the Zeeland 15min/5min runs were.
#
# At 5min most Limburg stations - whose native cadence is 10-15min or coarser - will have 2-3x as
# many all-NaN grid steps as real readings; cuts_plus_rca.py's existing fill policy already handles
# that (same mechanism the 15min run relies on for its own missing data), so no code changes are
# needed, just a finer --freq. Expect ~3x more timesteps per session and correspondingly slower
# training/scoring than the 15min run - and, per the Zeeland result, more epochs may be needed to
# reach convergence, not just more time per epoch (the Zeeland 5min run needed ~100 epochs where the
# 15min run had already converged by ~36).
#
# Usage:
#   cuts_plus_prototype/run_limburg_flood_eval_5min.sh [total_epoch] [predict_chunk_size] [flood_min_alert]
#
#   total_epoch         default 50 (matches run_limburg_flood_eval.sh - raise this if held_out/val_mse
#                       hasn't plateaued yet, as happened on Zeeland)
#   predict_chunk_size  default 32 (lower this if validation/scoring runs out of memory - more likely
#                       here than at 15min, given the ~3x longer sessions)
#   flood_min_alert     default high (see run_limburg_flood_eval.sh)
set -euo pipefail

TOTAL_EPOCH="${1:-50}"
PREDICT_CHUNK_SIZE="${2:-32}"
FLOOD_MIN_ALERT="${3:-high}"

DATA_DIR=cuts_plus_prototype/scratch/limburg_flood_eval_dataset_5min
FLOOD_EVENTS_CSV=flood_events_output/municipality_high/events.csv
RESAMPLE_FREQ=5min
FILL_POLICY=rate_0.1_after_10
LAMBDA_S_START=0.3
LAMBDA_S_END=0.03
VAL_RATIO=0.1
TEST_RATIO=0.1
HELD_OUT_EVAL_EVERY=5

RUN_DIR=cuts_plus_prototype/scratch/limburg_flood_eval_5min

if [ ! -f "$FLOOD_EVENTS_CSV" ]; then
  echo "$FLOOD_EVENTS_CSV not found - build it first with:"
  echo "  python3 cuts_plus_prototype/prepare_rws_waterinfo.py --input-dir rws_waterinfo --output-dir /tmp/wl_adapted"
  echo "  python3 flood_events_municipality.py --data-dir /tmp/wl_adapted --out flood_events_output"
  exit 1
fi

echo "=== Building Limburg dataset at $RESAMPLE_FREQ -> $DATA_DIR ==="
cuts_plus_prototype/build_dataset.sh "$DATA_DIR" limburg rws_waterinfo "" "" "$RESAMPLE_FREQ"

echo "=== Training on $DATA_DIR ($TOTAL_EPOCH epoch(s), fill_policy=$FILL_POLICY, flood_min_alert=$FLOOD_MIN_ALERT) ==="
python3 cuts_plus_prototype/cuts_plus_rca.py \
  --data-dir "$DATA_DIR" \
  --total-epoch "$TOTAL_EPOCH" \
  --predict-chunk-size "$PREDICT_CHUNK_SIZE" \
  --lambda-s-start "$LAMBDA_S_START" \
  --lambda-s-end "$LAMBDA_S_END" \
  --fill-policy "$FILL_POLICY" \
  --flood-events-csv "$FLOOD_EVENTS_CSV" \
  --flood-min-alert "$FLOOD_MIN_ALERT" \
  --flood-aware-split true \
  --val-ratio "$VAL_RATIO" \
  --test-ratio "$TEST_RATIO" \
  --held-out-eval-every "$HELD_OUT_EVAL_EVERY" \
  --save-dir "$RUN_DIR/models" \
  --log-dir "$RUN_DIR/runs"

echo "Done. Compare against the 15min/auto run (run_limburg_flood_eval.sh -> "
echo "cuts_plus_prototype/scratch/limburg_flood_eval/) for the effect of sampling period: AC@k was"
echo "printed per flood score session above, and held_out/val_mse every $HELD_OUT_EVAL_EVERY epochs is"
echo "in TensorBoard:"
echo "  tensorboard --logdir $RUN_DIR/runs"
