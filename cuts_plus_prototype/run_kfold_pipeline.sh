#!/usr/bin/env bash
# Runs a full k-fold cross-validation pipeline in one command: train -> aggregate -> map.
#
# Trains --n-folds independent models (sweep.py, sequential), aggregates their graphs into per-edge
# mean/std/recurrence (aggregate_kfold_graphs.py), then generates a hover-mode causal map from the
# aggregated mean graph (plot_causal_map.py). Since there's no ground truth graph, read the map
# alongside the aggregator's printed per-edge std/recurrence counts as the confidence signal - low
# std and high recurrence across folds means a real, robust edge, not a single run's noise.
#
# Safe to interrupt and rerun with the SAME arguments at any point during training - sweep.py skips
# folds that already finished and resumes an interrupted one from its own checkpoint automatically.
#
# Usage:
#   cuts_plus_prototype/run_kfold_pipeline.sh [data_dir] [n_folds] [total_epoch] \
#       [min_channel_availability] [batch_size] [predict_chunk_size] [locations_csv]
#
#   data_dir                  default rws_data_prepared_10min
#   n_folds                   default 5
#   total_epoch               default 2    (smoke test - raise this for a real run, e.g. 75)
#   min_channel_availability  default 0.10
#   batch_size                default 128
#   predict_chunk_size        default 32
#   locations_csv              default rws_data_adapted/locations.csv
set -euo pipefail

DATA_DIR="${1:-rws_data_prepared_10min}"
N_FOLDS="${2:-5}"
TOTAL_EPOCH="${3:-2}"
MIN_CHANNEL_AVAILABILITY="${4:-0.10}"
BATCH_SIZE="${5:-128}"
PREDICT_CHUNK_SIZE="${6:-32}"
LOCATIONS_CSV="${7:-rws_data_adapted/locations.csv}"

FOLD_LIST=$(seq -s, 0 $((N_FOLDS - 1)))
SWEEP_ROOT=cuts_plus_prototype/scratch/kfold_pipeline
AGG_DIR=cuts_plus_prototype/scratch/kfold_pipeline_agg

# --resume errors out if --sweep-root has no prior sweep to resume - only pass it once one exists,
# so this script works unmodified on both the first run and every rerun after that.
RESUME_FLAG=()
if [ -d "$SWEEP_ROOT" ] && [ -n "$(ls -A "$SWEEP_ROOT" 2>/dev/null)" ]; then
  RESUME_FLAG=(--resume)
fi

echo "=== 1/3: training $N_FOLDS folds (fold=$FOLD_LIST) on $DATA_DIR ==="
python3 cuts_plus_prototype/sweep.py \
  --sweep "fold=$FOLD_LIST" \
  --n-folds "$N_FOLDS" \
  --data-dir "$DATA_DIR" \
  --total-epoch "$TOTAL_EPOCH" \
  --min-channel-availability "$MIN_CHANNEL_AVAILABILITY" \
  --batch-size "$BATCH_SIZE" \
  --predict-chunk-size "$PREDICT_CHUNK_SIZE" \
  --sweep-root "$SWEEP_ROOT" \
  "${RESUME_FLAG[@]}"

# sweep.py timestamps a new subdirectory per fresh invocation, and --resume continues (does not
# duplicate) the most recent one - either way, the lexicographically-last subdirectory is this run's.
SWEEP_DIR=$(ls -d "$SWEEP_ROOT"/*/ | sort | tail -1)
SWEEP_DIR="${SWEEP_DIR%/}"

echo "=== 2/3: aggregating folds from $SWEEP_DIR ==="
python3 cuts_plus_prototype/aggregate_kfold_graphs.py \
  --sweep-dir "$SWEEP_DIR" \
  --output "$AGG_DIR"

echo "=== 3/3: generating causal map ==="
python3 cuts_plus_prototype/plot_causal_map.py \
  --graph "$AGG_DIR/kfold_mean_graph.npy" \
  --data-dir "$DATA_DIR" \
  --locations-csv "$LOCATIONS_CSV" \
  --output "$AGG_DIR/causal_map_kfold.html" \
  --mode hover --top-n 5

echo "Done. Aggregate stats in $AGG_DIR, map at $AGG_DIR/causal_map_kfold.html"
