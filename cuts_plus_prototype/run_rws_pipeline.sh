#!/usr/bin/env bash
# Runs the full rws_data pipeline end to end in one command: adapt -> prepare -> train -> map.
#
# Intended for a smoke test on a fresh machine (e.g. after git pull + scp'ing rws_data/ to a GPU
# machine) - default epoch count is small. Pass a larger --total-epoch for a real run once the
# smoke test confirms the pipeline runs cleanly.
#
# Usage:
#   cuts_plus_prototype/run_rws_pipeline.sh [total_epoch] [predict_chunk_size]
#
#   total_epoch         default 2  (smoke test - raise this for a real run, e.g. 50)
#   predict_chunk_size  default 512 (lower this if validation/scoring OOMs at this channel count -
#                        see predict_residuals in cuts_plus_rca.py)
set -euo pipefail
cd "$(dirname "$0")/.."

TOTAL_EPOCH="${1:-2}"
PREDICT_CHUNK_SIZE="${2:-512}"

RAW_DIR=rws_data
ADAPTED_DIR=rws_data_adapted
PREPARED_DIR=rws_data_prepared
RUN_DIR=cuts_plus_prototype/scratch/rws_pipeline

echo "=== 1/4: adapting $RAW_DIR -> $ADAPTED_DIR ==="
python3 cuts_plus_prototype/adapt_rws_data.py --input-dir "$RAW_DIR" --output-dir "$ADAPTED_DIR"

echo "=== 2/4: preparing $ADAPTED_DIR -> $PREPARED_DIR ==="
python3 cuts_plus_prototype/prepare_data.py --input-dir "$ADAPTED_DIR" --output-dir "$PREPARED_DIR"

echo "=== 3/4: training ($TOTAL_EPOCH epoch(s), predict_chunk_size=$PREDICT_CHUNK_SIZE) ==="
python3 cuts_plus_prototype/cuts_plus_rca.py \
  --data-dir "$PREPARED_DIR" \
  --total-epoch "$TOTAL_EPOCH" \
  --predict-chunk-size "$PREDICT_CHUNK_SIZE" \
  --save-dir "$RUN_DIR/models" \
  --log-dir "$RUN_DIR/runs"

echo "=== 4/4: generating causal map ==="
python3 cuts_plus_prototype/plot_causal_map.py \
  --graph "$RUN_DIR/models/cuts_plus_graph.npy" \
  --data-dir "$PREPARED_DIR" \
  --locations-csv "$ADAPTED_DIR/locations.csv" \
  --output "$RUN_DIR/causal_map.html" \
  --mode hover --top-n 5

echo "Done. Map at $RUN_DIR/causal_map.html"
