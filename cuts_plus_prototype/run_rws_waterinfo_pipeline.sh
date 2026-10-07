#!/usr/bin/env bash
# Runs the rws_waterinfo pipeline end to end in one command: occlude implausible values -> resample ->
# drop sparse channels -> train -> causal maps (outgoing hover, incoming hover, and static).
#
# Input is the output of download_rws_waterlevel.py (rws_waterinfo/). The raw download is never modified.
#
# Usage:
#   cuts_plus_prototype/run_rws_waterinfo_pipeline.sh [total_epoch] [predict_chunk_size] [min_availability]
#
#   total_epoch         default 2  (smoke test - raise this for a real run)
#   predict_chunk_size  default 32 (lower this if validation/scoring runs out of memory)
#   min_availability    default 0.10 (channels observed in fewer rows than this are removed)
#
# Fixed settings (edit here):
#   LAMBDA_S_START/END  sparsity penalty on the graph, 3x the CUTS+ default (0.1 -> 0.01)
#   MIN_WEIGHT          maps leave out edges weaker than this (0.9)
#   HOVER_TOP_N         hover map shows this many outgoing edges per node (3)
set -euo pipefail
cd "$(dirname "$0")/.."

TOTAL_EPOCH="${1:-2}"
PREDICT_CHUNK_SIZE="${2:-32}"
MIN_AVAILABILITY="${3:-0.10}"

LAMBDA_S_START=0.3
LAMBDA_S_END=0.03
MIN_WEIGHT=0.9
HOVER_TOP_N=3

RAW_DIR=rws_waterinfo
ADAPTED_DIR=rws_waterinfo_adapted
PREPARED_DIR=rws_waterinfo_prepared
RUN_DIR=cuts_plus_prototype/scratch/rws_waterinfo_pipeline

echo "=== 1/6: occluding implausible values $RAW_DIR -> $ADAPTED_DIR ==="
python3 cuts_plus_prototype/prepare_rws_waterinfo.py --input-dir "$RAW_DIR" --output-dir "$ADAPTED_DIR"

echo "=== 2/6: resampling $ADAPTED_DIR -> $PREPARED_DIR ==="
python3 cuts_plus_prototype/prepare_data.py --input-dir "$ADAPTED_DIR" --output-dir "$PREPARED_DIR"

echo "=== 3/6: dropping channels below $MIN_AVAILABILITY availability ==="
python3 cuts_plus_prototype/drop_sparse_channels.py --data-dir "$PREPARED_DIR" --min-availability "$MIN_AVAILABILITY"

echo "=== 4/6: training ($TOTAL_EPOCH epoch(s), predict_chunk_size=$PREDICT_CHUNK_SIZE, lambda_s $LAMBDA_S_START -> $LAMBDA_S_END) ==="
python3 cuts_plus_prototype/cuts_plus_rca.py \
  --data-dir "$PREPARED_DIR" \
  --total-epoch "$TOTAL_EPOCH" \
  --predict-chunk-size "$PREDICT_CHUNK_SIZE" \
  --lambda-s-start "$LAMBDA_S_START" \
  --lambda-s-end "$LAMBDA_S_END" \
  --save-dir "$RUN_DIR/models" \
  --log-dir "$RUN_DIR/runs"

echo "=== 5/7: outgoing hover causal map (top $HOVER_TOP_N per node, edges >= $MIN_WEIGHT) ==="
python3 cuts_plus_prototype/plot_causal_map.py \
  --graph "$RUN_DIR/models/cuts_plus_graph.npy" \
  --data-dir "$PREPARED_DIR" \
  --locations-csv "$ADAPTED_DIR/locations.csv" \
  --output "$RUN_DIR/causal_map.html" \
  --mode hover --top-n "$HOVER_TOP_N" --min-weight "$MIN_WEIGHT"

echo "=== 6/7: incoming hover causal map (top $HOVER_TOP_N causes per node, edges >= $MIN_WEIGHT) ==="
python3 cuts_plus_prototype/plot_causal_map.py \
  --graph "$RUN_DIR/models/cuts_plus_graph.npy" \
  --data-dir "$PREPARED_DIR" \
  --locations-csv "$ADAPTED_DIR/locations.csv" \
  --output "$RUN_DIR/causal_map_incoming.html" \
  --mode hover-incoming --top-n "$HOVER_TOP_N" --min-weight "$MIN_WEIGHT"

echo "=== 7/7: static causal map (strongest edges >= $MIN_WEIGHT) ==="
python3 cuts_plus_prototype/plot_causal_map.py \
  --graph "$RUN_DIR/models/cuts_plus_graph.npy" \
  --data-dir "$PREPARED_DIR" \
  --locations-csv "$ADAPTED_DIR/locations.csv" \
  --output "$RUN_DIR/causal_map_static.html" \
  --mode top-k --min-weight "$MIN_WEIGHT"

echo "Done. Maps at $RUN_DIR/causal_map.html (outgoing hover), $RUN_DIR/causal_map_incoming.html (incoming hover), and $RUN_DIR/causal_map_static.html (static)"
