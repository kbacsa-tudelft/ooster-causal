#!/usr/bin/env bash
# Runs the rws_waterinfo pipeline end to end in one command: build the trainable dataset on the fly ->
# build the flood-events table -> train (with flood-event root-cause analysis) -> causal maps (outgoing
# hover, incoming hover, and static).
#
# Input is the output of download_rws_waterlevel.py (rws_waterinfo/). The raw download is never modified;
# everything derived from it (occlusion, resampling) is built into a scratch directory by
# build_dataset.sh and never kept as its own permanently-named directory.
#
# Usage:
#   cuts_plus_prototype/run_rws_waterinfo_pipeline.sh [total_epoch] [predict_chunk_size] [min_availability] [flood_min_alert]
#
#   total_epoch         default 2  (smoke test - raise this for a real run)
#   predict_chunk_size  default 32 (lower this if validation/scoring runs out of memory)
#   min_availability    default 0.10 (channels observed in fewer training rows than this are dropped at
#                        train time via cuts_plus_rca.py's --min-channel-availability)
#   flood_min_alert      default medium (low/medium/high - which flood_events.py alerts to use)
#
# Fixed settings (edit here):
#   LAMBDA_S_START/END  sparsity penalty on the graph, 3x the CUTS+ default (0.1 -> 0.01)
#   MIN_WEIGHT          maps leave out edges weaker than this (0.9)
#   HOVER_TOP_N         hover map shows this many outgoing edges per node (3)
set -euo pipefail

TOTAL_EPOCH="${1:-2}"
PREDICT_CHUNK_SIZE="${2:-32}"
MIN_AVAILABILITY="${3:-0.10}"
FLOOD_MIN_ALERT="${4:-medium}"

LAMBDA_S_START=0.3
LAMBDA_S_END=0.03
MIN_WEIGHT=0.9
HOVER_TOP_N=3

RAW_DIR=rws_waterinfo
RUN_DIR=cuts_plus_prototype/scratch/rws_waterinfo_pipeline
PREPARED_DIR="$RUN_DIR/dataset"

echo "=== 1/6: building the trainable dataset $RAW_DIR -> $PREPARED_DIR ==="
cuts_plus_prototype/build_dataset.sh "$PREPARED_DIR" "" "$RAW_DIR" "" ""

FLOOD_EVENTS_CSV=flood_events_output/events.csv
if [ ! -f "$FLOOD_EVENTS_CSV" ]; then
  echo "=== 2/6: building the flood events table -> $FLOOD_EVENTS_CSV ==="
  WL_ADAPTED="$RUN_DIR/wl_adapted_for_flood_events"
  python3 cuts_plus_prototype/prepare_rws_waterinfo.py --input-dir "$RAW_DIR" --output-dir "$WL_ADAPTED"
  python3 flood_events.py --data-dir "$WL_ADAPTED" --out flood_events_output
  rm -rf "$WL_ADAPTED"
else
  echo "=== 2/6: $FLOOD_EVENTS_CSV already exists, skipping ==="
fi

echo "=== 3/6: training ($TOTAL_EPOCH epoch(s), predict_chunk_size=$PREDICT_CHUNK_SIZE, lambda_s $LAMBDA_S_START -> $LAMBDA_S_END) ==="
echo "  flood root-cause analysis: $FLOOD_MIN_ALERT+ alert events, masked out of train/val only -"
echo "  score sessions keep real flood values so root-cause ranking can be evaluated against them"
python3 cuts_plus_prototype/cuts_plus_rca.py \
  --data-dir "$PREPARED_DIR" \
  --total-epoch "$TOTAL_EPOCH" \
  --predict-chunk-size "$PREDICT_CHUNK_SIZE" \
  --min-channel-availability "$MIN_AVAILABILITY" \
  --lambda-s-start "$LAMBDA_S_START" \
  --lambda-s-end "$LAMBDA_S_END" \
  --flood-events-csv "$FLOOD_EVENTS_CSV" \
  --flood-min-alert "$FLOOD_MIN_ALERT" \
  --save-dir "$RUN_DIR/models" \
  --log-dir "$RUN_DIR/runs"

echo "=== 4/6: outgoing hover causal map (top $HOVER_TOP_N per node, edges >= $MIN_WEIGHT) ==="
python3 cuts_plus_prototype/plot_causal_map.py \
  --graph "$RUN_DIR/models/cuts_plus_graph.npy" \
  --data-dir "$PREPARED_DIR" \
  --locations-csv "$RAW_DIR/locations.csv" \
  --output "$RUN_DIR/causal_map.html" \
  --mode hover --top-n "$HOVER_TOP_N" --min-weight "$MIN_WEIGHT"

echo "=== 5/6: incoming hover causal map (top $HOVER_TOP_N causes per node, edges >= $MIN_WEIGHT) ==="
python3 cuts_plus_prototype/plot_causal_map.py \
  --graph "$RUN_DIR/models/cuts_plus_graph.npy" \
  --data-dir "$PREPARED_DIR" \
  --locations-csv "$RAW_DIR/locations.csv" \
  --output "$RUN_DIR/causal_map_incoming.html" \
  --mode hover-incoming --top-n "$HOVER_TOP_N" --min-weight "$MIN_WEIGHT"

echo "=== 6/6: static causal map (strongest edges >= $MIN_WEIGHT) ==="
python3 cuts_plus_prototype/plot_causal_map.py \
  --graph "$RUN_DIR/models/cuts_plus_graph.npy" \
  --data-dir "$PREPARED_DIR" \
  --locations-csv "$RAW_DIR/locations.csv" \
  --output "$RUN_DIR/causal_map_static.html" \
  --mode top-k --min-weight "$MIN_WEIGHT"

echo "Done. Maps at $RUN_DIR/causal_map.html (outgoing hover), $RUN_DIR/causal_map_incoming.html (incoming hover), and $RUN_DIR/causal_map_static.html (static)"
