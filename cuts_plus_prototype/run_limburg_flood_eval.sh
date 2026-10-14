#!/usr/bin/env bash
# Trains CUTS+ on the Limburg/Maas corridor dataset with a flood-aware split, so flood detection can be
# verified against sessions that actually contain a real flood - rather than the chronological
# val_ratio/test_ratio split sweep_fill_policy.sh/run_rws_waterinfo_pipeline.sh use, where a flood could
# land in train, val, or score depending only on when it happened to occur.
#
# --flood-aware-split routes every session touching a --flood-min-alert+ national event window straight
# to score (test), instead of masking it out of train/val mid-session; the remaining flood-free sessions
# split train/val by --val-ratio (0.2 below -> 80/20). Validation stays flood-free, so residual median/
# std/POT thresholds are calibrated on pure-normal data; score then contains both the flood window itself
# and the normal timesteps surrounding it in those same sessions, so true/false-positive flood detection
# (root_cause_analysis's AC@k, computed automatically below since --flood-events-csv is set) has both
# positive and negative samples to work with. See cuts_plus_rca.py's flood_aware_split config field.
#
# flood_events_output/events.csv + station_events.csv must already exist - build them once with:
#   python3 flood_events.py --data-dir rws_waterinfo_adapted_full --out flood_events_output
# (national tables, not Limburg-specific - the same ones run_rws_waterinfo_pipeline.sh uses; a station's
# event still routes the whole dataset's session for that window, per flood_events.py's own convention
# that a flood is a regime shift felt by every channel, not just the station(s) that triggered it).
#
# --fill-policy defaults to cuts_plus_rca.py's own default (rate_0.1_after_10, chosen from
# sweep_fill_policy.sh's results) but is passed explicitly here for clarity.
#
# Usage:
#   cuts_plus_prototype/run_limburg_flood_eval.sh [total_epoch] [predict_chunk_size] [flood_min_alert]
#
#   total_epoch         default 50 (matches the fill_policy sweep runs)
#   predict_chunk_size  default 32 (lower this if validation/scoring runs out of memory)
#   flood_min_alert      default medium (low/medium/high - which flood_events.py alerts route to score)
set -euo pipefail

TOTAL_EPOCH="${1:-50}"
PREDICT_CHUNK_SIZE="${2:-32}"
FLOOD_MIN_ALERT="${3:-medium}"

DATA_DIR=rws_waterinfo_limburg
FLOOD_EVENTS_CSV=flood_events_output/events.csv
FILL_POLICY=rate_0.1_after_10
LAMBDA_S_START=0.3
LAMBDA_S_END=0.03
VAL_RATIO=0.2

RUN_DIR=cuts_plus_prototype/scratch/limburg_flood_eval

if [ ! -f "$FLOOD_EVENTS_CSV" ]; then
  echo "$FLOOD_EVENTS_CSV not found - build it first with:"
  echo "  python3 flood_events.py --data-dir rws_waterinfo_adapted_full --out flood_events_output"
  exit 1
fi

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
  --save-dir "$RUN_DIR/models" \
  --log-dir "$RUN_DIR/runs"

echo "Done. Flood detection (AC@k) was printed per flood score session above, and logged to TensorBoard:"
echo "  tensorboard --logdir $RUN_DIR/runs"
