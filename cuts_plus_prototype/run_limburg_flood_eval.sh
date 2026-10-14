#!/usr/bin/env bash
# Trains CUTS+ on the Limburg/Maas corridor dataset with a flood-aware split, so flood detection can be
# verified against sessions that actually contain a real flood - rather than the chronological
# val_ratio/test_ratio split sweep_fill_policy.sh/run_rws_waterinfo_pipeline.sh use, where a flood could
# land in train, val, or score depending only on when it happened to occur.
#
# --flood-aware-split routes every session touching a --flood-min-alert+ national event window straight
# to score (test), instead of masking it out of train/val mid-session, then pads score with the most
# recent flood-free sessions up to the --test-ratio target (0.1 below) and splits the remaining
# flood-free sessions train/val by --val-ratio (0.1 below) - 0.8/0.1/0.1 overall, confirmed against this
# dataset's actual flood count (33 of 1330 Limburg sessions touch a medium+ window, well under the 133-
# session test target) to give an exact 1064/133/133 train/val/score split. Validation stays flood-free,
# so residual median/std/POT thresholds are calibrated on pure-normal data; score then contains both the
# flood window itself and the normal timesteps surrounding it in those same sessions, so true/false-
# positive flood detection (root_cause_analysis's AC@k, computed automatically below since
# --flood-events-csv is set) has both positive and negative samples to work with. See cuts_plus_rca.py's
# flood_aware_split config field.
#
# flood_events_output/events.csv + station_events.csv must already exist - build them once with:
#   python3 cuts_plus_prototype/prepare_rws_waterinfo.py --input-dir rws_waterinfo --output-dir /tmp/wl_adapted
#   python3 flood_events.py --data-dir /tmp/wl_adapted --out flood_events_output
# (national tables, not Limburg-specific - the same ones run_rws_waterinfo_pipeline.sh uses; a station's
# event still routes the whole dataset's session for that window, per flood_events.py's own convention
# that a flood is a regime shift felt by every channel, not just the station(s) that triggered it.
# flood_events.py needs an occluded directory, not the raw rws_waterinfo/ - this is a small, one-off,
# infrequently-rebuilt artifact, so a scratch adapted copy built just for this is fine; it doesn't need
# to be the Limburg dataset build below, which is water-level-only and region-filtered separately).
#
# --region limburg below is built on the fly by build_dataset.sh (water level only, matching the
# `1974`/`imputation`/sweep_fill_policy.sh runs this eval was built to extend).
#
# --fill-policy defaults to cuts_plus_rca.py's own default (rate_0.1_after_10, chosen from
# sweep_fill_policy.sh's results) but is passed explicitly here for clarity.
#
# --held-out-eval-every 5 logs held_out/val_mse to TensorBoard every 5 epochs - the genuinely held-out
# counterpart to the training-only S1/S2 loss curves, to check whether 50 epochs is actually the right
# stopping point (see cuts_plus_rca.py's held_out_eval_every config field) rather than assuming it.
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

DATA_DIR=cuts_plus_prototype/scratch/limburg_flood_eval_dataset
FLOOD_EVENTS_CSV=flood_events_output/events.csv
FILL_POLICY=rate_0.1_after_10
LAMBDA_S_START=0.3
LAMBDA_S_END=0.03
VAL_RATIO=0.1
TEST_RATIO=0.1
HELD_OUT_EVAL_EVERY=5

RUN_DIR=cuts_plus_prototype/scratch/limburg_flood_eval

if [ ! -f "$FLOOD_EVENTS_CSV" ]; then
  echo "$FLOOD_EVENTS_CSV not found - build it first with:"
  echo "  python3 cuts_plus_prototype/prepare_rws_waterinfo.py --input-dir rws_waterinfo --output-dir /tmp/wl_adapted"
  echo "  python3 flood_events.py --data-dir /tmp/wl_adapted --out flood_events_output"
  exit 1
fi

echo "=== Building Limburg dataset -> $DATA_DIR ==="
cuts_plus_prototype/build_dataset.sh "$DATA_DIR" limburg rws_waterinfo "" ""

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

echo "Done. Flood detection (AC@k) was printed per flood score session above, and logged to TensorBoard,"
echo "along with held_out/val_mse every $HELD_OUT_EVAL_EVERY epochs (check whether it's still improving"
echo "by epoch $TOTAL_EPOCH or has already peaked):"
echo "  tensorboard --logdir $RUN_DIR/runs"
