#!/usr/bin/env bash
# Trains CUTS+ on the Zeeland coastal/estuary dataset with a flood-aware split, so flood detection can
# be verified against sessions that actually contain a real flood - rather than the chronological
# split convergence_sweep_zeeland (sweep_convergence.sh) used, where none of this dataset's known
# Zeeland flood events (all 1952-2013) ever land in the held-out score window (2017+).
#
# --flood-aware-split routes every session touching a --flood-min-alert+ municipality event window
# straight to score (test), instead of masking it out of train/val mid-session, then pads score with
# the most recent flood-free sessions up to the --test-ratio target (0.1 below) and splits the
# remaining flood-free sessions train/val by --val-ratio (0.1 below) - the exact resulting split
# depends on how many Zeeland sessions touch a --flood-min-alert+ window in the actual table in use,
# so check the printed "Sessions: N train, N val, N score" line rather than assuming a fixed count.
# Validation stays flood-free, so residual median/std/POT thresholds are calibrated on pure-normal
# data; score then contains both the flood window itself and the normal timesteps surrounding it in
# those same sessions, so true/false-positive flood detection (root_cause_analysis's AC@k, computed
# automatically below since --flood-events-csv is set) has both positive and negative samples to work
# with. See cuts_plus_rca.py's flood_aware_split config field, and run_limburg_flood_eval.sh for the
# Limburg counterpart this mirrors.
#
# Hyperparameters below match convergence_sweep_zeeland/sparsity_fine_08 - the best-performing config
# found by this project's 13-way convergence sweep on Zeeland (fixed lambda_s=0.08, no tau/sparsity
# anneal) - rather than run_limburg_flood_eval.sh's original annealed lambda_s=0.3->0.03, which is the
# schedule that motivated that sweep in the first place (see sweep_convergence.sh's own docstring).
# This script answers "does the best Zeeland config actually detect real floods", not "does the
# original default config".
#
# flood_events_output/municipality_high/events.csv + station_events.csv must already exist - build
# them once with:
#   python3 cuts_plus_prototype/prepare_rws_waterinfo.py --input-dir rws_waterinfo --output-dir /tmp/wl_adapted
#   python3 flood_events_municipality.py --data-dir /tmp/wl_adapted --out flood_events_output
# (national/municipality tables, not Zeeland-specific - the same ones run_rws_waterinfo_pipeline.sh and
# run_limburg_flood_eval.sh use; a station's event still routes the whole dataset's session for that
# window, per flood_events.py's own convention that a flood is a regime shift felt by every channel,
# not just the station(s) that triggered it. This is the municipality_high/ table (alert=="high" only)
# rather than flood_events.py's original 11-station events.csv - lower alert thresholds on the full
# municipality-level table cover an implausibly large fraction of the whole dataset, so only "high" is
# usable here.)
#
# --region zeeland below is built on the fly by build_dataset.sh (water level only, matching
# convergence_sweep_zeeland's own dataset build).
#
# --held-out-eval-every 2 matches convergence_sweep_zeeland/sparsity_fine_08's own training command.
#
# Usage:
#   cuts_plus_prototype/run_zeeland_flood_eval.sh [total_epoch] [predict_chunk_size] [flood_min_alert]
#
#   total_epoch         default 50 (matches convergence_sweep_zeeland)
#   predict_chunk_size  default 32 (lower this if validation/scoring runs out of memory)
#   flood_min_alert      default high (the municipality_high/ table only ever has "high" rows, so
#                        medium/low would trivially select everything anyway)
set -euo pipefail
cd "$(dirname "$0")/.."

TOTAL_EPOCH="${1:-50}"
PREDICT_CHUNK_SIZE="${2:-32}"
FLOOD_MIN_ALERT="${3:-high}"

DATA_DIR=cuts_plus_prototype/scratch/zeeland_flood_eval_dataset
FLOOD_EVENTS_CSV=flood_events_output/municipality_high/events.csv
FILL_POLICY=rate_0.1_after_10
LAMBDA_S_START=0.08
LAMBDA_S_END=0.08
END_TAU=1.0
VAL_RATIO=0.1
TEST_RATIO=0.1
HELD_OUT_EVAL_EVERY=2

RUN_DIR=zeeland_flood_eval

if [ ! -f "$FLOOD_EVENTS_CSV" ]; then
  echo "$FLOOD_EVENTS_CSV not found - build it first with:" >&2
  echo "  python3 cuts_plus_prototype/prepare_rws_waterinfo.py --input-dir rws_waterinfo --output-dir /tmp/wl_adapted" >&2
  echo "  python3 flood_events_municipality.py --data-dir /tmp/wl_adapted --out flood_events_output" >&2
  exit 1
fi

echo "=== Building Zeeland dataset -> $DATA_DIR ==="
cuts_plus_prototype/build_dataset.sh "$DATA_DIR" zeeland rws_waterinfo "" ""

echo "=== Training on $DATA_DIR ($TOTAL_EPOCH epoch(s), fill_policy=$FILL_POLICY, flood_min_alert=$FLOOD_MIN_ALERT) ==="
python3 cuts_plus_prototype/cuts_plus_rca.py \
  --data-dir "$DATA_DIR" \
  --total-epoch "$TOTAL_EPOCH" \
  --predict-chunk-size "$PREDICT_CHUNK_SIZE" \
  --lambda-s-start "$LAMBDA_S_START" \
  --lambda-s-end "$LAMBDA_S_END" \
  --end-tau "$END_TAU" \
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
