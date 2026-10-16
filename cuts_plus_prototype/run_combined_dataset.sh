#!/usr/bin/env bash
# Builds the combined rws water-level + discharge + KNMI rainfall dataset, ready for cuts_plus_rca.py,
# in one command.
#
# Stages:
#   1. KNMI hourly rainfall for 1950-2025        -> knmi_rain/           (skipped if already downloaded)
#   2. rws discharge (Q, m3/s) for 1950-2025     -> rws_discharge/       (skipped if already downloaded)
#   3. build the trainable dataset on the fly (occlude water level + discharge, combine with rain,
#      resample to a common grid) -> a scratch directory - build_dataset.sh chains
#      prepare_rws_waterinfo.py -> combine_rws_knmi.py -> prepare_data.py without keeping any of their
#      intermediate output as its own permanently-named directory (rain needs no occlusion - nothing
#      in this pipeline occludes it, only prefixes it at download time)
#   4. build flood_events_output/municipality_high/events.csv (flood_events_municipality.py, 408
#      open-water stations joined to their own municipality, filtered to alert=="high" - lower
#      thresholds cover an implausibly large fraction of the whole dataset, confirmed this session)
#      if not already there - used by cuts_plus_rca.py's --flood-events-csv (training step, not run
#      here) to mask flood windows out of train/val only and evaluate root-cause ranking against them
#      on held-out sessions
#
# No drop-sparse-channels stage: that's handled at train time by cuts_plus_rca.py's
# --min-channel-availability, computed from the actual training split.
#
# Needs rws_waterinfo/ (the output of run_rws_download.sh or download_rws_waterlevel.py) in the repository
# root. Python needs pandas, pyarrow and, for stage 1 only, hydropandas. If .venv-rws exists (created by
# run_rws_download.sh), its interpreter is used for the download stages only; install hydropandas into it
# with
#   .venv-rws/bin/pip install hydropandas
#
# Run from the repository root.
#
# Usage:
#   cuts_plus_prototype/run_combined_dataset.sh [min_availability] [flood_min_alert]
#
#   min_availability  default 0.10 (passed through to cuts_plus_rca.py's --min-channel-availability)
#   flood_min_alert    default medium (low/medium/high - passed through to the training step below)
#
# Training afterwards (not run here):
#   python3 cuts_plus_prototype/cuts_plus_rca.py --data-dir cuts_plus_prototype/scratch/combined_dataset \
#     --predict-chunk-size 32 --min-channel-availability <min_availability> \
#     --flood-events-csv flood_events_output/events.csv --flood-min-alert medium \
#     --save-dir <run>/models --log-dir <run>/runs
set -euo pipefail

MIN_AVAILABILITY="${1:-0.10}"
FLOOD_MIN_ALERT="${2:-high}"

if [ -x .venv-rws/bin/python ]; then PY=.venv-rws/bin/python; else PY=python3; fi

WL_RAW=rws_waterinfo
Q_RAW=rws_discharge
RAIN_RAW=knmi_rain
PREPARED=cuts_plus_prototype/scratch/combined_dataset

if [ ! -d "$RAIN_RAW/sessions" ] || [ -z "$(ls -A "$RAIN_RAW/sessions" 2>/dev/null)" ]; then
  echo "=== 1/4: downloading KNMI hourly rainfall -> $RAIN_RAW ==="
  "$PY" -u cuts_plus_prototype/download_knmi_rain.py --out "$RAIN_RAW" --start 1950-01-01 --end 2025-01-01
else
  echo "=== 1/4: KNMI rainfall already in $RAIN_RAW, skipping download ==="
fi

if [ ! -d "$Q_RAW/sessions" ] || [ -z "$(ls -A "$Q_RAW/sessions" 2>/dev/null)" ]; then
  echo "=== 2/4: downloading rws discharge -> $Q_RAW ==="
  "$PY" -u cuts_plus_prototype/download_rws_waterlevel.py --out "$Q_RAW" --grootheid Q --eenheid m3/s \
    --start 1950-01-01 --end 2025-01-01
else
  echo "=== 2/4: rws discharge already in $Q_RAW, skipping download ==="
fi

echo "=== 3/4: building the combined dataset -> $PREPARED ==="
cuts_plus_prototype/build_dataset.sh "$PREPARED" "" "$WL_RAW" "$Q_RAW" "$RAIN_RAW"

FLOOD_EVENTS_CSV=flood_events_output/municipality_high/events.csv
if [ ! -f "$FLOOD_EVENTS_CSV" ]; then
  echo "=== 4/4: building the municipality-level flood events table -> $FLOOD_EVENTS_CSV ==="
  WL_ADAPTED="$(mktemp -d)"
  trap 'rm -rf "$WL_ADAPTED"' EXIT
  "$PY" cuts_plus_prototype/prepare_rws_waterinfo.py --input-dir "$WL_RAW" --output-dir "$WL_ADAPTED"
  "$PY" flood_events_municipality.py --data-dir "$WL_ADAPTED" --out flood_events_output
else
  echo "=== 4/4: $FLOOD_EVENTS_CSV already exists, skipping ==="
fi

echo "Done. Combined dataset in $PREPARED"
echo "Train with: cuts_plus_prototype/cuts_plus_rca.py --data-dir $PREPARED --min-channel-availability $MIN_AVAILABILITY --flood-events-csv $FLOOD_EVENTS_CSV --flood-min-alert $FLOOD_MIN_ALERT ..."
