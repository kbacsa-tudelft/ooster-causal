#!/usr/bin/env bash
# Builds the combined rws water-level + discharge + KNMI rainfall dataset, ready for cuts_plus_rca.py,
# in one command.
#
# Stages:
#   1. KNMI hourly rainfall for 2005-2025        -> knmi_rain/           (skipped if already downloaded)
#   2. rws discharge (Q, m3/s) for 2005-2025     -> rws_discharge/       (skipped if already downloaded)
#   3. occlude implausible water levels          -> rws_waterinfo_adapted/
#   4. occlude implausible discharge              -> rws_discharge_adapted/
#   5. combine water levels + discharge + rain    -> combined_adapted/   (rain spread evenly over 10-min
#                                                                          steps; discharge needs no such
#                                                                          spreading, same cadence as WL)
#   6. resample to a common 10-min grid           -> combined_prepared/
#   7. drop channels below 10% availability        (applies to water level, discharge and rain alike)
#
# Needs rws_waterinfo/ (the output of run_rws_download.sh or download_rws_waterlevel.py) in the repository
# root. Python needs pandas, pyarrow and, for stage 1 only, hydropandas. If .venv-rws exists (created by
# run_rws_download.sh), its interpreter is used; install hydropandas into it with
#   .venv-rws/bin/pip install hydropandas
#
# Usage:
#   cuts_plus_prototype/run_combined_dataset.sh [min_availability]
#
#   min_availability  default 0.10
#
# Training afterwards (not run here):
#   python3 cuts_plus_prototype/cuts_plus_rca.py --data-dir combined_prepared --predict-chunk-size 32 \
#     --save-dir <run>/models --log-dir <run>/runs
set -euo pipefail
cd "$(dirname "$0")/.."

MIN_AVAILABILITY="${1:-0.10}"

if [ -x .venv-rws/bin/python ]; then PY=.venv-rws/bin/python; else PY=python3; fi

WL_RAW=rws_waterinfo
Q_RAW=rws_discharge
RAIN_RAW=knmi_rain
WL_ADAPTED=rws_waterinfo_adapted
Q_ADAPTED=rws_discharge_adapted
COMBINED=combined_adapted
PREPARED=combined_prepared

# Discharge plausibility window (m3/s): generous on both sides - the Rhine's recorded maximum at
# Lobith is around 12,600 m3/s, and small sluices can show brief reverse (negative) flow.
Q_MIN_VALUE=-500
Q_MAX_VALUE=20000

if [ ! -d "$RAIN_RAW/sessions" ] || [ -z "$(ls -A "$RAIN_RAW/sessions" 2>/dev/null)" ]; then
  echo "=== 1/7: downloading KNMI hourly rainfall -> $RAIN_RAW ==="
  "$PY" -u cuts_plus_prototype/download_knmi_rain.py --out "$RAIN_RAW" --start 2005-01-01 --end 2025-01-01
else
  echo "=== 1/7: KNMI rainfall already in $RAIN_RAW, skipping download ==="
fi

if [ ! -d "$Q_RAW/sessions" ] || [ -z "$(ls -A "$Q_RAW/sessions" 2>/dev/null)" ]; then
  echo "=== 2/7: downloading rws discharge -> $Q_RAW ==="
  "$PY" -u cuts_plus_prototype/download_rws_waterlevel.py --out "$Q_RAW" --grootheid Q --eenheid m3/s \
    --start 2005-01-01 --end 2025-01-01
else
  echo "=== 2/7: rws discharge already in $Q_RAW, skipping download ==="
fi

echo "=== 3/7: occluding implausible water levels $WL_RAW -> $WL_ADAPTED ==="
"$PY" cuts_plus_prototype/prepare_rws_waterinfo.py --input-dir "$WL_RAW" --output-dir "$WL_ADAPTED"

echo "=== 4/7: occluding implausible discharge $Q_RAW -> $Q_ADAPTED ==="
"$PY" cuts_plus_prototype/prepare_rws_waterinfo.py --input-dir "$Q_RAW" --output-dir "$Q_ADAPTED" \
  --prefix Q_ --min-value "$Q_MIN_VALUE" --max-value "$Q_MAX_VALUE"

echo "=== 5/7: combining water levels + discharge + rainfall -> $COMBINED ==="
"$PY" cuts_plus_prototype/combine_rws_knmi.py \
  --wl-dir "$WL_ADAPTED" --rain-dir "$RAIN_RAW/sessions" \
  --wl-locations "$WL_ADAPTED/locations.csv" --rain-locations "$RAIN_RAW/locations.csv" \
  --discharge-dir "$Q_ADAPTED" --discharge-locations "$Q_ADAPTED/locations.csv" \
  --output "$COMBINED"

echo "=== 6/7: resampling $COMBINED -> $PREPARED ==="
"$PY" cuts_plus_prototype/prepare_data.py --input-dir "$COMBINED" --output-dir "$PREPARED"

echo "=== 7/7: dropping channels below $MIN_AVAILABILITY availability ==="
"$PY" cuts_plus_prototype/drop_sparse_channels.py --data-dir "$PREPARED" --min-availability "$MIN_AVAILABILITY"

echo "Done. Combined dataset in $PREPARED (locations in $COMBINED/locations.csv)"
