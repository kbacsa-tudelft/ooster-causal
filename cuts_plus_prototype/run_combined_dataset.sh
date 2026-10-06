#!/usr/bin/env bash
# Builds the combined rws water-level + KNMI rainfall dataset, ready for cuts_plus_rca.py, in one command.
#
# Stages:
#   0. KNMI hourly rainfall for 2005-2025 -> knmi_rain/   (skipped if already downloaded)
#   1. occlude implausible water levels   -> rws_waterinfo_adapted/
#   2. combine water levels + rainfall    -> combined_adapted/   (rain spread evenly over 10-min steps)
#   3. resample to a common 10-min grid   -> combined_prepared/
#   4. drop channels below 10% availability
#
# Needs rws_waterinfo/ (the output of run_rws_download.sh or download_rws_waterlevel.py) in the repository
# root. Python needs pandas, pyarrow and, for stage 0 only, hydropandas. If .venv-rws exists (created by
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

RWS_RAW=rws_waterinfo
RAIN_RAW=knmi_rain
ADAPTED=rws_waterinfo_adapted
COMBINED=combined_adapted
PREPARED=combined_prepared

if [ ! -d "$RAIN_RAW/sessions" ] || [ -z "$(ls -A "$RAIN_RAW/sessions" 2>/dev/null)" ]; then
  echo "=== 0/4: downloading KNMI hourly rainfall -> $RAIN_RAW ==="
  "$PY" -u cuts_plus_prototype/download_knmi_rain.py --out "$RAIN_RAW" --start 2005-01-01 --end 2025-01-01
else
  echo "=== 0/4: KNMI rainfall already in $RAIN_RAW, skipping download ==="
fi

echo "=== 1/4: occluding implausible water levels $RWS_RAW -> $ADAPTED ==="
"$PY" cuts_plus_prototype/prepare_rws_waterinfo.py --input-dir "$RWS_RAW" --output-dir "$ADAPTED"

echo "=== 2/4: combining water levels with rainfall -> $COMBINED ==="
"$PY" cuts_plus_prototype/combine_rws_knmi.py \
  --wl-dir "$ADAPTED" --rain-dir "$RAIN_RAW/sessions" \
  --wl-locations "$ADAPTED/locations.csv" --rain-locations "$RAIN_RAW/locations.csv" \
  --output "$COMBINED"

echo "=== 3/4: resampling $COMBINED -> $PREPARED ==="
"$PY" cuts_plus_prototype/prepare_data.py --input-dir "$COMBINED" --output-dir "$PREPARED"

echo "=== 4/4: dropping channels below $MIN_AVAILABILITY availability ==="
"$PY" cuts_plus_prototype/drop_sparse_channels.py --data-dir "$PREPARED" --min-availability "$MIN_AVAILABILITY"

echo "Done. Combined dataset in $PREPARED (locations in $COMBINED/locations.csv)"
