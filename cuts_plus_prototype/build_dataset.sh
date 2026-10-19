#!/usr/bin/env bash
# Builds a trainable CUTS+ dataset into OUTPUT_DIR by chaining the existing (unmodified) prep scripts -
# occlude/prefix each raw source, merge water level + discharge + rain, resample onto one frequency,
# optionally filter to a region - into a scratch staging directory that's removed afterward, so nothing
# but OUTPUT_DIR survives. Replaces hand-running prepare_rws_waterinfo.py -> combine_rws_knmi.py ->
# prepare_data.py -> filter_by_region.py into a new permanently-named directory every time a different
# dataset variant (region, resample frequency, water-level-only vs combined) is needed - the raw
# downloads (rws_waterinfo/, rws_discharge/, knmi_rain/) are the only dataset directories meant to be
# kept around; everything else is rebuilt on demand by this script.
#
# No drop_sparse_channels.py stage: that's already handled at training time by
# cuts_plus_rca.py's --min-channel-availability, computed from the actual training split.
#
# Usage:
#   cuts_plus_prototype/build_dataset.sh OUTPUT_DIR [REGION] [WL_DIR] [Q_DIR] [RAIN_DIR] [RESAMPLE_FREQ] [START] [RAIN_WINDOW]
#
#   OUTPUT_DIR     required - where the final trainable dataset lands (overwritten if it exists)
#   REGION         'zeeland' | 'limburg' | '' (default '', no region filtering)
#   WL_DIR         default rws_waterinfo
#   Q_DIR          default rws_discharge; pass '' to omit discharge
#   RAIN_DIR       default knmi_rain; pass '' to omit rain
#   RESAMPLE_FREQ  default 'auto' (prepare_data.py's own coarsest-native-cadence auto-detection - a
#                  handful of anomalously coarse-sampled sessions can drag this down badly, see START)
#   START          default '1974-01-01' (prepare_data.py's --start: drops sessions before this date).
#                  rws_waterinfo's raw history includes a 1950-1970 era sampled 3-hourly, far coarser
#                  than the 10-15min cadence everywhere else - left in, --freq auto picks 3h and every
#                  2-week session then falls below --min-rows, dropping the entire dataset. 1974-01-01
#                  is this project's own established cutoff (matches the historical
#                  rws_waterinfo_prepared_1974/rws_waterinfo_limburg/rws_waterinfo_zeeland directories'
#                  session count exactly - 1330 sessions). Pass '' to disable (keep every session).
#   RAIN_WINDOW    default '' (no rolling-sum accumulation); forwarded to prepare_data.py --rain-window
#                  (e.g. '24h') when non-empty - replaces each RH_* channel with its own rolling sum
#                  over that window before normalization. Ignored when RAIN_DIR is ''.
#
#   Water-level-only: pass '' for both Q_DIR and RAIN_DIR - the merge stage is skipped entirely and
#   water level is resampled/filtered directly.
#
# Region bboxes (lat_min lat_max lon_min lon_max) - not recorded verbatim anywhere else; confirmed by
# cross-referencing rws_waterinfo_limburg/rws_waterinfo_zeeland's actual station coordinates before
# those directories were deleted in favor of this script.
set -euo pipefail

OUTPUT_DIR="${1:?Usage: build_dataset.sh OUTPUT_DIR [REGION] [WL_DIR] [Q_DIR] [RAIN_DIR] [RESAMPLE_FREQ]}"
REGION="${2:-}"
WL_DIR="${3:-rws_waterinfo}"
Q_DIR="${4-rws_discharge}"
RAIN_DIR="${5-knmi_rain}"
RESAMPLE_FREQ="${6:-auto}"
START="${7-1974-01-01}"
RAIN_WINDOW="${8-}"

declare -A REGION_BBOX=(
  [zeeland]="51.20 51.78 3.30 4.30"
  [limburg]="50.75 51.80 5.60 6.20"
)
if [ -n "$REGION" ] && [ -z "${REGION_BBOX[$REGION]+x}" ]; then
  echo "Unknown region '$REGION' - known regions: ${!REGION_BBOX[*]}" >&2
  exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "=== Occluding water level: $WL_DIR -> staging ==="
python3 cuts_plus_prototype/prepare_rws_waterinfo.py --input-dir "$WL_DIR" --output-dir "$STAGE/wl_adapted"

PREPARE_INPUT="$STAGE/wl_adapted"
LOCATIONS_CSV="$WL_DIR/locations.csv"

if [ -n "$Q_DIR" ] || [ -n "$RAIN_DIR" ]; then
  COMBINE_ARGS=(--wl-dir "$STAGE/wl_adapted" --wl-locations "$WL_DIR/locations.csv" --output "$STAGE/combined_adapted")

  if [ -n "$Q_DIR" ]; then
    echo "=== Occluding discharge: $Q_DIR -> staging ==="
    python3 cuts_plus_prototype/prepare_rws_waterinfo.py --input-dir "$Q_DIR" --output-dir "$STAGE/q_adapted" \
      --prefix Q_ --min-value -500 --max-value 20000
    COMBINE_ARGS+=(--discharge-dir "$STAGE/q_adapted" --discharge-locations "$Q_DIR/locations.csv")
  fi

  if [ -n "$RAIN_DIR" ]; then
    COMBINE_ARGS+=(--rain-dir "$RAIN_DIR/sessions" --rain-locations "$RAIN_DIR/locations.csv")
  else
    # combine_rws_knmi.py's --rain-dir/--rain-locations have no "disabled" sentinel (unlike discharge,
    # which accepts an omitted flag) - point it at an empty session dir + a header-only locations.csv
    # instead, which its own glob-returns-empty / pd.concat-of-an-empty-frame handling already no-ops.
    mkdir -p "$STAGE/empty_rain_sessions"
    echo "name,lat,lon" > "$STAGE/empty_rain_locations.csv"
    COMBINE_ARGS+=(--rain-dir "$STAGE/empty_rain_sessions" --rain-locations "$STAGE/empty_rain_locations.csv")
  fi

  echo "=== Merging water level + discharge + rain -> staging ==="
  python3 cuts_plus_prototype/combine_rws_knmi.py "${COMBINE_ARGS[@]}"
  PREPARE_INPUT="$STAGE/combined_adapted"
  LOCATIONS_CSV="$STAGE/combined_adapted/locations.csv"
fi

echo "=== Resampling -> staging (freq=$RESAMPLE_FREQ, start=${START:-<none>}) ==="
PREPARE_DATA_ARGS=(--input-dir "$PREPARE_INPUT" --output-dir "$STAGE/prepared" --freq "$RESAMPLE_FREQ")
if [ -n "$START" ]; then
  PREPARE_DATA_ARGS+=(--start "$START")
fi
if [ -n "$RAIN_DIR" ] && [ -n "$RAIN_WINDOW" ]; then
  PREPARE_DATA_ARGS+=(--rain-window "$RAIN_WINDOW")
fi
python3 cuts_plus_prototype/prepare_data.py "${PREPARE_DATA_ARGS[@]}"

rm -rf "$OUTPUT_DIR"
if [ -n "$REGION" ]; then
  read -r LAT_MIN LAT_MAX LON_MIN LON_MAX <<< "${REGION_BBOX[$REGION]}"
  echo "=== Filtering to region '$REGION' ($LAT_MIN-$LAT_MAX lat, $LON_MIN-$LON_MAX lon) -> $OUTPUT_DIR ==="
  python3 cuts_plus_prototype/filter_by_region.py --input-dir "$STAGE/prepared" --locations-csv "$LOCATIONS_CSV" \
    --output-dir "$OUTPUT_DIR" --lat-min "$LAT_MIN" --lat-max "$LAT_MAX" --lon-min "$LON_MIN" --lon-max "$LON_MAX"
else
  mkdir -p "$(dirname "$OUTPUT_DIR")"
  mv "$STAGE/prepared" "$OUTPUT_DIR"
fi

echo "Done: $OUTPUT_DIR"
