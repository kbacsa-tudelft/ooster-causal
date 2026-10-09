#!/usr/bin/env bash
# Sets up a private Python environment and runs download_rws_waterlevel.py with the full 1950-2025
# range. Historical coverage varies by station - the script already tries every measuring-device code
# a station's catalog entry lists, picking up whichever era-appropriate one has data, so no special
# handling is needed here for that; expect far fewer stations and coarser cadence before the 1990s-2000s.
#
# Safe to interrupt and rerun with the same arguments: finished years and sessions are skipped.
#
# Run from the repository root.
#
# Usage:
#   cuts_plus_prototype/run_rws_download.sh [output_dir] [extra args for download_rws_waterlevel.py]
#
#   output_dir  default rws_waterinfo (relative to the current directory)
#
# Long run: start it detached so it survives closing the terminal:
#   nohup cuts_plus_prototype/run_rws_download.sh > rws_download.log 2>&1 &
#   tail -f rws_download.log
set -euo pipefail

OUT_DIR="${1:-rws_waterinfo}"
shift || true

VENV=.venv-rws
if [ ! -d "$VENV" ]; then
  echo "=== creating environment in $VENV ==="
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install rws-waterinfo pandas pyarrow
fi

echo "=== downloading to $OUT_DIR ==="
"$VENV/bin/python" -u cuts_plus_prototype/download_rws_waterlevel.py \
  --out "$OUT_DIR" \
  --start 1950-01-01 --end 2025-01-01 \
  --workers 10 \
  "$@"

echo "Done. Sessions in $OUT_DIR/sessions, provenance in $OUT_DIR/channels.csv, coordinates in $OUT_DIR/locations.csv"
