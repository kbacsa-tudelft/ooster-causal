#!/usr/bin/env bash
# One-shot GPU-machine runner for the rain-inclusive Zeeland contrast experiment: builds a
# Zeeland dataset with KNMI rainfall folded in as causal-only inputs, then trains the exact same
# config that won the water-level-only convergence sweep (convergence_sweep_zeeland/sparsity_fine_08),
# so the two runs can be compared directly with compare_fill_policies.py.
#
# Run from the repo root:
#   cuts_plus_prototype/run_zeeland_rain_experiment.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RAIN_WINDOW=24h
DATASET_DIR=cuts_plus_prototype/scratch/convergence_dataset_zeeland_rain
RUN_DIR=convergence_sweep_zeeland_rain/sparsity_fine_08

echo "=== Step 1/3: rain data ==="
if [ ! -d knmi_rain ]; then
  if [ -d download_package/knmi_rain ]; then
    echo "Copying download_package/knmi_rain -> knmi_rain"
    cp -r download_package/knmi_rain knmi_rain
  else
    echo "knmi_rain/ not found at repo root or in download_package/ - copy it in before running this script." >&2
    exit 1
  fi
else
  echo "knmi_rain/ already present, reusing it."
fi

echo "=== Step 2/3: build the Zeeland+rain dataset (rain-window=$RAIN_WINDOW) ==="
cuts_plus_prototype/build_dataset.sh "$DATASET_DIR" zeeland rws_waterinfo "" knmi_rain auto 1974-01-01 "$RAIN_WINDOW"

echo "=== Step 3/3: train sparsity_fine_08 against it ==="
python3 cuts_plus_prototype/cuts_plus_rca.py --data-dir "$DATASET_DIR" \
  --total-epoch 50 --predict-chunk-size 32 --lambda-s-start 0.08 --lambda-s-end 0.08 --end-tau 1.0 \
  --fill-policy rate_0.1_after_10 --held-out-eval-every 2 \
  --save-dir "$RUN_DIR/models" \
  --log-dir "$RUN_DIR/runs"

echo "Done. Compare against the water-level-only run with:"
echo "  python3 cuts_plus_prototype/compare_fill_policies.py --sweep-root convergence_sweep_zeeland"
echo "  python3 cuts_plus_prototype/compare_fill_policies.py --sweep-root convergence_sweep_zeeland_rain"
