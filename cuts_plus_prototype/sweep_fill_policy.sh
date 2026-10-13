#!/usr/bin/env bash
# Trains one full CUTS+ RCA run per (region, --fill-policy) pair, each into its own save-dir/log-dir,
# so the imputation mechanism's effect on graph inference can be compared the same way the single
# `1974` and `imputation` runs were analyzed by hand: TensorBoard (S1 pred_loss vs S2 data_loss/
# sparsity_loss) and the discovered graph's diagonal (self-edge) strength. See
# compare_fill_policies.py to pull that comparison automatically, once per region, after the runs
# are done.
#
# Two regions by default - Zeeland (mostly tidal/estuary, few rivers) and Limburg (the Maas river
# corridor) - built with filter_by_region.py, to see whether the imputation mechanism's effect
# differs between a river-dominated and a tide-dominated network. Other hyperparameters are fixed to
# match the `1974`/`imputation` runs (lambda_s 0.3 -> 0.03) so region and fill_policy are the only
# things varying across the sweep.
#
# Usage:
#   cuts_plus_prototype/sweep_fill_policy.sh [total_epoch] [predict_chunk_size]
#
#   total_epoch         default 50 (matches the `imputation` run)
#   predict_chunk_size  default 32 (lower this if validation/scoring runs out of memory)
set -euo pipefail

TOTAL_EPOCH="${1:-50}"
PREDICT_CHUNK_SIZE="${2:-32}"

LAMBDA_S_START=0.3
LAMBDA_S_END=0.03

SWEEP_ROOT=cuts_plus_prototype/scratch/fill_policy_sweep

declare -A REGION_DATA_DIRS=(
  [zeeland]="rws_waterinfo_zeeland"
  [limburg]="rws_waterinfo_limburg"
)

# none: today's default (static mean-fill, never updated). rate_X_after_Y: soft EMA blend toward the
# model's own predictions at missing positions, starting epoch Y. every_N: hard overwrite with the
# model's predictions (and the positions start counting as real loss targets) every N epochs - the
# most aggressive policy, included as an upper bound on how much feedback can help or hurt.
POLICIES=(
  "none"
  "rate_0.1_after_10"
  "rate_0.3_after_10"
  "every_1"
  "every_2"
  "every_3"
  "every_4"
  "every_5"
  "every_10"
)

for region in "${!REGION_DATA_DIRS[@]}"; do
  data_dir="${REGION_DATA_DIRS[$region]}"
  for policy in "${POLICIES[@]}"; do
    run_dir="$SWEEP_ROOT/$region/$policy"
    echo "=== region=$region fill_policy=$policy -> $run_dir ==="
    python3 cuts_plus_prototype/cuts_plus_rca.py \
      --data-dir "$data_dir" \
      --total-epoch "$TOTAL_EPOCH" \
      --predict-chunk-size "$PREDICT_CHUNK_SIZE" \
      --lambda-s-start "$LAMBDA_S_START" \
      --lambda-s-end "$LAMBDA_S_END" \
      --fill-policy "$policy" \
      --save-dir "$run_dir/models" \
      --log-dir "$run_dir/runs"
  done
done

echo "Done. Compare each region with:"
for region in "${!REGION_DATA_DIRS[@]}"; do
  echo "  python3 cuts_plus_prototype/compare_fill_policies.py --sweep-root $SWEEP_ROOT/$region"
done

echo "Or inspect individually with TensorBoard:"
for region in "${!REGION_DATA_DIRS[@]}"; do
  logdir_spec=""
  for policy in "${POLICIES[@]}"; do
    logdir_spec="${logdir_spec:+$logdir_spec,}$policy:$SWEEP_ROOT/$region/$policy/runs"
  done
  echo "  tensorboard --logdir_spec=$logdir_spec"
done
