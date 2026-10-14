#!/usr/bin/env bash
# Trains 6 named configs on the Limburg dataset to test candidate fixes for the convergence problem
# found on the real limburg_flood_eval run: graph_discov/data_loss (S2) climbed almost monotonically
# through training while held_out/val_mse (the genuinely held-out metric) peaked at epoch 10 and never
# recovered. Mining that run's checkpoint.pt/TensorBoard history pinned down two concrete events:
#   1. A group-refinement reset at epoch 5 (n_groups jumps 32 -> 57 = n_nodes in one step, discarding
#      all Adam momentum via a brand-new graph optimizer) - data_loss climbs for 5 epochs right after
#      this while held_out/val_mse actually *improves* over the same window (its best value is at
#      epoch 10) - the climbing training loss there is post-reset resettling noise, not real
#      degradation.
#   2. fill_policy=rate_0.1_after_10's blending starts at epoch 10 (right after that best reading),
#      coinciding with another data_loss jump and held_out/val_mse's one regression (0.445 -> 0.550 by
#      epoch 15). After that, held_out/val_mse plateaus/oscillates while data_loss keeps climbing
#      smoothly - consistent with the continuous gumbel_tau anneal (sharper discrete sampling) and
#      lambda_s decay (less sparsity pressure) making the *training* metric mechanically noisier over
#      time, largely independent of true graph quality.
#
# All 4 candidate interventions below are reachable via existing CLI flags - no vendor/cuts_plus.py
# training-loop changes needed to test them (a real fix - preserving Adam momentum across a group-
# refinement reset instead of discarding it - is a plausible follow-up if this sweep confirms the
# reset is the dominant contributor, but is out of scope here).
#
# Configs:
#   baseline              reproduces the real run's config (fill_policy rate_0.1_after_10, default
#                         group_policy/n_groups/tau/lambda_s) - reference point
#   no_coarse_to_fine     --n-groups 1000000 - cuts_plus_rca.py's build_opt clamps n_groups to
#                         min(n_groups, n_nodes), so this lands exactly on n_nodes from epoch 0:
#                         vendor/cuts_plus.py's "elif epoch_i == 0 and self.n_groups ==
#                         self.args.n_nodes" branch fires immediately and the "self.n_groups <
#                         self.args.n_nodes" doubling branch never fires again - skips the epoch-5
#                         reset entirely
#   gentle_anneal         --end-tau 0.3 --lambda-s-end 0.1 - softer Gumbel-softmax sharpening and more
#                         sustained sparsity pressure through training
#   delayed_fill_policy   --fill-policy rate_0.1_after_20 - pushes the imputation-feedback onset 10
#                         epochs past the point where post-reset settling has had time to stabilize
#   combined              all three interventions together
#   strict_availability   --min-channel-availability 0.15 - drops channels observed in less than 15%
#                         of training rows (60/80 Limburg channels; the dataset's own real_data run
#                         logged data/train_missing_fraction=0.77 at the default 0.0 threshold, so most
#                         of the training signal the model is fitting is its own mean-fill/imputation
#                         feedback, not real readings - 0.15 was picked over the originally-discussed
#                         0.1 because it lands on a plateau (0.15 and 0.20 both keep exactly the same
#                         20 channels at 54% missing - confirmed by sweeping --min-channel-availability
#                         from 0.0 to 0.7 against this dataset's actual train split), the least
#                         aggressive point before the missing-fraction curve flattens out further
#
# Usage:
#   cuts_plus_prototype/sweep_convergence.sh [total_epoch] [predict_chunk_size] [held_out_eval_every]
#
#   total_epoch          default 50 (matches the real run, for comparability)
#   predict_chunk_size   default 32
#   held_out_eval_every  default 2 (finer than the 5 used before, to localize the epoch-5-15 dynamics)
set -euo pipefail

TOTAL_EPOCH="${1:-50}"
PREDICT_CHUNK_SIZE="${2:-32}"
HELD_OUT_EVAL_EVERY="${3:-2}"

LAMBDA_S_START=0.3
LAMBDA_S_END=0.03

SWEEP_ROOT=cuts_plus_prototype/scratch/convergence_sweep
# Built outside SWEEP_ROOT (not a sibling of the config run dirs) so compare_fill_policies.py
# --sweep-root "$SWEEP_ROOT" only ever sees the 6 config directories, not this dataset build too.
DATA_DIR=cuts_plus_prototype/scratch/convergence_dataset_limburg

CONFIGS=(baseline no_coarse_to_fine gentle_anneal delayed_fill_policy combined strict_availability)
declare -A CONFIG_OVERRIDES=(
  [baseline]=""
  [no_coarse_to_fine]="--n-groups 1000000"
  [gentle_anneal]="--end-tau 0.3 --lambda-s-end 0.1"
  [delayed_fill_policy]="--fill-policy rate_0.1_after_20"
  [combined]="--n-groups 1000000 --end-tau 0.3 --lambda-s-end 0.1 --fill-policy rate_0.1_after_20"
  [strict_availability]="--min-channel-availability 0.15"
)

echo "=== Building Limburg dataset -> $DATA_DIR ==="
cuts_plus_prototype/build_dataset.sh "$DATA_DIR" limburg rws_waterinfo "" ""

for config in "${CONFIGS[@]}"; do
  run_dir="$SWEEP_ROOT/$config"
  read -ra OVERRIDE_ARGS <<< "${CONFIG_OVERRIDES[$config]}"
  echo "=== config=$config -> $run_dir (overrides: ${CONFIG_OVERRIDES[$config]:-none}) ==="
  python3 cuts_plus_prototype/cuts_plus_rca.py \
    --data-dir "$DATA_DIR" \
    --total-epoch "$TOTAL_EPOCH" \
    --predict-chunk-size "$PREDICT_CHUNK_SIZE" \
    --lambda-s-start "$LAMBDA_S_START" \
    --lambda-s-end "$LAMBDA_S_END" \
    --fill-policy rate_0.1_after_10 \
    --held-out-eval-every "$HELD_OUT_EVAL_EVERY" \
    --save-dir "$run_dir/models" \
    --log-dir "$run_dir/runs" \
    "${OVERRIDE_ARGS[@]}"
done

echo "Done. Compare configs with:"
echo "  python3 cuts_plus_prototype/compare_fill_policies.py --sweep-root $SWEEP_ROOT"
echo "Or inspect individually with TensorBoard:"
logdir_spec=""
for config in "${CONFIGS[@]}"; do
  logdir_spec="${logdir_spec:+$logdir_spec,}$config:$SWEEP_ROOT/$config/runs"
done
echo "  tensorboard --logdir_spec=$logdir_spec"
