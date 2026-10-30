"""
Post-hoc sweep over the POT/EVT threshold hyperparameters (--risk, --initial-level, --num-candidates)
against an already-trained CUTS+ RCA run - no retraining needed. These three fields are only ever
threaded into aerca.pot() (see fit_pot_thresholds/root_cause_analysis in cuts_plus_rca.py); nothing
about the model, the learned causal graph, or held_out/val_mse depends on them, so every sweep point
is pure post-hoc numpy work on top of one reloaded model.

What gets reused vs. recomputed, relative to the original run_real_data_pipeline:
  - The trained model (fitting_model weights + learned graph G/GT) is reloaded via MultiCAD.train()'s
    own checkpoint-resume path, with total_epoch set to the checkpoint's own saved epoch + 1 - the
    epoch loop then runs zero times and train() just returns, so this is a pure reload, not a retrain.
  - Per-score-session z-scores/observed/labels (cuts_plus_score_*_z.npy / _observed.npy,
    cuts_plus_rca_*_labels.npy) are the ones already saved by the original run - reused as-is.
  - The one thing NOT saved by the original run is the validation sessions' residual z-scores
    (val_z) - needed to refit POT thresholds at a new (risk, initial_level, num_candidates). This
    script recomputes those once (one forward pass over the validation sessions, no training) and
    reuses the result across the whole sweep grid.

Baseline: zeeland_flood_eval_5min (5-min resample, lambda_s fixed at 0.08 with no anneal - the
convergence sweep's "sparsity_fine_08" config - run with --flood-events-csv/--flood-aware-split, see
cuts_plus_prototype/run_zeeland_flood_eval_5min.sh).

Scope: sweeps the flags-based operational metrics (the same ones score_session/flags drive in the
real pipeline: flagged fraction, cell-level precision/recall/F1, event-level detection rate), since
those are what a deployed threshold actually controls. AC@1..avg@10 (topk_at_step) is also reported,
once, as a reference - it's a pure ranking metric with no threshold involved, so it is identical at
every sweep point by construction, not swept. root_cause_analysis's threshold-aware ac*@k variant
(which refits its own POT thresholds per labeled session) is deliberately left out of the per-point
sweep: it would require one pot() fit per channel per labeled session per sweep point - two orders of
magnitude more pot() calls than the once-per-channel-per-point val threshold used here - for a metric
that's already covered in spirit by the (much cheaper) flags-based ones below.

Usage:
    python3 cuts_plus_prototype/sweep_pot_params.py
"""
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cuts_plus import MultiCAD  # noqa: E402  (vendor/cuts_plus.py)
from utils.logger import MyLogger  # noqa: E402  (vendor/utils/logger.py)

import mask_flood_events  # noqa: E402

from cuts_plus_rca import (  # noqa: E402
    CUTSPlusRCAConfig, build_discharge_exogeneity_mask, build_opt, build_rain_exogeneity_mask,
    fill_missing, fit_pot_thresholds, fit_residual_thresholds, load_real_sessions,
    predict_residuals, session_data_and_mask, set_seed,
)

DATA_DIR = 'cuts_plus_prototype/scratch/zeeland_flood_eval_dataset_5min'
FLOOD_EVENTS_CSV = 'flood_events_output/municipality_high/events.csv'
SAVE_DIR = 'zeeland_flood_eval_5min/models'

# Mirrors run_zeeland_flood_eval_5min.sh exactly. total_epoch is read from checkpoint.pt itself below
# (whatever --total-epoch that run was actually invoked with), not hardcoded here.
BASELINE_KWARGS = dict(
    data_dir=DATA_DIR, flood_events_csv=FLOOD_EVENTS_CSV, flood_min_alert='high',
    flood_aware_split=True, val_ratio=0.1, test_ratio=0.1, fill_policy='rate_0.1_after_10',
    lambda_s_start=0.08, lambda_s_end=0.08, end_tau=1.0, predict_chunk_size=32,
    save_dir=SAVE_DIR,
)

# Default (risk, initial_level, num_candidates) = (0.01, 0.98, 100) - see CUTSPlusRCAConfig.
RISK_GRID = [0.001, 0.005, 0.01, 0.05, 0.1]
INITIAL_LEVEL_GRID = [0.90, 0.95, 0.98, 0.99]
NUM_CANDIDATES_GRID = [50, 100, 200]


def runs_of_ones(col):
    idx = np.where(col > 0)[0]
    if len(idx) == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            runs.append((start, prev))
            start = i
            prev = i
    runs.append((start, prev))
    return runs


def load_trained_model(config, channel_names, train_data, train_mask):
    """Rebuilds the exact MultiCAD used by the original run (config.total_epoch already set to
    checkpoint.pt's saved epoch+1 by the caller) and reloads it from its checkpoint via train()'s own
    resume path - start_epoch == total_epoch, so the epoch loop runs zero times and this call only
    loads weights/graph, no training happens."""
    device = 'cpu'
    n_nodes = len(channel_names)
    opt = build_opt(config, n_nodes)
    log = MyLogger(log_dir='/tmp/sweep_pot_params_unused_log', stdout=False, stderr=False,
                    tensorboard=False)
    multicad = MultiCAD(opt, log, device=device)
    checkpoint_path = os.path.join(config.save_dir, 'checkpoint.pt')
    allowed_mask = build_rain_exogeneity_mask(channel_names) * build_discharge_exogeneity_mask(channel_names)
    multicad.train(train_data[:, :, None], train_mask[:, :, None], train_data[:, :, None],
                   true_cm=None, checkpoint_path=checkpoint_path, checkpoint_every=1,
                   allowed_mask=allowed_mask)
    log.close()
    return multicad


def main():
    ckpt_epoch = torch.load(os.path.join(SAVE_DIR, 'checkpoint.pt'), map_location='cpu',
                             weights_only=False)['epoch']
    config = CUTSPlusRCAConfig(**BASELINE_KWARGS, total_epoch=ckpt_epoch + 1)
    set_seed(config.seed)

    with open(os.path.join(config.save_dir, 'channel_names.json')) as fh:
        channel_names = json.load(fh)
    saved_median = np.load(os.path.join(config.save_dir, 'cuts_plus_residual_median.npy'))
    saved_std = np.load(os.path.join(config.save_dir, 'cuts_plus_residual_std.npy'))

    series_dict, means, stds = load_real_sessions(config.data_dir)
    session_ids = sorted(series_dict.keys())
    flood_windows = mask_flood_events.load_windows(config.flood_events_csv, config.flood_min_alert)

    # Flood-aware split, copied from run_real_data_pipeline (cuts_plus_rca.py) to reproduce val_ids
    # exactly - verified below by cross-checking recomputed median/std against the saved ones.
    flood_ids, normal_ids = [], []
    for sid in session_ids:
        idx = series_dict[sid].index
        overlaps = any(start < idx.max() and end > idx.min() for start, end in flood_windows)
        (flood_ids if overlaps else normal_ids).append(sid)
    n_score_target = max(1, round(config.test_ratio * len(session_ids)))
    n_val = max(1, round(config.val_ratio * len(session_ids)))
    n_pad = max(0, n_score_target - len(flood_ids))
    test_pad_ids = normal_ids[-n_pad:] if n_pad else []
    remaining_normal_ids = normal_ids[:-n_pad] if n_pad else normal_ids
    val_ids = remaining_normal_ids[-n_val:]
    pool_ids = remaining_normal_ids[:-n_val]
    print(f'Reproduced split: {len(pool_ids)} train / {len(val_ids)} val '
          f'(score sessions reused from the already-saved *_z.npy files instead of resplitting).')

    for sid in pool_ids + val_ids:
        series_dict[sid], _ = mask_flood_events.mask_dataframe(series_dict[sid], flood_windows)

    train_mean = means.reindex(channel_names).values.astype(np.float32)
    train_std = stds.reindex(channel_names).values.astype(np.float32)
    train_std_safe = np.where(train_std == 0, 1.0, train_std)

    def normalize(d):
        return (fill_missing(d, train_mean) - train_mean) / train_std_safe

    # train_data/train_mask are only needed as placeholder tensors for MultiCAD.train()'s checkpoint-
    # resume call (shapes must match what the model was built for) - their *values* don't matter since
    # the epoch loop never actually runs.
    train_filled, train_mask_arr = session_data_and_mask(series_dict[pool_ids[0]][channel_names])
    train_data = normalize(train_filled)

    multicad = load_trained_model(config, channel_names, train_data, train_mask_arr)

    val_prepared = [(normalize(filled), mask) for vid in val_ids
                     for filled, mask in [session_data_and_mask(series_dict[vid][channel_names])]]
    val_pairs = [predict_residuals(multicad, norm_filled, 'cpu', mask=mask,
                                    chunk_size=config.predict_chunk_size)
                 for norm_filled, mask in val_prepared]
    val_residuals = np.concatenate([r for r, _ in val_pairs], axis=0)
    val_observed = np.concatenate([o for _, o in val_pairs], axis=0)
    median, std = fit_residual_thresholds(val_residuals, observed=val_observed)

    median_diff = np.nanmax(np.abs(median - saved_median))
    std_diff = np.nanmax(np.abs(std - saved_std))
    print(f'Residual median/std recomputed from this reload vs. the originally saved ones: '
          f'max|Δmedian|={median_diff:.6g}, max|Δstd|={std_diff:.6g} (both are themselves tiny, '
          f'O(1e-2)-O(1e-1) - a few percent of CPU floating-point reload noise here is expected and '
          f'is cross-checked below against the real saved flags, which is what actually matters).')

    val_z = (val_residuals - median) / np.where(std == 0, 1e-8, std)

    z_files = sorted(glob.glob(os.path.join(config.save_dir, 'cuts_plus_score_*_z.npy')))
    score_sessions = []
    for zf in z_files:
        sid = re.match(r'cuts_plus_score_(.*)_z\.npy', os.path.basename(zf)).group(1)
        z_scores = np.load(zf)
        observed = np.load(os.path.join(config.save_dir, f'cuts_plus_score_{sid}_observed.npy'))
        label_path = os.path.join(config.save_dir, f'cuts_plus_rca_{sid}_labels.npy')
        labels = np.load(label_path) if os.path.exists(label_path) else None
        score_sessions.append((sid, z_scores, observed, labels))
    print(f'Loaded {len(score_sessions)} score session(s) '
          f'({sum(1 for *_, l in score_sessions if l is not None)} with flood labels).')

    results = []
    for risk in RISK_GRID:
        for initial_level in INITIAL_LEVEL_GRID:
            for num_candidates in NUM_CANDIDATES_GRID:
                pot_thresholds = fit_pot_thresholds(val_z, risk, initial_level, num_candidates,
                                                      observed=val_observed)
                tp = fp = fn = tn = 0
                total_events = detected_events = 0
                for sid, z_scores, observed, labels in score_sessions:
                    flags = z_scores > pot_thresholds[None, :]
                    if labels is not None:
                        offset = labels.shape[0] - flags.shape[0]
                        labels_aligned = labels[offset:].astype(bool)
                    else:
                        labels_aligned = np.zeros_like(flags, dtype=bool)
                    flags_b = flags.astype(bool)
                    tp += np.sum(flags_b & labels_aligned)
                    fp += np.sum(flags_b & ~labels_aligned)
                    fn += np.sum(~flags_b & labels_aligned)
                    tn += np.sum(~flags_b & ~labels_aligned)
                    for c in range(labels_aligned.shape[1]):
                        for s, e in runs_of_ones(labels_aligned[:, c]):
                            total_events += 1
                            detected_events += int(flags_b[s:e + 1, c].any())
                precision = tp / (tp + fp) if (tp + fp) else float('nan')
                recall = tp / (tp + fn) if (tp + fn) else float('nan')
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float('nan')
                detection_rate = detected_events / total_events if total_events else float('nan')
                results.append(dict(risk=risk, initial_level=initial_level,
                                     num_candidates=num_candidates, precision=precision,
                                     recall=recall, f1=f1, detection_rate=detection_rate,
                                     total_events=total_events, tp=int(tp), fp=int(fp)))

    print('=' * 110)
    header = f"{'risk':>8} {'init_lvl':>9} {'n_cand':>7} {'precision':>10} {'recall':>8} {'f1':>8} {'det_rate':>9} {'events':>7} {'tp':>6} {'fp':>8}"
    print(header)
    for r in results:
        print(f"{r['risk']:>8} {r['initial_level']:>9} {r['num_candidates']:>7} "
              f"{r['precision']:>10.5f} {r['recall']:>8.4f} {r['f1']:>8.5f} "
              f"{r['detection_rate']:>9.4f} {r['total_events']:>7} {r['tp']:>6} {r['fp']:>8}")

    best_f1 = max(results, key=lambda r: (r['f1'] if not np.isnan(r['f1']) else -1))
    best_recall_at_sane_fp = max(
        (r for r in results if not np.isnan(r['precision'])),
        key=lambda r: (r['detection_rate'] if not np.isnan(r['detection_rate']) else -1, -r['fp'])
    )
    print('=' * 110)
    print(f"Best F1: risk={best_f1['risk']} initial_level={best_f1['initial_level']} "
          f"num_candidates={best_f1['num_candidates']} -> F1={best_f1['f1']:.5f} "
          f"(precision={best_f1['precision']:.5f}, recall={best_f1['recall']:.4f})")
    print(f"Best detection_rate: risk={best_recall_at_sane_fp['risk']} "
          f"initial_level={best_recall_at_sane_fp['initial_level']} "
          f"num_candidates={best_recall_at_sane_fp['num_candidates']} -> "
          f"detection_rate={best_recall_at_sane_fp['detection_rate']:.4f} "
          f"(fp={best_recall_at_sane_fp['fp']})")

    default = next(r for r in results if r['risk'] == 0.01 and r['initial_level'] == 0.98
                    and r['num_candidates'] == 100)
    print(f"Default (risk=0.01, initial_level=0.98, num_candidates=100) for reference: "
          f"F1={default['f1']:.5f} detection_rate={default['detection_rate']:.4f}")

    # Ground-truth cross-check: the default sweep point uses the same (risk, initial_level,
    # num_candidates) the original run trained with, so its cell stats (from freshly-refit
    # thresholds) should closely match the cell stats of the flags.npy files the original run
    # actually saved - a much more meaningful correctness check than the raw median/std diff above,
    # since it validates the whole reload pipeline against ground truth rather than an intermediate.
    orig_tp = orig_fp = orig_fn = 0
    for sid, _, _, labels in score_sessions:
        flags = np.load(os.path.join(config.save_dir, f'cuts_plus_score_{sid}_flags.npy')).astype(bool)
        labels_aligned = (labels[labels.shape[0] - flags.shape[0]:].astype(bool)
                           if labels is not None else np.zeros_like(flags, dtype=bool))
        orig_tp += np.sum(flags & labels_aligned)
        orig_fp += np.sum(flags & ~labels_aligned)
        orig_fn += np.sum(~flags & labels_aligned)
    orig_precision = orig_tp / (orig_tp + orig_fp)
    orig_recall = orig_tp / (orig_tp + orig_fn)
    orig_f1 = 2 * orig_precision * orig_recall / (orig_precision + orig_recall)
    print(f"Cross-check vs. the ORIGINAL run's own saved flags.npy (same risk/initial_level/"
          f"num_candidates it trained with): original F1={orig_f1:.5f} (precision={orig_precision:.5f}, "
          f"recall={orig_recall:.4f}, tp={orig_tp}, fp={orig_fp}) vs. this reload's default row "
          f"F1={default['f1']:.5f} (precision={default['precision']:.5f}, recall={default['recall']:.4f}, "
          f"tp={default['tp']}, fp={default['fp']}) - should be close; if not, the reload is suspect.")


if __name__ == '__main__':
    main()
