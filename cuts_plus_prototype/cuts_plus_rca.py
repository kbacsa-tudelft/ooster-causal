"""
Root-cause / anomaly-detection layer on top of CUTS+ (vendor/cuts_plus.py), reproducing the same
pipeline as ../aerca.py's main() - train, fit residual thresholds, root-cause analysis, causal
discovery evaluation - but using CUTS+'s predictor + learned causal graph instead of AERCA's
encoder/decoder.

CUTS+ itself only does causal discovery (vendor/cuts_plus.py's MultiCAD.train returns a graph). It
has no anomaly scoring or root-cause ranking. This module adds that layer on top, reusing the same
residual/EVT-threshold/top-k machinery aerca.py already implements (pot, topk, topk_at_step,
eval_causal_structure*) so the two pipelines are directly comparable.

Run:
    python3 cuts_plus_prototype/cuts_plus_rca.py --total-epoch 5 --synthetic-num-vars 20 \
        --synthetic-series-len 600
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass, fields

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cuts_plus import MultiCAD  # noqa: E402  (vendor/cuts_plus.py)
from utils.logger import MyLogger  # noqa: E402  (vendor/utils/logger.py)

from aerca import (  # noqa: E402
    eval_causal_structure,
    eval_causal_structure_binary,
    load_series_dict,
    pot,
    set_seed,
    topk,
    topk_at_step,
)


@dataclass
class CUTSPlusRCAConfig:
    # CUTS+ predictor architecture
    mlp_hid: int = 32
    gru_layers: int = 1
    shared_weights_decoder: bool = False
    concat_h: bool = True
    input_step: int = 1
    batch_size: int = 128

    # Optimization
    total_epoch: int = 30
    lr_data_start: float = 1e-2
    lr_data_end: float = 1e-3
    weight_decay: float = 1e-4  # L2 regularization on the predictor (fitting_model), via Adam's weight_decay
    lr_graph_start: float = 1e-3
    lr_graph_end: float = 1e-4
    lambda_s_start: float = 1e-1
    lambda_s_end: float = 1e-2
    start_tau: float = 1.0
    end_tau: float = 0.1

    # Coarse-to-fine graph discovery
    n_groups: int = 32
    group_policy: str = 'multiply_2_every_5'
    show_graph_every: int = 1000  # effectively never - keep runs cheap
    graph_plot_every: int = 10  # log a labeled adjacency-matrix figure to TensorBoard every N epochs

    # Root-cause / EVT thresholds (same semantics as aerca.AERCAConfig)
    causal_quantile: float = 0.80
    risk: float = 1e-2
    initial_level: float = 0.98
    num_candidates: int = 100

    # Data split: one continuous series, temporal split (CUTS+'s native data model)
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    # Synthetic data generation
    synthetic_num_vars: int = 174
    synthetic_series_len: int = 3000
    synthetic_edge_prob: float = 0.1
    synthetic_num_anomalies: int = 10
    synthetic_var_order: int = 4

    # Runtime / paths
    data_dir: str = ''  # directory of .parquet sessions (aerca.load_series_dict format); empty = synthetic
    save_dir: str = 'saved_models'
    log_dir: str = 'runs'
    device: str = ''
    predict_chunk_size: int = 512  # windows per forward pass in predict_residuals; lower this if
    # scoring/validation OOMs at high channel counts (message passing is O(n_nodes^2) per window)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train CUTS+ + root-cause-analysis on synthetic data.')
    defaults = CUTSPlusRCAConfig()
    for f in fields(defaults):
        value = getattr(defaults, f.name)
        arg_type = (lambda s: s.lower() in ('1', 'true', 'yes')) if f.type is bool else f.type
        parser.add_argument(f'--{f.name.replace("_", "-")}', type=arg_type, default=value)
    return parser


def make_synthetic_series(config: CUTSPlusRCAConfig):
    """Single (T, N) VAR-driven series + (N, N) ground-truth adjacency (causal_struct_value[i, j] == 1
    means j causes i, matching aerca.make_synthetic_dataset's convention) + (T, N) binary anomaly
    labels. Everything before the final test_ratio segment is anomaly-free."""
    rng = np.random.default_rng(config.seed)
    p = config.synthetic_num_vars
    order = config.synthetic_var_order
    T = config.synthetic_series_len

    causal_struct_value = (rng.random((p, p)) < config.synthetic_edge_prob).astype(float)
    np.fill_diagonal(causal_struct_value, 1.0)
    coeffs = [causal_struct_value * rng.uniform(-0.3, 0.3, size=(p, p)) / order for _ in range(order)]

    test_len = int(config.test_ratio * T)
    clean_len = T - test_len
    shock_times = rng.choice(range(order * 2, test_len), size=config.synthetic_num_anomalies, replace=False)
    shock_vars = rng.integers(0, p, size=config.synthetic_num_anomalies)
    shocks = {clean_len + int(t): int(v) for t, v in zip(shock_times, shock_vars)}

    x = np.zeros((T, p))
    x[:order] = rng.normal(scale=0.1, size=(order, p))
    labels = np.zeros((T, p))
    for t in range(order, T):
        value = sum(coeffs[k] @ x[t - k - 1] for k in range(order))
        value += rng.normal(scale=0.1, size=p)
        if t in shocks:
            value[shocks[t]] += 5.0
        x[t] = value
    for t, var in shocks.items():
        affected = np.where(causal_struct_value[:, var] > 0)[0]
        labels[t:t + order, affected] = 1.0

    return x.astype(np.float32), causal_struct_value, labels, clean_len


def load_real_sessions(data_dir: str):
    """Loads a directory of .parquet sessions via aerca.load_series_dict. Returns
    (series_dict, means, stds) - means/stds are None unless a normalization_stats.json is present
    alongside the parquet files."""
    series_dict, means, stds = load_series_dict(data_dir)
    if len(series_dict) < 3:
        raise ValueError(
            f'load_real_sessions found {len(series_dict)} session(s) in {data_dir}, need at least 3 '
            f'(at least one each for training, threshold-fitting, and scoring).'
        )
    return series_dict, means, stds


def concat_sessions(series_dict: dict, session_ids: list, boundary_gap: int):
    """Concatenates session DataFrames (in session_ids order) into one (T, N) array + an observ_mask
    that is 0 wherever the source data was NaN (real missing readings) and additionally 0 for the
    first `boundary_gap` timesteps of every session after the first, so CUTS+ (which natively
    supports missing/masked data) never trains on a window whose input reaches back into a different,
    unrelated session.

    Missing values are forward-filled (zero-order hold, matching CUTS+'s own recommended imputation
    strategy) independently per session, before concatenation - so a fill never bleeds from one
    session into the next. A session's leading NaN (before its first real observation) can't be
    forward-filled and is left as NaN - callers should still run fill_missing (e.g. with the training
    mean) as a fallback for that case, since the mask only excludes points from the loss, it doesn't
    stop NaN propagating through the forward pass."""
    raw_arrays = [series_dict[sid].values.astype(np.float32) for sid in session_ids]
    filled_arrays = [series_dict[sid].ffill().values.astype(np.float32) for sid in session_ids]
    data = np.concatenate(filled_arrays, axis=0)
    mask = (~np.isnan(np.concatenate(raw_arrays, axis=0))).astype(np.float32)
    offset = 0
    for arr in raw_arrays[:-1]:
        offset += arr.shape[0]
        mask[offset:offset + boundary_gap] = 0.0
    return data, mask


def build_opt(config: CUTSPlusRCAConfig, n_nodes: int):
    return OmegaConf.create({
        'n_nodes': n_nodes,
        'input_step': config.input_step,
        'batch_size': config.batch_size,
        'data_dim': 1,
        'total_epoch': config.total_epoch,
        'n_groups': min(config.n_groups, n_nodes),
        'group_policy': config.group_policy,
        'supervision_policy': 'full',
        'fill_policy': 'none',
        'show_graph_every': config.show_graph_every,
        'data_pred': {
            'pred_step': 1,
            'mlp_hid': config.mlp_hid,
            'gru_layers': config.gru_layers,
            'shared_weights_decoder': config.shared_weights_decoder,
            'concat_h': config.concat_h,
            'lr_data_start': config.lr_data_start,
            'lr_data_end': config.lr_data_end,
            'weight_decay': config.weight_decay,
        },
        'graph_discov': {
            'lambda_s_start': config.lambda_s_start,
            'lambda_s_end': config.lambda_s_end,
            'lr_graph_start': config.lr_graph_start,
            'lr_graph_end': config.lr_graph_end,
            'start_tau': config.start_tau,
            'end_tau': config.end_tau,
        },
    })


def predict_residuals(multicad: MultiCAD, data: np.ndarray, device, mask: np.ndarray = None,
                       chunk_size: int = 512):
    """One-step-ahead prediction residuals (actual - predicted) for every valid window in `data`,
    using the trained fitting_model and its current learned graph, in time order. `data` is 2D
    (T, N), already normalized the same way the training data was. `mask` (T, N), 1=observed,
    0=missing/imputed - defaults to all-observed. Returns (residuals, observed), where `observed`
    is the target-side mask aligned the same way as `residuals`, so callers can exclude residuals
    computed against imputed (never actually observed) values.

    Windows are processed `chunk_size` at a time rather than all at once: the network's internal
    message passing is O(n_nodes^2) per window, so a session with thousands of windows at a high
    channel count can exhaust memory in a single forward pass (this OOM'd at 633 channels with the
    whole-session batch this function used to build). Each window only depends on its own
    input_step-length slice - there's no recurrent state carried across windows - so chunking is
    purely a memory/batch-size choice and changes no result, only computed and reassembled in
    smaller pieces before being sorted back into time order."""
    n_nodes = data.shape[1]
    input_step = multicad.args.input_step
    if mask is None:
        mask = np.ones_like(data)
    data_t = torch.from_numpy(data[:, :, None]).float().to(device)
    mask_t = torch.from_numpy(mask[:, :, None]).float().to(device)

    t_length = data.shape[0]
    n_windows = t_length - input_step  # every valid window, none dropped
    t_idx_all = torch.arange(input_step, t_length, device=device, dtype=torch.long)
    x_offsets = torch.arange(-input_step, 0, device=device, dtype=torch.long)
    y_offsets = torch.zeros(1, device=device, dtype=torch.long)

    # Recompute the (untransposed, source->target) edge-weight matrix the network was actually
    # trained with - NOT the value returned by MultiCAD.train(), which gets transposed once for
    # comparison against true_cm and is not the convention the forward pass expects.
    graph = torch.einsum('nm,ml->nl', multicad.G, torch.sigmoid(multicad.GT))

    multicad.fitting_model.eval()
    residual_chunks, observed_chunks, t_idx_chunks = [], [], []
    with torch.no_grad():
        for start in range(0, n_windows, chunk_size):
            t_idx = t_idx_all[start:start + chunk_size]
            x_idx = t_idx.unsqueeze(1) + x_offsets.unsqueeze(0)
            y_idx = t_idx.unsqueeze(1) + y_offsets.unsqueeze(0)
            x = data_t[x_idx].permute(0, 2, 1, 3)
            y = data_t[y_idx].permute(0, 2, 1, 3)
            mask_x = mask_t[x_idx].permute(0, 2, 1, 3)
            mask_y = mask_t[y_idx].permute(0, 2, 1, 3)
            graph_expanded = graph[None].expand(x.shape[0], -1, -1)

            y_pred = multicad.fitting_model(x, mask_x, graph_expanded)
            residual_chunks.append((y - y_pred).squeeze(-1).squeeze(-1).cpu())
            observed_chunks.append(mask_y.squeeze(-1).squeeze(-1).cpu())
            t_idx_chunks.append(t_idx.cpu())

    residual = torch.cat(residual_chunks).numpy()
    observed = torch.cat(observed_chunks).numpy()
    order = np.argsort(torch.cat(t_idx_chunks).numpy())
    return residual[order], observed[order]


def session_data_and_mask(df):
    """Forward-fills one session's DataFrame (zero-order hold) and returns (filled_values, mask),
    where mask is 1 wherever the *original* (pre-fill) value was real - mirrors concat_sessions'
    per-session handling, for the single-session case (validation/scoring)."""
    raw = df.values.astype(np.float32)
    filled = df.ffill().values.astype(np.float32)
    mask = (~np.isnan(raw)).astype(np.float32)
    return filled, mask


def fill_missing(data: np.ndarray, fill_values: np.ndarray) -> np.ndarray:
    """Fallback fill for whatever NaN survives forward-filling (a session's leading gap, before its
    first real observation, can't be forward-filled). Replaces NaN in `data` (T, N) with the
    per-column `fill_values` (N,) - e.g. the training mean, so a masked position becomes exactly 0
    once normalized against that same mean."""
    return np.where(np.isnan(data), fill_values[None, :], data)


def fit_residual_thresholds(residuals: np.ndarray, observed: np.ndarray = None):
    """Per-column median/std of residuals. If `observed` (same shape, 1=observed) is given, only
    residuals computed against genuinely-observed values are used. Non-finite residuals (the network
    can output NaN/Inf for a window where some other channel's value is a numerically extreme
    outlier, since message passing mixes all channels together) are dropped before the statistics
    are computed, rather than poisoning median/std into NaN for the whole channel."""
    if observed is None:
        return np.median(residuals, axis=0), np.std(residuals, axis=0)
    median = np.full(residuals.shape[1], np.nan)
    std = np.full(residuals.shape[1], np.nan)
    for c in range(residuals.shape[1]):
        col = residuals[observed[:, c] > 0, c]
        col = col[np.isfinite(col)]
        if len(col) == 0:
            continue
        median[c] = np.median(col)
        std[c] = np.std(col)
    return median, std


def root_cause_analysis(residuals_test, labels_test, median, std, risk, initial_level, num_candidates,
                         input_step):
    """Mirrors aerca.AERCA._testing_root_cause: z-score residuals against validation statistics, fit
    a per-variable POT threshold, and rank root causes with topk / topk_at_step."""
    std_safe = np.where(std == 0, 1e-8, std)
    z_scores = (residuals_test - median) / std_safe

    pot_thresholds = np.array([
        pot(z_scores[:, i], risk, initial_level, num_candidates)[0]
        for i in range(z_scores.shape[1])
    ])

    # residuals_test[k] predicts labels_test[k + input_step] (batch_generater's windowing).
    labels_aligned = labels_test[input_step:]
    k_all = topk(z_scores, labels_aligned, pot_thresholds)
    k_at_step_all = topk_at_step(z_scores, labels_aligned)

    return {
        'ac@1': k_at_step_all[0], 'ac@3': k_at_step_all[2], 'ac@5': k_at_step_all[4],
        'ac@10': k_at_step_all[9], 'avg@10': np.mean(k_at_step_all),
        'ac*@1': k_all[0], 'ac*@10': k_all[9], 'ac*@100': k_all[99],
        'ac*@500': k_all[min(499, len(k_all) - 1)], 'avg*@500': np.mean(k_all),
    }


def fit_pot_thresholds(z_scores_val: np.ndarray, risk, initial_level, num_candidates,
                        observed: np.ndarray = None, min_observed: int = 50):
    """Unsupervised: fits a per-variable POT/EVT threshold from a validation (known-normal) z-score
    distribution. Unlike root_cause_analysis (which mirrors AERCA's benchmark convention of fitting
    POT on the evaluation window itself), this fits on held-out normal data and is meant to be
    applied to new data afterwards - the right shape for real deployment with no labels. If
    `observed` is given, only genuinely-observed z-scores are used per column - a column with fewer
    than `min_observed` of those (not enough for a meaningful tail fit; pot() can crash on very small
    samples), or whose median/std came out NaN (see fit_residual_thresholds) making every z-score in
    it non-finite, gets threshold=NaN, which score_session/flags treats as "never flags" rather than
    crashing (a channel with almost no validation data can't have anomalies meaningfully detected)."""
    thresholds = np.full(z_scores_val.shape[1], np.nan)
    skipped_nonfinite = []
    for i in range(z_scores_val.shape[1]):
        raw_col = z_scores_val[observed[:, i] > 0, i] if observed is not None else z_scores_val[:, i]
        col = raw_col[np.isfinite(raw_col)]
        if len(col) < min_observed:
            if len(col) < len(raw_col):
                skipped_nonfinite.append(i)
            continue
        thresholds[i] = pot(col, risk, initial_level, num_candidates)[0]
    if skipped_nonfinite:
        print(f'{len(skipped_nonfinite)} channel(s) had non-finite (NaN/Inf) validation z-scores '
              f'(column index: {skipped_nonfinite}) - skipped rather than crashing pot().')
    return thresholds


def score_session(multicad: MultiCAD, data_2d: np.ndarray, median, std, pot_thresholds, device,
                   mask: np.ndarray = None, chunk_size: int = 512):
    """Scores one (already-normalized) session against thresholds fit on validation data. Returns
    (z_scores, flags, observed) with flags[t, i] True where variable i's residual at time t exceeds
    its POT threshold - i.e. a candidate anomaly, with no label to check it against. `observed[t, i]`
    is 1 where that position was a real (not imputed) reading - callers should ignore flags/z_scores
    where observed is 0."""
    residuals, observed = predict_residuals(multicad, data_2d, device, mask=mask, chunk_size=chunk_size)
    std_safe = np.where(std == 0, 1e-8, std)
    z_scores = (residuals - median) / std_safe
    flags = z_scores > pot_thresholds[None, :]
    return z_scores, flags, observed


def causal_discovery_eval(graph: np.ndarray, causal_struct_value: np.ndarray, causal_quantile: float):
    auroc, auprc = eval_causal_structure(a_true=causal_struct_value, a_pred=graph)
    q = np.quantile(graph, q=causal_quantile)
    binary = (graph >= q).astype(float)
    _, _, _, _, hamming = eval_causal_structure_binary(a_true=causal_struct_value, a_pred=binary)
    f1 = f1_score(causal_struct_value.flatten(), binary.flatten())
    return {'auroc': auroc, 'auprc': auprc, 'f1': f1, 'hamming': hamming}


def plot_labeled_adjacency(graph: np.ndarray, channel_names: list, dpi: int = 300):
    """Heatmap of the discovered causal adjacency matrix (rows=effect, cols=cause) with channel
    names as axis tick labels, for logging to TensorBoard. Sized and rendered so labels stay
    legible even at ~100+ channels: figure size scales with channel count (not capped small),
    font size has a readable floor rather than shrinking to near-illegibility, and dpi is bumped
    well above matplotlib's 100dpi default."""
    n = graph.shape[0]
    side = max(8.0, n * 0.18)  # inches - grows with channel count instead of capping out
    fig, ax = plt.subplots(figsize=(side, side), dpi=dpi, layout='constrained')
    im = ax.imshow(graph, cmap='magma', vmin=float(np.min(graph)), vmax=float(np.max(graph)))
    tick_fontsize = max(7, min(11, 1400 / max(n, 1)))  # floor of 7pt - readable, not the old 4pt floor
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(channel_names, rotation=90, fontsize=tick_fontsize)
    ax.set_yticklabels(channel_names, fontsize=tick_fontsize)
    ax.set_xlabel('cause')
    ax.set_ylabel('effect')
    ax.set_title('Discovered causal adjacency (effect <- cause)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def run_pipeline(config: CUTSPlusRCAConfig, log_dir_name: str = 'cuts_plus_rca'):
    """Runs the full train -> causal-discovery-eval -> root-cause-analysis pipeline and returns
    (multicad, graph, causal_metrics, rc_metrics). Separated from main() so tests can call it
    directly without going through argparse."""
    set_seed(config.seed)
    device = config.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    x, causal_struct_value, labels, clean_len = make_synthetic_series(config)
    val_len = int(config.val_ratio * len(x))
    train_len = clean_len - val_len

    train_data, val_data, test_data = x[:train_len], x[train_len:clean_len], x[clean_len:]
    test_labels = labels[clean_len:]

    train_mean, train_std = train_data.mean(axis=0), train_data.std(axis=0)
    train_std_safe = np.where(train_std == 0, 1.0, train_std)

    def normalize(d):
        return (d - train_mean) / train_std_safe

    n_nodes = x.shape[1]
    opt = build_opt(config, n_nodes)
    log = MyLogger(log_dir=os.path.join(os.getcwd(), config.log_dir, log_dir_name),
                    stdout=False, stderr=False, tensorboard=True)

    print(f'Training CUTS+ on {train_len} timesteps ({n_nodes} variables) using device={device}')
    channel_names = [f'var_{i}' for i in range(n_nodes)]

    def plot_epoch_graph(epoch_i, raw_graph):
        # raw_graph is untransposed (row=cause, col=effect) - transpose to match this pipeline's
        # display convention (row=effect, col=cause), same as the final graph below.
        if (epoch_i + 1) % config.graph_plot_every == 0:
            log.log_figures(plot_labeled_adjacency(raw_graph.T, channel_names), name='causal_graph',
                             iters=epoch_i + 1)

    multicad = MultiCAD(opt, log, device=device)
    train_norm = normalize(train_data)
    train_mask = np.ones_like(train_norm)
    graph = multicad.train(train_norm[:, :, None], train_mask[:, :, None], train_norm[:, :, None],
                            true_cm=causal_struct_value, epoch_callback=plot_epoch_graph)

    log.log_figures(plot_labeled_adjacency(graph, channel_names), name='causal_graph',
                     iters=config.total_epoch)

    print('=' * 50)
    causal_metrics = causal_discovery_eval(graph, causal_struct_value, config.causal_quantile)
    print(f"Causal discovery F1: {causal_metrics['f1']:.5f}")
    print(f"Causal discovery AUROC: {causal_metrics['auroc']:.5f}")
    print(f"Causal discovery AUPRC: {causal_metrics['auprc']:.5f}")
    print(f"Causal discovery Hamming Distance: {causal_metrics['hamming']:.5f}")
    for k, v in causal_metrics.items():
        log.log_metrics({f'test/causal_{k}': float(v)}, config.total_epoch)

    val_residuals, _ = predict_residuals(multicad, normalize(val_data), device,
                                          chunk_size=config.predict_chunk_size)
    median, std = fit_residual_thresholds(val_residuals)

    test_residuals, _ = predict_residuals(multicad, normalize(test_data), device,
                                           chunk_size=config.predict_chunk_size)
    rc_metrics = root_cause_analysis(test_residuals, test_labels, median, std,
                                      config.risk, config.initial_level, config.num_candidates,
                                      config.input_step)
    print('=' * 50)
    print(f"Root cause analysis AC@1: {rc_metrics['ac@1']:.5f}")
    print(f"Root cause analysis AC@3: {rc_metrics['ac@3']:.5f}")
    print(f"Root cause analysis AC@5: {rc_metrics['ac@5']:.5f}")
    print(f"Root cause analysis AC@10: {rc_metrics['ac@10']:.5f}")
    print(f"Root cause analysis Avg@10: {rc_metrics['avg@10']:.5f}")
    print(f"Root cause analysis AC*@1: {rc_metrics['ac*@1']:.5f}")
    print(f"Root cause analysis AC*@10: {rc_metrics['ac*@10']:.5f}")
    print(f"Root cause analysis AC*@100: {rc_metrics['ac*@100']:.5f}")
    print(f"Root cause analysis AC*@500: {rc_metrics['ac*@500']:.5f}")
    print(f"Root cause analysis Avg*@500: {rc_metrics['avg*@500']:.5f}")
    for k, v in rc_metrics.items():
        log.log_metrics({f'test/root_cause_{k}': float(v)}, config.total_epoch)

    save_dir = os.path.join(os.getcwd(), config.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(multicad.fitting_model.state_dict(), os.path.join(save_dir, 'cuts_plus_fitting_model.pt'))
    np.save(os.path.join(save_dir, 'cuts_plus_graph.npy'), graph)
    np.save(os.path.join(save_dir, 'cuts_plus_residual_median.npy'), median)
    np.save(os.path.join(save_dir, 'cuts_plus_residual_std.npy'), std)

    log.close()
    return multicad, graph, causal_metrics, rc_metrics


def run_real_data_pipeline(config: CUTSPlusRCAConfig, log_dir_name: str = 'cuts_plus_rca_real'):
    """Real-data counterpart to run_pipeline: no ground truth, so instead of computing AUROC/AC@k
    against labels that don't exist, it trains on all sessions but the most recent val_ratio/
    test_ratio fractions (held out as whole sessions, never sliced, so nothing straddles a
    train/val/score boundary), fits residual + POT thresholds on the validation sessions, and scores
    the held-out sessions - saving flagged anomalies and the learned causal graph to save_dir."""
    set_seed(config.seed)
    device = config.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    series_dict, means, stds = load_real_sessions(config.data_dir)
    session_ids = sorted(series_dict.keys())
    n_score = max(1, round(config.test_ratio * len(session_ids)))
    n_val = max(1, round(config.val_ratio * len(session_ids)))
    if len(session_ids) <= n_val + n_score:
        raise ValueError(
            f'Only {len(session_ids)} sessions found in {config.data_dir}; need more than '
            f'val+score ({n_val}+{n_score}) so at least one whole session is left for training.'
        )
    train_ids = session_ids[:-(n_val + n_score)]
    val_ids = session_ids[-(n_val + n_score):-n_score]
    score_ids = session_ids[-n_score:]
    print(f'Sessions: {len(train_ids)} train, {len(val_ids)} val, {len(score_ids)} score '
          f'(out of {len(session_ids)} total)')

    channel_names = list(series_dict[session_ids[0]].columns)

    train_data, train_mask = concat_sessions(series_dict, train_ids, boundary_gap=config.input_step)

    # A channel with zero real observations across all training sessions has no train_mean, so
    # fill_missing can't fill it (NaN stays NaN), which poisons the whole network's gradients within
    # a step or two. Drop any such channel entirely - nothing can be learned about (or from) it if
    # training never observed it once.
    fully_missing = train_mask.sum(axis=0) == 0
    if fully_missing.any():
        dropped = [c for c, drop in zip(channel_names, fully_missing) if drop]
        print(f'Dropping {len(dropped)} channel(s) with zero observations across all training '
              f'sessions: {dropped}')
        channel_names = [c for c, drop in zip(channel_names, fully_missing) if not drop]
        train_data = train_data[:, ~fully_missing]
        train_mask = train_mask[:, ~fully_missing]

    n_nodes = len(channel_names)
    boundary_cells = config.input_step * n_nodes * (len(train_ids) - 1)
    n_missing = (train_mask == 0).sum() - boundary_cells  # exclude boundary gaps, count only real NaN
    if n_missing > 0:
        print(f'{n_missing} / {train_data.size} training values are missing (NaN) - masked out of '
              f'training and filled with each channel\'s mean.')

    if means is not None and stds is not None:
        train_mean = means.reindex(channel_names).values.astype(np.float32)
        train_std = stds.reindex(channel_names).values.astype(np.float32)
    else:
        # nanmean/nanstd: train_data still has real NaN in it here, not yet filled.
        train_mean, train_std = np.nanmean(train_data, axis=0), np.nanstd(train_data, axis=0)
    train_std_safe = np.where(train_std == 0, 1.0, train_std)

    def normalize(d):
        return (fill_missing(d, train_mean) - train_mean) / train_std_safe

    opt = build_opt(config, n_nodes)
    log = MyLogger(log_dir=os.path.join(os.getcwd(), config.log_dir, log_dir_name),
                    stdout=False, stderr=False, tensorboard=True)
    log.log_metrics({'data/n_channels_dropped': int(fully_missing.sum()),
                      'data/n_channels_used': n_nodes,
                      'data/train_missing_fraction': float(n_missing / train_data.size)}, 0)

    print(f'Training CUTS+ on {train_data.shape[0]} timesteps across {len(train_ids)} sessions '
          f'({n_nodes} variables) using device={device}')

    def plot_epoch_graph(epoch_i, raw_graph):
        if (epoch_i + 1) % config.graph_plot_every == 0:
            log.log_figures(plot_labeled_adjacency(raw_graph.T, channel_names), name='causal_graph',
                             iters=epoch_i + 1)

    multicad = MultiCAD(opt, log, device=device)
    train_norm = normalize(train_data)
    # true_cm=None (no ground truth) - so, unlike run_pipeline, the graph MultiCAD.train() returns is
    # NOT auto-transposed into "row=effect, col=cause" convention. Transpose it ourselves below.
    graph = multicad.train(train_norm[:, :, None], train_mask[:, :, None], train_norm[:, :, None],
                            true_cm=None, epoch_callback=plot_epoch_graph)
    graph = graph.T

    log.log_figures(plot_labeled_adjacency(graph, channel_names), name='causal_graph',
                     iters=config.total_epoch)

    val_pairs = []
    for vid in val_ids:
        filled, mask = session_data_and_mask(series_dict[vid][channel_names])
        val_pairs.append(predict_residuals(multicad, normalize(filled), device, mask=mask,
                                            chunk_size=config.predict_chunk_size))
    val_residuals = np.concatenate([r for r, _ in val_pairs], axis=0)
    val_observed = np.concatenate([o for _, o in val_pairs], axis=0)
    median, std = fit_residual_thresholds(val_residuals, observed=val_observed)
    val_z = (val_residuals - median) / np.where(std == 0, 1e-8, std)
    pot_thresholds = fit_pot_thresholds(val_z, config.risk, config.initial_level, config.num_candidates,
                                         observed=val_observed)
    unscoreable = [c for c, t in zip(channel_names, pot_thresholds) if np.isnan(t)]
    if unscoreable:
        print(f'{len(unscoreable)} channel(s) have too little validation data to set an anomaly '
              f'threshold, so they will never be flagged: {unscoreable}')

    save_dir = os.path.join(os.getcwd(), config.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print('=' * 50)
    print('Scoring held-out sessions (no ground truth - reporting flagged fractions, not accuracy):')
    for sid in score_ids:
        filled, mask = session_data_and_mask(series_dict[sid][channel_names])
        z_scores, flags, observed = score_session(multicad, normalize(filled), median, std, pot_thresholds,
                                                    device, mask=mask, chunk_size=config.predict_chunk_size)
        flagged_fraction = flags[observed > 0].mean() if observed.sum() > 0 else float('nan')
        per_var_fraction = np.array([
            flags[observed[:, c] > 0, c].mean() if observed[:, c].sum() > 0 else np.nan
            for c in range(flags.shape[1])
        ])
        top_vars = np.argsort(np.nan_to_num(per_var_fraction, nan=-1))[::-1][:5]
        top_str = ', '.join(f'{channel_names[i]} ({per_var_fraction[i] * 100:.1f}%)' for i in top_vars)
        print(f'  {sid}: {flagged_fraction * 100:.2f}% of observed (timestep, variable) pairs flagged; '
              f'most-flagged: {top_str}')
        log.log_metrics({f'test/score_{sid}_flagged_fraction': float(flagged_fraction)}, config.total_epoch)
        np.save(os.path.join(save_dir, f'cuts_plus_score_{sid}_z.npy'), z_scores)
        np.save(os.path.join(save_dir, f'cuts_plus_score_{sid}_flags.npy'), flags)
        np.save(os.path.join(save_dir, f'cuts_plus_score_{sid}_observed.npy'), observed)

    edge_strength = graph.copy()
    np.fill_diagonal(edge_strength, 0.0)
    flat_idx = np.argsort(edge_strength.ravel())[::-1][:15]
    rows, cols = np.unravel_index(flat_idx, edge_strength.shape)
    print('=' * 50)
    print('Strongest discovered causal edges (effect <- cause):')
    for i, j in zip(rows, cols):
        print(f'  {channel_names[i]} <- {channel_names[j]}: {edge_strength[i, j]:.4f}')

    torch.save(multicad.fitting_model.state_dict(), os.path.join(save_dir, 'cuts_plus_fitting_model.pt'))
    np.save(os.path.join(save_dir, 'cuts_plus_graph.npy'), graph)
    np.save(os.path.join(save_dir, 'cuts_plus_residual_median.npy'), median)
    np.save(os.path.join(save_dir, 'cuts_plus_residual_std.npy'), std)
    np.save(os.path.join(save_dir, 'cuts_plus_pot_thresholds.npy'), pot_thresholds)
    # channel_names may differ from the dataset's raw column list (see the fully_missing drop above),
    # so the graph's row/column order can't always be recovered from the dataset alone.
    with open(os.path.join(save_dir, 'channel_names.json'), 'w') as fh:
        json.dump(channel_names, fh, indent=2)

    log.close()
    return multicad, graph


def main():
    config = CUTSPlusRCAConfig(**vars(build_arg_parser().parse_args()))
    if config.data_dir:
        run_real_data_pipeline(config)
    else:
        run_pipeline(config)


if __name__ == '__main__':
    main()
