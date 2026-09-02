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

from cuts_plus import MultiCAD, batch_generater  # noqa: E402  (vendor/cuts_plus.py)
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
    weight_decay: float = 0.0
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
    that zeroes out the first `boundary_gap` timesteps of every session after the first, so CUTS+
    (which natively supports missing/masked data) never trains on a window whose input reaches back
    into a different, unrelated session."""
    arrays = [series_dict[sid].values.astype(np.float32) for sid in session_ids]
    data = np.concatenate(arrays, axis=0)
    mask = np.ones_like(data)
    offset = 0
    for arr in arrays[:-1]:
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


def predict_residuals(multicad: MultiCAD, data: np.ndarray, device) -> np.ndarray:
    """One-step-ahead prediction residuals (actual - predicted) for every valid window in `data`,
    using the trained fitting_model and its current learned graph, in time order. `data` is 2D
    (T, N), already normalized the same way the training data was."""
    n_nodes = data.shape[1]
    input_step = multicad.args.input_step
    data3 = data[:, :, None]
    mask3 = np.ones_like(data3)
    data_t = torch.from_numpy(data3).float().to(device)
    mask_t = torch.from_numpy(mask3).float().to(device)

    t_length = data.shape[0]
    bs = t_length - input_step  # one batch covering every valid window, none dropped
    x, y, t_idx, mask_x, mask_y = next(batch_generater(
        data_t, mask_t, bs=bs, n_nodes=n_nodes, input_step=input_step, pred_step=1, block_size=None))

    # Recompute the (untransposed, source->target) edge-weight matrix the network was actually
    # trained with - NOT the value returned by MultiCAD.train(), which gets transposed once for
    # comparison against true_cm and is not the convention the forward pass expects.
    graph = torch.einsum('nm,ml->nl', multicad.G, torch.sigmoid(multicad.GT))
    graph_expanded = graph[None].expand(x.shape[0], -1, -1)

    multicad.fitting_model.eval()
    with torch.no_grad():
        y_pred = multicad.fitting_model(x, mask_x, graph_expanded)

    residual = (y - y_pred).squeeze(-1).squeeze(-1).cpu().numpy()
    order = np.argsort(t_idx.cpu().numpy())
    return residual[order]


def fit_residual_thresholds(residuals: np.ndarray):
    return np.median(residuals, axis=0), np.std(residuals, axis=0)


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


def fit_pot_thresholds(z_scores_val: np.ndarray, risk, initial_level, num_candidates):
    """Unsupervised: fits a per-variable POT/EVT threshold from a validation (known-normal) z-score
    distribution. Unlike root_cause_analysis (which mirrors AERCA's benchmark convention of fitting
    POT on the evaluation window itself), this fits on held-out normal data and is meant to be
    applied to new data afterwards - the right shape for real deployment with no labels."""
    return np.array([
        pot(z_scores_val[:, i], risk, initial_level, num_candidates)[0]
        for i in range(z_scores_val.shape[1])
    ])


def score_session(multicad: MultiCAD, data_2d: np.ndarray, median, std, pot_thresholds, device):
    """Scores one (already-normalized) session against thresholds fit on validation data. Returns
    (z_scores, flags) with flags[t, i] True where variable i's residual at time t exceeds its POT
    threshold - i.e. a candidate anomaly, with no label to check it against."""
    residuals = predict_residuals(multicad, data_2d, device)
    std_safe = np.where(std == 0, 1e-8, std)
    z_scores = (residuals - median) / std_safe
    flags = z_scores > pot_thresholds[None, :]
    return z_scores, flags


def causal_discovery_eval(graph: np.ndarray, causal_struct_value: np.ndarray, causal_quantile: float):
    auroc, auprc = eval_causal_structure(a_true=causal_struct_value, a_pred=graph)
    q = np.quantile(graph, q=causal_quantile)
    binary = (graph >= q).astype(float)
    _, _, _, _, hamming = eval_causal_structure_binary(a_true=causal_struct_value, a_pred=binary)
    f1 = f1_score(causal_struct_value.flatten(), binary.flatten())
    return {'auroc': auroc, 'auprc': auprc, 'f1': f1, 'hamming': hamming}


def plot_labeled_adjacency(graph: np.ndarray, channel_names: list):
    """Heatmap of the discovered causal adjacency matrix (rows=effect, cols=cause) with channel
    names as axis tick labels, for logging to TensorBoard."""
    n = graph.shape[0]
    side = max(6.0, min(30.0, n * 0.3))
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(graph, cmap='magma', vmin=float(np.min(graph)), vmax=float(np.max(graph)))
    tick_fontsize = max(4, min(9, int(300 / max(n, 1))))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(channel_names, rotation=90, fontsize=tick_fontsize)
    ax.set_yticklabels(channel_names, fontsize=tick_fontsize)
    ax.set_xlabel('cause')
    ax.set_ylabel('effect')
    ax.set_title('Discovered causal adjacency (effect <- cause)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
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

    val_residuals = predict_residuals(multicad, normalize(val_data), device)
    median, std = fit_residual_thresholds(val_residuals)

    test_residuals = predict_residuals(multicad, normalize(test_data), device)
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
    n_nodes = len(channel_names)

    train_data, train_mask = concat_sessions(series_dict, train_ids, boundary_gap=config.input_step)

    if means is not None and stds is not None:
        train_mean = means.reindex(channel_names).values.astype(np.float32)
        train_std = stds.reindex(channel_names).values.astype(np.float32)
    else:
        train_mean, train_std = train_data.mean(axis=0), train_data.std(axis=0)
    train_std_safe = np.where(train_std == 0, 1.0, train_std)

    def normalize(d):
        return (d - train_mean) / train_std_safe

    opt = build_opt(config, n_nodes)
    log = MyLogger(log_dir=os.path.join(os.getcwd(), config.log_dir, log_dir_name),
                    stdout=False, stderr=False, tensorboard=True)

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

    val_residuals = np.concatenate([
        predict_residuals(multicad, normalize(series_dict[vid].values.astype(np.float32)), device)
        for vid in val_ids
    ], axis=0)
    median, std = fit_residual_thresholds(val_residuals)
    val_z = (val_residuals - median) / np.where(std == 0, 1e-8, std)
    pot_thresholds = fit_pot_thresholds(val_z, config.risk, config.initial_level, config.num_candidates)

    save_dir = os.path.join(os.getcwd(), config.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print('=' * 50)
    print('Scoring held-out sessions (no ground truth - reporting flagged fractions, not accuracy):')
    for sid in score_ids:
        data = series_dict[sid].values.astype(np.float32)
        z_scores, flags = score_session(multicad, normalize(data), median, std, pot_thresholds, device)
        flagged_fraction = flags.mean()
        per_var_fraction = flags.mean(axis=0)
        top_vars = np.argsort(per_var_fraction)[::-1][:5]
        top_str = ', '.join(f'{channel_names[i]} ({per_var_fraction[i] * 100:.1f}%)' for i in top_vars)
        print(f'  {sid}: {flagged_fraction * 100:.2f}% of (timestep, variable) pairs flagged; '
              f'most-flagged: {top_str}')
        np.save(os.path.join(save_dir, f'cuts_plus_score_{sid}_z.npy'), z_scores)
        np.save(os.path.join(save_dir, f'cuts_plus_score_{sid}_flags.npy'), flags)

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
