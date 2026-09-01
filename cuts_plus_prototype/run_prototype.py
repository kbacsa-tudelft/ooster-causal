"""
Throwaway benchmark: compares AERCA (this repo, ../aerca.py) against vendored CUTS+
(vendor/cuts_plus.py, from https://github.com/jarrycyx/UNN/tree/main/CUTS_Plus) on synthetic
high-channel-count data, to see whether CUTS+ is worth pursuing further before the 174-channel
real dataset arrives.

Not part of the AERCA pipeline - run manually:
    python3 cuts_plus_prototype/run_prototype.py
"""
import os
import resource
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cuts_plus import MultiCAD  # noqa: E402  (vendor/cuts_plus.py)
from utils.logger import MyLogger  # noqa: E402  (vendor/utils/logger.py)

from aerca import (  # noqa: E402
    AERCA,
    AERCAConfig,
    eval_causal_structure,
    eval_causal_structure_binary,
    make_dataloader,
    make_synthetic_dataset,
    set_seed,
    split_series_dict,
)

from cuts_plus_rca import CUTSPlusRCAConfig, run_pipeline as run_cuts_plus_rca_pipeline  # noqa: E402

from sklearn.metrics import f1_score


def peak_rss_mb() -> float:
    """Process peak resident set size so far, in MB. Monotonically non-decreasing since process
    start, so calling it right after a training run approximates that run's peak (this script
    doesn't do anything else memory-heavy beforehand)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def summarize_causal_estimate(estimate: np.ndarray, causal_struct_value: np.ndarray, quantile: float = 0.80):
    auroc, auprc = eval_causal_structure(a_true=causal_struct_value, a_pred=estimate)
    q = np.quantile(estimate, q=quantile)
    binary = (estimate >= q).astype(float)
    _, _, _, _, hamming = eval_causal_structure_binary(a_true=causal_struct_value, a_pred=binary)
    f1 = f1_score(causal_struct_value.flatten(), binary.flatten())
    return {'auroc': auroc, 'auprc': auprc, 'f1': f1, 'hamming': hamming}


def run_aerca_leg(name: str, num_vars: int, series_len: int, epochs: int, chunk_len: int,
                   window_size: int, hidden_layer_size: int, save_dir: str, log_dir: str):
    print(f'\n=== {name} ===')
    config = AERCAConfig(
        hidden_layer_size=hidden_layer_size,
        num_hidden_layers=2,
        window_size=window_size,
        epochs=epochs,
        patience=epochs,
        chunk_len=chunk_len,
        val_ratio=0.2,
        synthetic_num_vars=num_vars,
        synthetic_num_series=3,
        synthetic_series_len=series_len,
        synthetic_num_anomalies=5,
        save_dir=save_dir,
        log_dir=log_dir,
        seed=0,
    )
    set_seed(config.seed)

    series_dict, label_dict, causal_struct_value = make_synthetic_dataset(config)
    test_dict = {'series_test': series_dict.pop('series_test')}
    train_dict, val_dict = split_series_dict(series_dict, val_ratio=config.val_ratio, seed=config.seed)

    train_loader = make_dataloader(train_dict, shuffle=True)
    val_loader = make_dataloader(val_dict, shuffle=False)
    test_loader = make_dataloader(test_dict, label_dict=label_dict, shuffle=False)

    model = AERCA(num_vars=num_vars, device=torch.device('cpu'), config=config)

    start = time.time()
    try:
        model._training(train_loader, val_loader)
        elapsed = time.time() - start

        encoder_causal_list = []
        model.eval()
        with torch.no_grad():
            for x, _, _ in test_loader:
                _, _, _, encoder_coeffs, _, _, _, _ = model._testing_step(x)
                estimate = torch.max(torch.median(torch.abs(encoder_coeffs), dim=0)[0],
                                      dim=0).values.cpu().numpy()
                encoder_causal_list.append(estimate)
        estimate = np.mean(np.stack(encoder_causal_list, axis=0), axis=0)
        metrics = summarize_causal_estimate(estimate, causal_struct_value, config.causal_quantile)
        rc_metrics = model._testing_root_cause(test_loader)
        model.writer.close()

        return {'name': name, 'ok': True, 'time_s': elapsed, 'peak_rss_mb': peak_rss_mb(),
                **metrics, **rc_metrics}
    except Exception as e:  # noqa: BLE001 - benchmark leg, report and move on
        elapsed = time.time() - start
        return {'name': name, 'ok': False, 'time_s': elapsed, 'peak_rss_mb': peak_rss_mb(), 'error': str(e)}


def make_cuts_plus_data(num_vars: int, series_len: int, edge_prob: float = 0.1, seed: int = 0):
    """Single (T, N) series + (N, N) ground-truth adjacency, same generative idea as
    aerca.make_synthetic_dataset (sparse random VAR process), shaped for CUTS+'s single-series API."""
    rng = np.random.default_rng(seed)
    order = 4
    causal_struct_value = (rng.random((num_vars, num_vars)) < edge_prob).astype(float)
    np.fill_diagonal(causal_struct_value, 1.0)
    coeffs = [causal_struct_value * rng.uniform(-0.3, 0.3, size=(num_vars, num_vars)) / order for _ in range(order)]

    x = np.zeros((series_len, num_vars))
    x[:order] = rng.normal(scale=0.1, size=(order, num_vars))
    for t in range(order, series_len):
        value = sum(coeffs[k] @ x[t - k - 1] for k in range(order))
        value += rng.normal(scale=0.1, size=num_vars)
        x[t] = value

    mask = np.ones_like(x)
    # CUTS+ expects true_cm[i, j] to mean j -> i (see rearrange("n m -> m n") in cuts_plus.train);
    # our causal_struct_value follows AERCA's convention (row i caused-by column j), so transpose.
    return x.astype(np.float32), mask.astype(np.float32), causal_struct_value.T


def run_cuts_plus_leg(name: str, num_vars: int, series_len: int, total_epoch: int, log_dir: str):
    print(f'\n=== {name} ===')
    data, mask, true_cm = make_cuts_plus_data(num_vars, series_len, seed=0)

    opt = OmegaConf.create({
        'n_nodes': 'auto',
        'input_step': 1,
        'batch_size': 128,
        'data_dim': 1,
        'total_epoch': total_epoch,
        'n_groups': min(32, num_vars),
        'group_policy': f'multiply_2_every_{max(1, total_epoch // 4)}',
        'supervision_policy': 'full',
        'fill_policy': 'none',
        'show_graph_every': max(1, total_epoch),  # only plot at the end, keep it cheap
        'data_pred': {
            'pred_step': 1,
            'mlp_hid': 32,
            'gru_layers': 1,
            'shared_weights_decoder': False,
            'concat_h': True,
            'lr_data_start': 1e-2,
            'lr_data_end': 1e-3,
            'weight_decay': 0,
        },
        'graph_discov': {
            'lambda_s_start': 1e-1,
            'lambda_s_end': 1e-2,
            'lr_graph_start': 1e-3,
            'lr_graph_end': 1e-4,
            'start_tau': 1,
            'end_tau': 0.1,
        },
    })

    log = MyLogger(log_dir=log_dir, stdout=False, stderr=False, tensorboard=True)

    start = time.time()
    try:
        from cuts_plus import main as cuts_plus_main
        graph = cuts_plus_main(data, mask, true_cm, opt, log, device='cpu')
        elapsed = time.time() - start
        log.close()

        metrics = summarize_causal_estimate(graph, true_cm.T, quantile=0.80)
        return {'name': name, 'ok': True, 'time_s': elapsed, 'peak_rss_mb': peak_rss_mb(), **metrics}
    except Exception as e:  # noqa: BLE001 - benchmark leg, report and move on
        elapsed = time.time() - start
        log.close()
        return {'name': name, 'ok': False, 'time_s': elapsed, 'peak_rss_mb': peak_rss_mb(), 'error': str(e)}


def run_cuts_plus_rca_leg(name: str, num_vars: int, series_len: int, total_epoch: int,
                           save_dir: str, log_dir: str):
    print(f'\n=== {name} ===')
    config = CUTSPlusRCAConfig(
        total_epoch=total_epoch,
        n_groups=min(32, num_vars),
        group_policy=f'multiply_2_every_{max(1, total_epoch // 4)}',
        show_graph_every=max(1, total_epoch),
        synthetic_num_vars=num_vars,
        synthetic_series_len=series_len,
        synthetic_num_anomalies=10,
        save_dir=save_dir,
        log_dir=os.path.dirname(log_dir),
        seed=0,
    )

    start = time.time()
    try:
        _, graph, causal_metrics, rc_metrics = run_cuts_plus_rca_pipeline(config, log_dir_name=os.path.basename(log_dir))
        elapsed = time.time() - start
        return {'name': name, 'ok': True, 'time_s': elapsed, 'peak_rss_mb': peak_rss_mb(),
                **causal_metrics, **rc_metrics}
    except Exception as e:  # noqa: BLE001 - benchmark leg, report and move on
        elapsed = time.time() - start
        return {'name': name, 'ok': False, 'time_s': elapsed, 'peak_rss_mb': peak_rss_mb(), 'error': str(e)}


def print_report(results):
    print('\n' + '=' * 100)
    print(f'{"leg":32s} {"ok":4s} {"time(s)":>9s} {"peakRSS(MB)":>12s} {"AUROC":>7s} {"AUPRC":>7s} '
          f'{"F1":>6s} {"RC_AC@1":>8s} {"RC_AC@10":>9s} {"RC_Avg@10":>10s}')
    for r in results:
        if r['ok']:
            has_rc = 'ac@1' in r
            rc_str = (f'{r["ac@1"]:8.3f} {r["ac@10"]:9.3f} {r["avg@10"]:10.3f}' if has_rc
                      else f'{"n/a":>8s} {"n/a":>9s} {"n/a":>10s}')
            print(f'{r["name"]:32s} {"yes":4s} {r["time_s"]:9.1f} {r["peak_rss_mb"]:12.1f} '
                  f'{r["auroc"]:7.3f} {r["auprc"]:7.3f} {r["f1"]:6.3f} {rc_str}')
        else:
            print(f'{r["name"]:32s} {"NO":4s} {r["time_s"]:9.1f} {r["peak_rss_mb"]:12.1f}   error: {r["error"]}')


SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scratch')

LEGS = {
    'aerca_naive': lambda: run_aerca_leg(
        'AERCA naive (chunk_len=1000)', num_vars=174, series_len=1000, epochs=2, chunk_len=1000,
        window_size=8, hidden_layer_size=32,
        save_dir=os.path.join(SCRATCH, 'aerca_naive', 'models'),
        log_dir=os.path.join(SCRATCH, 'aerca_naive', 'runs'),
    ),
    'aerca_tuned': lambda: run_aerca_leg(
        'AERCA tuned (chunk_len=128)', num_vars=174, series_len=1000, epochs=5, chunk_len=128,
        window_size=8, hidden_layer_size=32,
        save_dir=os.path.join(SCRATCH, 'aerca_tuned', 'models'),
        log_dir=os.path.join(SCRATCH, 'aerca_tuned', 'runs'),
    ),
    # total_epoch=8 with group_policy multiply_2_every_2 reaches the full 174x174 graph
    # (32 -> 64 -> 128 -> 174) by epoch 6.
    'cuts_plus': lambda: run_cuts_plus_leg(
        'CUTS+ (causal discovery only)', num_vars=174, series_len=1000, total_epoch=8,
        log_dir=os.path.join(SCRATCH, 'cuts_plus', 'runs'),
    ),
    'cuts_plus_rca': lambda: run_cuts_plus_rca_leg(
        'CUTS+ + root-cause layer', num_vars=174, series_len=1000, total_epoch=8,
        save_dir=os.path.join(SCRATCH, 'cuts_plus_rca', 'models'),
        log_dir=os.path.join(SCRATCH, 'cuts_plus_rca', 'runs'),
    ),
}


if __name__ == '__main__':
    import json

    os.makedirs(SCRATCH, exist_ok=True)

    if len(sys.argv) > 1 and sys.argv[1] == '--report':
        results = []
        for key in LEGS:
            path = os.path.join(SCRATCH, f'{key}.json')
            if os.path.exists(path):
                with open(path) as f:
                    results.append(json.load(f))
        print_report(results)
    elif len(sys.argv) > 1 and sys.argv[1] in LEGS:
        # Each leg MUST run in its own process: resource.getrusage(...).ru_maxrss is a
        # monotonically non-decreasing high-water mark for the whole process, so running
        # multiple legs in-process contaminates every later leg's "peak memory" with the
        # max of everything that ran before it.
        result = LEGS[sys.argv[1]]()
        with open(os.path.join(SCRATCH, f'{sys.argv[1]}.json'), 'w') as f:
            json.dump(result, f)
        print_report([result])
    else:
        print(f'usage: {sys.argv[0]} [{"|".join(LEGS)}|--report]')
        sys.exit(1)
