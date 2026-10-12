"""
Compares fill_policy sweep runs (see sweep_fill_policy.sh) on the diagnostics used to investigate the
`1974` and `imputation` runs by hand: whether S2 (graph_discov/data_loss) tracks or diverges from S1
(latent_data_pred/pred_loss), and whether channels' self-edges (the discovered graph's diagonal) hold
up or erode into spurious "hub" edges. Reads each variant's TensorBoard event file and
cuts_plus_graph.npy/channel_names.json directly - no training code involved, so it's safe to run
against in-progress runs too (anything not yet written is reported as incomplete).

Usage:
    python3 cuts_plus_prototype/compare_fill_policies.py --sweep-root cuts_plus_prototype/scratch/fill_policy_sweep
"""
import argparse
import glob
import json
import os

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tail_mean(ea: EventAccumulator, tag: str, tail_frac: float = 0.1) -> float:
    """Mean of a scalar's last tail_frac of logged points - less noisy than the single final point,
    since each logged point is one (un-averaged) batch's value."""
    if tag not in ea.Tags()['scalars']:
        return float('nan')
    vals = np.array([e.value for e in ea.Scalars(tag)])
    tail = max(1, int(len(vals) * tail_frac))
    return float(vals[-tail:].mean())


def summarize_run(run_dir: str) -> dict:
    graph_path = os.path.join(run_dir, 'models', 'cuts_plus_graph.npy')
    names_path = os.path.join(run_dir, 'models', 'channel_names.json')
    if not os.path.exists(graph_path) or not os.path.exists(names_path):
        return {'status': 'incomplete'}

    graph = np.load(graph_path)  # row=effect, col=cause
    with open(names_path) as f:
        n_channels = len(json.load(f))

    diag = np.diagonal(graph)
    off_diag = graph.copy()
    np.fill_diagonal(off_diag, 0.0)
    incoming_mass = off_diag.sum(axis=1)  # per-effect total mass from every OTHER channel
    diag_vs_hub_corr = (float(np.corrcoef(diag, incoming_mass)[0, 1])
                         if diag.std() > 0 and incoming_mass.std() > 0 else float('nan'))

    summary = {
        'status': 'complete',
        'n_channels': n_channels,
        'diag_mean': float(diag.mean()),
        'diag_eroded_frac': float((diag < 0.1).mean()),  # fraction of channels that lost their self-edge
        'diag_vs_hub_corr': diag_vs_hub_corr,  # weak self-edge -> more spurious incoming edges, if positive
    }

    event_files = glob.glob(os.path.join(run_dir, 'runs', '**', 'events.out.tfevents.*'), recursive=True)
    if event_files:
        ea = EventAccumulator(os.path.dirname(event_files[0]))
        ea.Reload()
        summary['s1_pred_loss'] = tail_mean(ea, 'latent_data_pred/pred_loss')
        summary['s2_data_loss'] = tail_mean(ea, 'graph_discov/data_loss')
        summary['sparsity_loss'] = tail_mean(ea, 'graph_discov/sparsity_loss')
        summary['s2_over_s1'] = (summary['s2_data_loss'] / summary['s1_pred_loss']
                                  if summary['s1_pred_loss'] else float('nan'))
    return summary


def print_table(rows: list):
    cols = ['fill_policy', 'status', 's1_pred_loss', 's2_data_loss', 's2_over_s1', 'sparsity_loss',
            'diag_mean', 'diag_eroded_frac', 'diag_vs_hub_corr']

    def fmt(v):
        if isinstance(v, float):
            return 'nan' if np.isnan(v) else f'{v:.4f}'
        return str(v)

    cells = [[fmt(r.get(c, '')) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    print('  '.join(c.ljust(w) for c, w in zip(cols, widths)))
    for row in cells:
        print('  '.join(v.ljust(w) for v, w in zip(row, widths)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sweep-root', required=True)
    args = p.parse_args()

    policies = sorted(d for d in os.listdir(args.sweep_root)
                       if os.path.isdir(os.path.join(args.sweep_root, d)))
    if not policies:
        raise ValueError(f'No run directories found under {args.sweep_root}')

    rows = []
    for policy in policies:
        summary = summarize_run(os.path.join(args.sweep_root, policy))
        summary['fill_policy'] = policy
        rows.append(summary)

    print_table(rows)


if __name__ == '__main__':
    main()
