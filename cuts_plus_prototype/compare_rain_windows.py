"""
Compares discovered RH_* -> WL_* causal edges across several rolling-window variants of the same
dataset, to pick which window size ('--rain-window' in prepare_data.py) produces the clearest,
most physically sensible rainfall-driven signal.

Water level is dominated by tidal correlation between coastal/estuary stations (WL_* <-> WL_*
edges land around 0.9+ after full training) - a much stronger signal than rainfall's effect on
water level, so the strongest *overall* edges are never rain-driven. This script restricts to
RH_* -> WL_* edges specifically, since that is the signal '--rain-window' is meant to improve, and
prints two things to judge across window sizes:
  - The top RH_* -> WL_* edges per window and their overall max/mean strength - a higher peak means
    a clearer, more confident edge.
  - Which target stations recur across independently-trained window sizes - recurrence is a much
    stronger indicator of a real effect than any single run's top edge, which could be noise. Also
    sanity-check the recurring targets by eye: real rain-driven stations should be physically
    rain-sensitive locations (small inland/polder basins, lock complexes), not coastal/tidal gauges
    (which respond to tide, not rain).

Usage:
    python3 cuts_plus_prototype/compare_rain_windows.py \
        --windows 6h 12h 24h 48h \
        --data-dir-template two_week_chunks_200_prepared_{w} \
        --graph-template cuts_plus_prototype/scratch/rain200_compare_{w}/models/cuts_plus_graph.npy \
        --top-k 10
"""
import argparse
import glob
import os
from collections import Counter

import numpy as np
import pandas as pd


def load_channel_names(data_dir: str) -> list:
    f = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))[0]
    return list(pd.read_parquet(f).columns)


def rain_edges(graph: np.ndarray, channel_names: list) -> list:
    """Returns [(weight, effect_name, cause_name), ...] for every RH_* -> WL_* edge, strongest first."""
    n = len(channel_names)
    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cause, effect = channel_names[j], channel_names[i]
            if cause.startswith('RH_') and effect.startswith('WL_'):
                pairs.append((graph[i, j], effect, cause))
    pairs.sort(key=lambda p: p[0], reverse=True)
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows', nargs='+', required=True, help='e.g. 6h 12h 24h 48h')
    parser.add_argument('--data-dir-template', required=True,
                         help='e.g. two_week_chunks_200_prepared_{w}')
    parser.add_argument('--graph-template', required=True,
                         help='e.g. cuts_plus_prototype/scratch/rain200_compare_{w}/models/'
                              'cuts_plus_graph.npy')
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args()

    top_targets_per_window = {}
    for w in args.windows:
        data_dir = args.data_dir_template.format(w=w)
        graph_path = args.graph_template.format(w=w)
        channel_names = load_channel_names(data_dir)
        graph = np.load(graph_path)
        edges = rain_edges(graph, channel_names)
        weights = np.array([e[0] for e in edges])

        print(f'\n=== window {w}: top {args.top_k} RH_* -> WL_* edges '
              f'(max={weights.max():.4f}, mean={weights.mean():.4f}) ===')
        for wgt, effect, cause in edges[:args.top_k]:
            print(f'  {effect} <- {cause}: {wgt:.4f}')
        top_targets_per_window[w] = {effect for _, effect, _ in edges[:args.top_k]}

    print('\n=== target stations recurring across multiple window sizes '
          '(stronger evidence than any single run) ===')
    counts = Counter(t for targets in top_targets_per_window.values() for t in targets)
    recurring = [(t, n) for t, n in counts.items() if n > 1]
    if not recurring:
        print('  none - no target station appears in more than one window\'s top list')
    for target, n in sorted(recurring, key=lambda kv: -kv[1]):
        print(f'  {target}: appears in {n}/{len(args.windows)} windows')


if __name__ == '__main__':
    main()
