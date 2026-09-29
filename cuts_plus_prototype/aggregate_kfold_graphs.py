"""
Aggregates K per-fold cuts_plus_graph.npy files (from a k-fold cross-validation sweep - see
CUTSPlusRCAConfig.n_folds/fold in cuts_plus_rca.py and sweep.py) into per-edge mean/std and a
recurrence count, as an empirical confidence signal for discovered edges when there's no ground
truth graph to check against: an edge that shows up as "strong" consistently across folds trained
on different (overlapping) subsets of sessions is much more likely to be a real effect than one
that's only strong in a single fold.

Usage:
    # from a sweep.py run that swept `fold` (e.g. --sweep fold=0,1,2,3,4 --n-folds 5)
    python3 cuts_plus_prototype/aggregate_kfold_graphs.py \
        --sweep-dir cuts_plus_prototype/runs/sweep/20260929_120000 \
        --output cuts_plus_prototype/scratch/kfold_agg_v1

    # or an explicit list of fold model directories (each containing cuts_plus_graph.npy +
    # channel_names.json), if the folds weren't produced by sweep.py
    python3 cuts_plus_prototype/aggregate_kfold_graphs.py \
        --fold-dirs run0/models run1/models run2/models \
        --output cuts_plus_prototype/scratch/kfold_agg_v1

Writes to --output:
    kfold_mean_graph.npy  - per-edge mean across folds (row=effect, col=cause, same convention as
                             cuts_plus_graph.npy)
    kfold_std_graph.npy   - per-edge std across folds
    kfold_recurrence.npy  - per-edge int count in [0, K]: how many folds rank this edge "strong"
                             (see --recurrence-mode)
    channel_names.json    - copied verbatim (identical across folds, enforced below) so
                             plot_causal_map.py / compare_rain_windows.py can point straight at
                             kfold_mean_graph.npy with no changes to either script
"""
import argparse
import glob
import json
import os
import re
import shutil

import numpy as np


def discover_fold_dirs(sweep_dir: str) -> list:
    """Finds every completed fold's model directory under a sweep.py sweep directory that swept
    `fold` (sweep.py's run_name() names each combo's subdirectory 'fold=<i>' when `fold` is the only
    swept field), sorted by fold index. Sweeps that vary `fold` together with other fields produce
    differently-named subdirectories - pass --fold-dirs explicitly in that case."""
    candidates = glob.glob(os.path.join(sweep_dir, 'fold=*', 'models'))
    if not candidates:
        raise ValueError(
            f'No fold=*/models directories found under {sweep_dir!r}. Was this sweep run with '
            f'--sweep fold=0,1,...? Use --fold-dirs to pass explicit directories instead.'
        )

    def fold_index(path):
        return int(re.search(r'fold=(\d+)', path).group(1))

    return sorted(candidates, key=fold_index)


def load_fold(model_dir: str):
    """Returns (graph, channel_names) for one fold, matching the file names run_real_data_pipeline
    saves (cuts_plus_graph.npy, channel_names.json)."""
    graph_path = os.path.join(model_dir, 'cuts_plus_graph.npy')
    names_path = os.path.join(model_dir, 'channel_names.json')
    if not os.path.exists(graph_path) or not os.path.exists(names_path):
        raise ValueError(f'{model_dir} is missing cuts_plus_graph.npy and/or channel_names.json - '
                          f'was this fold\'s run_real_data_pipeline run to completion?')
    graph = np.load(graph_path)
    with open(names_path) as fh:
        channel_names = json.load(fh)
    if graph.shape != (len(channel_names), len(channel_names)):
        raise ValueError(f'{model_dir}: graph shape {graph.shape} does not match '
                          f'{len(channel_names)} channel_names.')
    return graph, channel_names


def load_all_folds(model_dirs: list):
    """Loads every fold's graph, hard-erroring if any fold's channel_names differs from the first
    fold's. By construction (run_real_data_pipeline computes the sparse-channel mask once from the
    full training pool and shares it across folds), every fold of the same k-fold run must produce
    identical channel_names (same members, same order) - a mismatch here means the consistency
    guarantee was violated (or these aren't folds of the same run), and mean/std/recurrence would
    otherwise silently compare unrelated channel pairs."""
    graphs, channel_names = [], None
    for d in model_dirs:
        graph, names = load_fold(d)
        if channel_names is None:
            channel_names = names
        elif names != channel_names:
            raise ValueError(
                f'{d}\'s channel_names differs from {model_dirs[0]}\'s - these folds are not '
                f'comparable. This should never happen for folds of the same k-fold run.'
            )
        graphs.append(graph)
    return np.stack(graphs, axis=0), channel_names  # (K, N, N)


def top_n_outgoing_mask(graph: np.ndarray, top_n: int) -> np.ndarray:
    """Boolean (N, N) mask, True at [i, j] if cause j's edge to effect i is among cause j's top_n
    strongest outgoing edges in this graph (diagonal excluded). Dataset-agnostic - no coordinate/
    WL_-only restriction like plot_causal_map.top_n_outgoing_per_node, since this aggregator doesn't
    know about any particular dataset's channel-naming conventions."""
    edge_strength = graph.copy()
    np.fill_diagonal(edge_strength, -np.inf)
    n = graph.shape[0]
    mask = np.zeros_like(edge_strength, dtype=bool)
    for j in range(n):  # j = cause/source column
        keep = min(top_n, n - 1)
        if keep <= 0:
            continue
        top_rows = np.argpartition(edge_strength[:, j], -keep)[-keep:]
        mask[top_rows, j] = True
    return mask


def compute_recurrence(graphs: np.ndarray, mode: str, top_n: int = 3, top_k: int = 15) -> np.ndarray:
    """graphs: (K, N, N) stacked per-fold graphs (row=effect, col=cause). Returns an (N, N) int array
    counting, per edge, in how many of the K folds that edge qualifies as "strong" by mode:
      - 'top-n-outgoing': edge [i, j] counts for fold k if it's among cause j's top_n strongest
        outgoing edges in fold k's graph.
      - 'global-top-k': edge [i, j] counts for fold k if it's among the top_k strongest edges
        anywhere in fold k's graph (diagonal excluded).
    """
    n = graphs.shape[1]
    recurrence = np.zeros((n, n), dtype=int)
    for fold_graph in graphs:
        if mode == 'top-n-outgoing':
            mask = top_n_outgoing_mask(fold_graph, top_n)
        elif mode == 'global-top-k':
            edge_strength = fold_graph.copy()
            np.fill_diagonal(edge_strength, -np.inf)
            flat_idx = np.argsort(edge_strength.ravel())[::-1][:top_k]
            mask = np.zeros_like(edge_strength, dtype=bool)
            mask.ravel()[flat_idx] = True
        else:
            raise ValueError(f'Unknown recurrence mode: {mode!r}')
        recurrence += mask.astype(int)
    return recurrence


def summarize(mean_graph, std_graph, recurrence, channel_names, k, top_k, mode):
    """Prints the top top_k edges by cross-fold mean strength (diagonal excluded), each annotated
    with its std and recurrence count out of k."""
    edge_strength = mean_graph.copy()
    np.fill_diagonal(edge_strength, -np.inf)
    flat_idx = np.argsort(edge_strength.ravel())[::-1][:top_k]
    rows, cols = np.unravel_index(flat_idx, edge_strength.shape)
    print('=' * 78)
    print(f'Top {top_k} discovered causal edges by cross-fold mean strength '
          f'(K={k} folds, recurrence mode={mode}):')
    print(f'{"effect <- cause":45s} {"mean":>8s} {"std":>8s} {"recurrence":>12s}')
    for i, j in zip(rows, cols):
        edge = f'{channel_names[i]} <- {channel_names[j]}'
        print(f'{edge:45s} {mean_graph[i, j]:8.4f} {std_graph[i, j]:8.4f} '
              f'{recurrence[i, j]:5d} / {k}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--sweep-dir', help='sweep.py sweep directory that swept `fold`')
    src.add_argument('--fold-dirs', nargs='+', help='explicit fold model directories (each '
                      'containing cuts_plus_graph.npy + channel_names.json)')
    parser.add_argument('--output', required=True, help='directory to write aggregate outputs to')
    parser.add_argument('--recurrence-mode', choices=['top-n-outgoing', 'global-top-k'],
                         default='top-n-outgoing')
    parser.add_argument('--recurrence-top-n', type=int, default=3)
    parser.add_argument('--recurrence-top-k', type=int, default=15)
    parser.add_argument('--top-k', type=int, default=20, help='edges to print in the summary')
    args = parser.parse_args()

    model_dirs = discover_fold_dirs(args.sweep_dir) if args.sweep_dir else args.fold_dirs
    print(f'Aggregating {len(model_dirs)} fold(s):')
    for d in model_dirs:
        print(f'  {d}')

    graphs, channel_names = load_all_folds(model_dirs)
    k = graphs.shape[0]
    mean_graph = graphs.mean(axis=0)
    std_graph = graphs.std(axis=0)
    recurrence = compute_recurrence(graphs, args.recurrence_mode,
                                     top_n=args.recurrence_top_n, top_k=args.recurrence_top_k)

    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, 'kfold_mean_graph.npy'), mean_graph)
    np.save(os.path.join(args.output, 'kfold_std_graph.npy'), std_graph)
    np.save(os.path.join(args.output, 'kfold_recurrence.npy'), recurrence)
    shutil.copy(os.path.join(model_dirs[0], 'channel_names.json'),
                os.path.join(args.output, 'channel_names.json'))
    print(f'\nWrote kfold_mean_graph.npy, kfold_std_graph.npy, kfold_recurrence.npy, '
          f'channel_names.json to {args.output}')

    summarize(mean_graph, std_graph, recurrence, channel_names, k, args.top_k, args.recurrence_mode)


if __name__ == '__main__':
    main()
