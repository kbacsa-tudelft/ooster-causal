import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'cuts_plus_prototype'))

from aggregate_kfold_graphs import (  # noqa: E402
    compute_recurrence, load_all_folds, load_fold, top_n_outgoing_mask,
)


def _write_fold(tmp_path, name, graph, channel_names):
    d = tmp_path / name
    d.mkdir()
    np.save(d / 'cuts_plus_graph.npy', graph)
    with open(d / 'channel_names.json', 'w') as fh:
        json.dump(channel_names, fh)
    return str(d)


def test_load_all_folds_computes_mean_and_std(tmp_path):
    names = ['a', 'b', 'c']
    g0 = np.array([[0, 1.0, 0.2], [0.1, 0, 0.3], [0.4, 0.5, 0]])
    g1 = np.array([[0, 0.9, 0.0], [0.3, 0, 0.1], [0.2, 0.7, 0]])
    d0 = _write_fold(tmp_path, 'f0', g0, names)
    d1 = _write_fold(tmp_path, 'f1', g1, names)

    graphs, channel_names = load_all_folds([d0, d1])
    assert channel_names == names
    assert graphs.shape == (2, 3, 3)
    np.testing.assert_allclose(graphs.mean(axis=0), (g0 + g1) / 2)


def test_load_all_folds_rejects_mismatched_channel_names(tmp_path):
    g = np.zeros((2, 2))
    d0 = _write_fold(tmp_path, 'f0', g, ['a', 'b'])
    d1 = _write_fold(tmp_path, 'f1', g, ['a', 'c'])  # different channel set
    with pytest.raises(ValueError, match='channel_names differs'):
        load_all_folds([d0, d1])


def test_load_fold_rejects_shape_mismatch(tmp_path):
    d = tmp_path / 'f0'
    d.mkdir()
    np.save(d / 'cuts_plus_graph.npy', np.zeros((3, 3)))
    with open(d / 'channel_names.json', 'w') as fh:
        json.dump(['a', 'b'], fh)  # only 2 names for a 3x3 graph
    with pytest.raises(ValueError, match='does not match'):
        load_fold(str(d))


def test_top_n_outgoing_mask_picks_strongest_per_column():
    # column j's strongest outgoing edges are its largest entries (excluding the diagonal)
    graph = np.array([
        [0.0, 0.9, 0.1],
        [0.5, 0.0, 0.8],
        [0.2, 0.3, 0.0],
    ])
    mask = top_n_outgoing_mask(graph, top_n=1)
    # column 0: entries [0.0(diag), 0.5, 0.2] -> strongest is row 1
    # column 1: entries [0.9, 0.0(diag), 0.3] -> strongest is row 0
    # column 2: entries [0.1, 0.8, 0.0(diag)] -> strongest is row 1
    expected = np.zeros((3, 3), dtype=bool)
    expected[1, 0] = True
    expected[0, 1] = True
    expected[1, 2] = True
    np.testing.assert_array_equal(mask, expected)


def test_compute_recurrence_counts_folds_where_edge_is_strong():
    # 3 nodes so a column's "top-1 outgoing" is a real discriminating choice, not the only option.
    # fold 0: column 1's strongest outgoing is row 0 (0.9 > 0.2); column 0's strongest is row 2 (0.6 > 0.1)
    g0 = np.array([[0.0, 0.9, 0.1], [0.1, 0.0, 0.2], [0.6, 0.2, 0.0]])
    # fold 1: column 1's strongest outgoing is again row 0 (0.8 > 0.3); column 0's strongest is row 1 (0.7 > 0.4)
    g1 = np.array([[0.0, 0.8, 0.2], [0.7, 0.0, 0.1], [0.4, 0.3, 0.0]])
    graphs = np.stack([g0, g1])
    recurrence = compute_recurrence(graphs, mode='top-n-outgoing', top_n=1)
    assert recurrence[0, 1] == 2  # (effect=0 <- cause=1) recurs as column 1's top choice in both folds
    assert recurrence[2, 0] == 1  # column 0's top choice is row 2 in fold 0 only (row 1 in fold 1)
    assert recurrence[1, 0] == 1  # ... and row 1 in fold 1 only
