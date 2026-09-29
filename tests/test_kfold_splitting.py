import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'cuts_plus_prototype'))

from cuts_plus_rca import compute_sparse_mask, split_sessions_for_fold  # noqa: E402


def test_split_sessions_n_folds_1_is_noop():
    pool_ids = [f's{i}' for i in range(17)]  # deliberately non-round size
    assert split_sessions_for_fold(pool_ids, n_folds=1, fold=0) == pool_ids


def test_split_sessions_folds_partition_the_pool():
    pool_ids = [f's{i}' for i in range(17)]
    n_folds = 5
    held_out_per_fold = []
    for fold in range(n_folds):
        train_ids = split_sessions_for_fold(pool_ids, n_folds, fold)
        held_out = set(pool_ids) - set(train_ids)
        held_out_per_fold.append(held_out)
        # this fold's train_ids is exactly pool_ids minus its own stripe, nothing else touched
        assert train_ids == [sid for sid in pool_ids if sid not in held_out]

    # every pool id is held out in exactly one fold (a true partition of the pool by held-out sets)
    all_held_out = [sid for held in held_out_per_fold for sid in held]
    assert sorted(all_held_out) == sorted(pool_ids)
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            assert held_out_per_fold[i].isdisjoint(held_out_per_fold[j])


def test_split_sessions_striping_is_interleaved_not_contiguous():
    pool_ids = [f's{i}' for i in range(10)]
    held_out = set(pool_ids) - set(split_sessions_for_fold(pool_ids, n_folds=5, fold=0))
    # fold 0's held-out stripe is pool_ids[0::5] = s0, s5 - spread across the pool, not a contiguous
    # block like the first 2 elements (s0, s1) a naive chunk-based split would produce.
    assert held_out == {'s0', 's5'}


def test_compute_sparse_mask_matches_manual_threshold():
    mask = np.array([
        [1, 1, 0, 1],
        [1, 1, 0, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 0],
    ], dtype=np.float32)  # 4 rows; column sums: 4, 2, 0, 3

    sparse, min_count = compute_sparse_mask(mask, min_channel_availability=0.0)
    assert min_count == 1  # floored at 1 even when availability is 0
    assert list(sparse) == [False, False, True, False]  # only the all-zero column is dropped

    sparse, min_count = compute_sparse_mask(mask, min_channel_availability=0.6)
    assert min_count == round(0.6 * 4) == 2
    assert list(sparse) == [c < min_count for c in [4, 2, 0, 3]]  # only the col-sum-0 column drops
