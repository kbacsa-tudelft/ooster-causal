import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'cuts_plus_prototype'))

from cuts_plus_rca import (  # noqa: E402
    CUTSPlusRCAConfig, build_rain_exogeneity_mask, run_real_data_pipeline,
)


def test_build_rain_exogeneity_mask_blocks_only_non_rain_to_rain_edges():
    channel_names = ['WL_a', 'Q_b', 'RH_c', 'RH_d']
    mask = build_rain_exogeneity_mask(channel_names)

    assert mask.shape == (4, 4)
    # row = cause, column = effect (MultiCAD's internal convention)
    rh_c, rh_d = 2, 3
    for cause in (0, 1):  # WL_a, Q_b
        assert mask[cause, rh_c] == 0.0
        assert mask[cause, rh_d] == 0.0
    # rain -> rain and rain -> WL/Q stay open
    assert mask[rh_c, rh_d] == 1.0
    assert mask[rh_d, 0] == 1.0
    # WL -> WL/Q stays open
    assert mask[0, 1] == 1.0


def _write_session(path, start, periods, freq, channels, rng):
    index = pd.date_range(start, periods=periods, freq=freq)
    data = {c: rng.normal(size=periods).astype(np.float32) for c in channels}
    df = pd.DataFrame(data, index=index)
    df.index.name = 'timestamp'
    df.to_parquet(path)


def test_run_real_data_pipeline_never_learns_a_non_rain_to_rain_edge(tmp_path):
    rng = np.random.default_rng(0)
    channels = ['WL_stationa', 'WL_stationb', 'RH_rain1']
    data_dir = tmp_path / 'data'
    data_dir.mkdir()

    starts = pd.date_range('2020-01-01', periods=6, freq='10D')
    for i, start in enumerate(starts):
        _write_session(data_dir / f'session_{i}.parquet', start, periods=80, freq='10min',
                        channels=channels, rng=rng)

    config = CUTSPlusRCAConfig(
        data_dir=str(data_dir), save_dir=str(tmp_path / 'models'), log_dir=str(tmp_path / 'runs'),
        total_epoch=2, n_groups=3, group_policy='multiply_2_every_1', seed=0,
    )

    run_real_data_pipeline(config, log_dir_name='test_rain_exogeneity')

    graph = np.load(os.path.join(str(tmp_path / 'models'), 'cuts_plus_graph.npy'))
    # saved graph convention: row = effect, column = cause (see run_real_data_pipeline's
    # "Strongest discovered causal edges (effect <- cause)" print loop)
    rain_effect = channels.index('RH_rain1')
    for cause_name in ('WL_stationa', 'WL_stationb'):
        cause = channels.index(cause_name)
        assert graph[rain_effect, cause] == 0.0
