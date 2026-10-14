import glob
import os
import sys

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'cuts_plus_prototype'))

from cuts_plus_rca import CUTSPlusRCAConfig, run_real_data_pipeline  # noqa: E402


def _write_session(path, start, periods, freq, channels, rng):
    index = pd.date_range(start, periods=periods, freq=freq)
    data = {c: rng.normal(size=periods).astype(np.float32) for c in channels}
    df = pd.DataFrame(data, index=index)
    df.index.name = 'timestamp'
    df.to_parquet(path)


def _load_accumulator(log_dir, log_dir_name):
    event_files = glob.glob(os.path.join(log_dir, log_dir_name, '**', 'events.out.tfevents.*'),
                             recursive=True)
    ea = EventAccumulator(os.path.dirname(event_files[0]))
    ea.Reload()
    return ea


def test_held_out_eval_logs_val_mse_only_every_held_out_eval_every_epochs(tmp_path):
    rng = np.random.default_rng(0)
    channels = ['WL_stationa', 'WL_stationb', 'RH_rain1']
    data_dir = tmp_path / 'data'
    data_dir.mkdir()

    # 10 sessions -> with default val_ratio/test_ratio=0.15, n_val=n_score=2, 6 train.
    starts = pd.date_range('2020-01-01', periods=10, freq='10D')
    for i, start in enumerate(starts):
        _write_session(data_dir / f'session_{i}.parquet', start, periods=80, freq='10min',
                        channels=channels, rng=rng)

    log_dir = str(tmp_path / 'runs')
    config = CUTSPlusRCAConfig(
        data_dir=str(data_dir), save_dir=str(tmp_path / 'models'), log_dir=log_dir,
        total_epoch=4, n_groups=3, group_policy='multiply_2_every_1', seed=0,
        held_out_eval_every=2,
    )

    run_real_data_pipeline(config, log_dir_name='test_held_out_eval')

    ea = _load_accumulator(log_dir, 'test_held_out_eval')
    scalars = ea.Scalars('held_out/val_mse')
    steps = sorted(s.step for s in scalars)
    assert steps == [2, 4]  # only every held_out_eval_every=2 epochs, not every epoch
    assert all(np.isfinite(s.value) for s in scalars)


def test_held_out_eval_disabled_by_default(tmp_path):
    rng = np.random.default_rng(0)
    channels = ['WL_stationa', 'WL_stationb', 'RH_rain1']
    data_dir = tmp_path / 'data'
    data_dir.mkdir()

    starts = pd.date_range('2020-01-01', periods=10, freq='10D')
    for i, start in enumerate(starts):
        _write_session(data_dir / f'session_{i}.parquet', start, periods=80, freq='10min',
                        channels=channels, rng=rng)

    log_dir = str(tmp_path / 'runs')
    config = CUTSPlusRCAConfig(
        data_dir=str(data_dir), save_dir=str(tmp_path / 'models'), log_dir=log_dir,
        total_epoch=3, n_groups=3, group_policy='multiply_2_every_1', seed=0,
    )

    run_real_data_pipeline(config, log_dir_name='test_held_out_eval_disabled')

    ea = _load_accumulator(log_dir, 'test_held_out_eval_disabled')
    assert 'held_out/val_mse' not in ea.Tags()['scalars']
