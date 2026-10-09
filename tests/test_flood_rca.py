import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'cuts_plus_prototype'))

import flood_labels  # noqa: E402
import mask_flood_events  # noqa: E402
from cuts_plus_rca import CUTSPlusRCAConfig, run_real_data_pipeline  # noqa: E402


def test_load_station_windows_filters_by_alert_and_strips_tz(tmp_path):
    csv = tmp_path / 'station_events.csv'
    pd.DataFrame([
        # full timestamps, tz-aware - matches real flood_events.py station_events.csv output
        {'station': 'A', 'code': 'stationa', 'kind': 'river',
         'start': '2021-07-14 08:00:00+02:00', 'end': '2021-07-14 20:00:00+02:00',
         'peak_time': '2021-07-14 14:00:00+02:00', 'peak_level': 500, 'alert': 'medium', 'detail': ''},
        {'station': 'B', 'code': 'stationb', 'kind': 'coast',
         'start': '2013-12-05 06:00:00+01:00', 'end': '2013-12-05 10:00:00+01:00',
         'peak_time': '2013-12-05 08:00:00+01:00', 'peak_level': 400, 'alert': 'low', 'detail': ''},
    ]).to_csv(csv, index=False)

    windows = flood_labels.load_station_windows(str(csv), min_alert='medium')
    assert len(windows) == 1
    code, start, end = windows[0]
    assert code == 'stationa'
    # tz stripped - compares cleanly against a tz-naive training index
    assert start.tzinfo is None and end.tzinfo is None
    assert start == pd.Timestamp('2021-07-14 08:00:00')
    assert end == pd.Timestamp('2021-07-14 20:00:00')

    # low threshold includes both
    assert len(flood_labels.load_station_windows(str(csv), min_alert='low')) == 2


def test_build_channel_labels_marks_only_the_triggering_station():
    index = pd.date_range('2021-07-13', periods=5, freq='1D')
    channel_names = ['WL_stationa', 'WL_stationb', 'RH_rain1']
    windows = [('stationa', pd.Timestamp('2021-07-14'), pd.Timestamp('2021-07-15'))]

    labels = flood_labels.build_channel_labels(channel_names, index, windows)

    assert labels.shape == (5, 3)
    # only WL_stationa (column 0) is ever labeled, never stationb or rain
    assert labels[:, 1].sum() == 0
    assert labels[:, 2].sum() == 0
    # labeled exactly on 2021-07-14 and 2021-07-15 (index positions 1, 2) - end is inclusive
    assert list(labels[:, 0]) == [0, 1, 1, 0, 0]


def test_build_channel_labels_skips_unmatched_station_code():
    index = pd.date_range('2021-07-13', periods=3, freq='1D')
    channel_names = ['WL_stationa']
    # station 'dropped' never matches any channel - should not raise, just contribute nothing
    windows = [('dropped', pd.Timestamp('2021-07-13'), pd.Timestamp('2021-07-14'))]
    labels = flood_labels.build_channel_labels(channel_names, index, windows)
    assert labels.sum() == 0


def test_mask_dataframe_only_touches_overlapping_rows():
    index = pd.date_range('2021-01-01', periods=6, freq='1D')
    df = pd.DataFrame({'WL_a': np.arange(6.0), 'WL_b': np.arange(6.0) * 10}, index=index)
    windows = [(pd.Timestamp('2021-01-03'), pd.Timestamp('2021-01-05'))]

    masked, hit = mask_flood_events.mask_dataframe(df, windows)

    assert list(hit) == [False, False, True, True, False, False]
    assert masked.loc[index[2]:index[3]].isna().all().all()
    assert not masked.loc[index[0]:index[1]].isna().any().any()
    assert not masked.loc[index[4]:index[5]].isna().any().any()


def test_mask_dataframe_no_overlap_is_a_noop():
    index = pd.date_range('2021-01-01', periods=4, freq='1D')
    df = pd.DataFrame({'WL_a': [1.0, 2.0, 3.0, 4.0]}, index=index)
    masked, hit = mask_flood_events.mask_dataframe(df, [(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-01-02'))])
    assert not hit.any()
    pd.testing.assert_frame_equal(masked, df)


def _write_session(path, start, periods, freq, channels, rng, flood_cols=None, flood_slice=None):
    index = pd.date_range(start, periods=periods, freq=freq)
    data = {c: rng.normal(size=periods).astype(np.float32) for c in channels}
    df = pd.DataFrame(data, index=index)
    df.index.name = 'timestamp'
    if flood_cols and flood_slice:
        for c in flood_cols:
            df.loc[df.index[flood_slice], c] += 8.0  # a clear spike, well outside normal residual range
    df.to_parquet(path)
    return index


def test_run_real_data_pipeline_labels_flood_in_score_session_only(tmp_path):
    rng = np.random.default_rng(0)
    channels = ['WL_stationa', 'WL_stationb', 'RH_rain1']
    data_dir = tmp_path / 'data'
    data_dir.mkdir()

    # 4 train-eligible + 1 val + 1 score session (sorted filenames -> chronological order).
    starts = pd.date_range('2020-01-01', periods=6, freq='10D')
    for i, start in enumerate(starts[:-1]):
        _write_session(data_dir / f'session_{i}.parquet', start, periods=80, freq='10min',
                        channels=channels, rng=rng)
    score_index = _write_session(data_dir / 'session_5.parquet', starts[-1], periods=300, freq='10min',
                                  channels=channels, rng=rng, flood_cols=['WL_stationa'],
                                  flood_slice=slice(100, 110))

    flood_dir = tmp_path / 'flood_events_output'
    flood_dir.mkdir()
    # events.csv (merged/national table) is day-granularity by construction (merge_national() calls
    # .date()); station_events.csv keeps full timestamps (and, in production, a tz-aware one) - see
    # flood_labels._naive. Mirror both conventions here rather than truncating everything to a date.
    pd.DataFrame([{
        'start': score_index[100].date().isoformat(), 'end': score_index[109].date().isoformat(),
        'alert': 'medium',
    }]).to_csv(flood_dir / 'events.csv', index=False)
    pd.DataFrame([{
        'station': 'Station A', 'code': 'stationa', 'kind': 'river',
        'start': score_index[100].isoformat(), 'end': score_index[109].isoformat(),
        'peak_time': score_index[105].isoformat(), 'peak_level': 999, 'alert': 'medium', 'detail': '',
    }]).to_csv(flood_dir / 'station_events.csv', index=False)

    config = CUTSPlusRCAConfig(
        data_dir=str(data_dir), save_dir=str(tmp_path / 'models'), log_dir=str(tmp_path / 'runs'),
        total_epoch=2, n_groups=3, group_policy='multiply_2_every_1', seed=0,
        flood_events_csv=str(flood_dir / 'events.csv'), flood_min_alert='medium',
    )

    run_real_data_pipeline(config, log_dir_name='test_flood_rca')

    labels_path = os.path.join(str(tmp_path / 'models'), 'cuts_plus_rca_session_5_labels.npy')
    assert os.path.exists(labels_path)
    labels = np.load(labels_path)
    assert labels.shape == (300, len(channels))
    assert labels[:, 0].sum() == 10  # only WL_stationa, only the 10 flagged timesteps
    assert labels[:, 1].sum() == 0
    assert labels[:, 2].sum() == 0
