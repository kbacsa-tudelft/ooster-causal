"""
Per-channel, per-timestep ground-truth labels for root-cause analysis during flood events, built from
flood_events.py's station_events.csv (one row per station per event it individually crossed its own
threshold for).

Root-cause ground truth is per-station, not per-event: a merged national event (e.g. a North Sea storm
surge) can be "caused" by several coastal stations crossing their own threshold near-simultaneously, and
many other stations are merely downstream/affected without crossing anything themselves. Only the
former are labeled 1; everything else is 0, even during an active event window - that asymmetry is what
makes AC@k a meaningful ranking metric instead of a label that's uniformly 1 and trivially satisfied.

Usage (as a library - see cuts_plus_rca.py's --flood-events-csv):
    windows = load_station_windows('flood_events_output/station_events.csv', min_alert='medium')
    labels = build_channel_labels(channel_names, session_df.index, windows)
"""
import numpy as np
import pandas as pd

ALERT_RANK = {'low': 1, 'medium': 2, 'high': 3}


def _naive(ts) -> pd.Timestamp:
    """station_events.csv's start/end/peak_time are full timestamps from the tz-aware
    (Europe/Amsterdam) series flood_events.py builds internally - unlike events.csv's start/end,
    which merge_national() explicitly reduces to a plain (tz-naive) date. Strip tz here so these
    compare cleanly against the training dataset's tz-naive local-wall-clock index."""
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tz is not None else ts


def load_station_windows(station_events_csv: str, min_alert: str) -> list:
    """Returns [(station_code, start, end), ...] for station-level exceedances at or above min_alert -
    the exact [start, end] timestamps (inclusive) the station's own value was above its threshold, at
    the dataset's native resolution (not day-granularity like events.csv's merged windows).
    station_code is the raw code from the dataset (e.g. 'borgharen.beneden'), not yet prefixed with
    WL_/Q_."""
    ev = pd.read_csv(station_events_csv)
    min_rank = ALERT_RANK[min_alert]
    ev = ev[ev['alert'].map(ALERT_RANK) >= min_rank]
    return [(r.code, _naive(r.start), _naive(r.end)) for r in ev.itertuples()]


def build_channel_labels(channel_names: list, index: pd.DatetimeIndex, windows: list) -> np.ndarray:
    """(len(index), len(channel_names)) binary array: 1 where that channel's own station has an active
    exceedance window ([start, end] inclusive) at that timestep. A window whose station code doesn't
    match any channel (e.g. the station was dropped by the sparse-channel filter, or it's a rain/
    discharge channel - station_events.csv only ever contains water-level stations) is silently
    skipped."""
    col_index = {name: i for i, name in enumerate(channel_names)}
    labels = np.zeros((len(index), len(channel_names)), dtype=np.float32)
    for code, start, end in windows:
        col = col_index.get('WL_' + code)
        if col is None:
            continue
        in_window = (index >= start) & (index <= end)
        if in_window.any():
            labels[in_window, col] = 1.0
    return labels
