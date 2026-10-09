"""
Prepares a two_week_chunks-style dataset for a run on another machine: cleans up duplicate rows,
regularizes every session onto one common frequency (auto-detected as the coarsest native cadence
found in the data, so genuinely higher-resolution sessions are correctly averaged down rather than
misread as mostly missing), harmonizes the column schema across sessions, and writes a clean staging
directory + normalization_stats.json ready to hand to cuts_plus_rca.py's --data-dir as-is.

Real-world quirks this handles (seen across different exports of this dataset):
  - Duplicate (timestamp, value) rows papering over real gaps by repeating the previous reading
    instead of leaving them missing - dropped before anything else.
  - A genuine sampling-frequency change partway through the data's history (e.g. hourly readings
    early on, 10-minute readings later) - handled by resampling onto the coarsest frequency actually
    present, so no session is upsampled/fabricated and no session is misread as mostly-missing from a
    naive reindex onto too-fine a grid.
  - A column schema that differs across sessions (stations added/removed over time) - handled by
    reindexing every session onto the column union; a channel absent from a given era is NaN there,
    which cuts_plus_rca.py's NaN-aware masking handles correctly.
  - Water level (WL_*) is an instantaneous physical quantity - resampled by mean. Rainfall (RH_*) is
    a per-interval incremental amount - resampled by sum (mean would understate the true total by the
    ratio of frequencies whenever native cadence is finer than the target), with an all-missing bucket
    correctly left NaN rather than a spurious 0 (min_count=1).
  - Water level physically responds to *accumulated* rainfall over hours (a catchment integrates and
    retains water), not the instantaneous rate one timestep back, which is what a raw rainfall channel
    gives a single-lag-back causal model like CUTS+. `--rain-window` replaces each RH_* channel with
    its own rolling sum over that window (e.g. '6h'), computed on the raw per-interval amounts before
    any normalization - cuts_plus_rca.py normalizes whatever channel it's given at train time, so no
    separate normalization step is needed here.

What this does, per session file:
  1. Drop duplicate index rows (keep the first occurrence).
  2. Resample onto a regular `--freq` grid spanning the file's own [min, max] timestamp (mean for
     WL_*, sum for RH_*) - `--freq` defaults to 'auto', using the coarsest native cadence found across
     the whole dataset.
  3. If `--rain-window` is given, replace each RH_* channel with its rolling sum over that window.
  4. Reindex columns to the full union across all sessions.
Then writes every session to `--output-dir` and a normalization_stats.json (per-channel mean/std over
the whole prepared dataset) alongside it.

`--start` drops sessions before a given date, matched against the date range encoded in each session's
filename (e.g. to exclude an earlier, lower-quality era from the resampled output without re-downloading).

Usage:
    python3 cuts_plus_prototype/prepare_data.py --input-dir two_week_chunks --output-dir two_week_chunks_full \
        --rain-window 6h
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RANGE_RE = re.compile(r'(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})')


def scan_columns_and_freq(files: list) -> tuple:
    """First pass, schema/index-only (no data columns loaded) - cheap even for hundreds of wide
    sessions. Returns (sorted column union, coarsest native cadence, total duplicate row count).
    Coarsest native cadence = the max across sessions of each session's own modal consecutive-
    timestamp diff (so a finer-sampled session doesn't override a genuinely coarser one)."""
    all_columns = set()
    modal_deltas = []
    total_dupes = 0
    for f in files:
        names = pq.ParquetFile(f).schema_arrow.names
        all_columns.update(c for c in names if c != 'timestamp' and not c.startswith('__index_level'))

        idx = pd.read_parquet(f, columns=[]).index  # index only - the data columns aren't loaded
        deduped = idx[~idx.duplicated(keep='first')]
        total_dupes += len(idx) - len(deduped)
        if len(deduped) >= 2:
            diffs = deduped.to_series().diff().dropna()
            if len(diffs):
                modal_deltas.append(diffs.value_counts().idxmax())

    if not modal_deltas:
        raise ValueError('Could not detect a native sampling frequency from any session.')
    return sorted(all_columns), max(modal_deltas), total_dupes


def prepare(input_dir: str, output_dir: str, freq: str = 'auto', min_rows: int = 144,
            rain_window: str = None, start: str = None):
    files = sorted(glob.glob(os.path.join(input_dir, '*.parquet')))
    if not files:
        raise ValueError(f'No .parquet files found in {input_dir}')

    if start:
        cutoff = pd.Timestamp(start)
        kept = []
        for f in files:
            m = RANGE_RE.search(os.path.basename(f))
            if not m:
                raise ValueError(f'--start given but no date range found in filename: {f}')
            if pd.Timestamp(m.group(1)) >= cutoff:
                kept.append(f)
        print(f'--start {start}: keeping {len(kept)}/{len(files)} session(s)')
        files = kept
        if not files:
            raise ValueError(f'No sessions on/after --start {start}')

    print(f'Scanning {len(files)} raw session(s) (schema/index only)...')
    channel_names, coarsest_delta, total_dupes = scan_columns_and_freq(files)
    if freq == 'auto':
        freq = pd.tseries.frequencies.to_offset(coarsest_delta).freqstr
        print(f'Auto-detected target frequency: {freq} (coarsest native cadence found)')
    n_channels = len(channel_names)
    print(f'{n_channels} channels in the union schema, {total_dupes} duplicate row(s) found')

    print(f'Resampling and writing {len(files)} session(s) onto a {freq} grid...')
    if rain_window:
        print(f'Replacing each RH_* channel with its rolling {rain_window} sum...')

    # Running per-channel sum/sum-of-squares/observed-count, updated one session at a time, so
    # normalization stats never require holding every (reindexed-to-the-full-union) session in
    # memory at once - the previous approach OOM'd on a wide, many-session dataset (417 sessions x
    # ~1300 union columns) despite each session on its own being perfectly manageable.
    os.makedirs(output_dir, exist_ok=True)
    sum_ = np.zeros(n_channels)
    sumsq = np.zeros(n_channels)
    count = np.zeros(n_channels)
    n_kept = 0
    total_rows = 0
    for f in files:
        name = os.path.basename(f)
        df = pd.read_parquet(f)
        dedup = df[~df.index.duplicated(keep='first')]
        del df

        rain_cols = [c for c in dedup.columns if c.startswith('RH_')]
        other_cols = [c for c in dedup.columns if c not in rain_cols]
        parts = [dedup[other_cols].resample(freq).mean()]
        if rain_cols:
            parts.append(dedup[rain_cols].resample(freq).sum(min_count=1))
        regular = pd.concat(parts, axis=1)[dedup.columns]
        del dedup

        if rain_window and rain_cols:
            regular[rain_cols] = regular[rain_cols].rolling(rain_window, min_periods=1).sum()

        if len(regular) < min_rows:
            print(f'  skipping {name}: only {len(regular)} rows on the {freq} grid (< {min_rows})')
            continue

        regular = regular.reindex(columns=channel_names)
        regular.to_parquet(os.path.join(output_dir, name))

        values = regular.values
        finite = np.isfinite(values)
        sum_ += np.where(finite, values, 0).sum(axis=0)
        sumsq += np.where(finite, values * values, 0).sum(axis=0)
        count += finite.sum(axis=0)
        n_kept += 1
        total_rows += len(regular)
        del regular, values

    count_safe = np.where(count == 0, 1, count)
    mean = np.where(count > 0, sum_ / count_safe, np.nan)
    variance = np.clip(sumsq / count_safe - mean ** 2, 0, None)  # clip: fp cancellation can go ~0-
    std = np.where(count > 0, np.sqrt(variance), np.nan)         # negative for a near-constant channel

    stats = {'mean': dict(zip(channel_names, mean.tolist())),
             'std': dict(zip(channel_names, std.tolist()))}
    with open(os.path.join(output_dir, 'normalization_stats.json'), 'w') as fh:
        json.dump(stats, fh, indent=2)

    total_cells = total_rows * n_channels
    n_missing = total_cells - int(count.sum())
    print(f'Wrote {n_kept} sessions to {output_dir}')
    print(f'Overall missing fraction: {n_missing / total_cells:.4f} ({n_missing}/{total_cells} cells)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', default='two_week_chunks')
    parser.add_argument('--output-dir', default='two_week_chunks_full')
    parser.add_argument('--freq', default='auto',
                         help='regular grid frequency to resample sessions onto, or "auto" to use '
                              'the coarsest native cadence found in the data')
    parser.add_argument('--min-rows', type=int, default=144,
                         help='drop any session with fewer than this many rows on the regular grid')
    parser.add_argument('--rain-window', default=None,
                         help='replace each RH_* channel with its rolling sum over this window (e.g. '
                              '"6h") instead of the raw per-interval amount; omit to leave RH_* as-is')
    parser.add_argument('--start', default=None,
                         help='drop sessions starting before this date (YYYY-MM-DD), matched against '
                              'the date range in each session filename; omit to include everything')
    args = parser.parse_args()
    prepare(args.input_dir, args.output_dir, args.freq, args.min_rows, args.rain_window, args.start)


if __name__ == '__main__':
    main()
