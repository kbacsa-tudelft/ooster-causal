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

Usage:
    python3 cuts_plus_prototype/prepare_data.py --input-dir two_week_chunks --output-dir two_week_chunks_full \
        --rain-window 6h
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def detect_coarsest_freq(dedup_frames: list) -> pd.Timedelta:
    """Each frame's native cadence = the mode of its own consecutive timestamp diffs. The dataset's
    target frequency is the coarsest (largest) of those across all sessions."""
    modal_deltas = []
    for df in dedup_frames:
        if len(df.index) < 2:
            continue
        diffs = df.index.to_series().diff().dropna()
        if len(diffs):
            modal_deltas.append(diffs.value_counts().idxmax())
    if not modal_deltas:
        raise ValueError('Could not detect a native sampling frequency from any session.')
    return max(modal_deltas)


def prepare(input_dir: str, output_dir: str, freq: str = 'auto', min_rows: int = 144,
            rain_window: str = None):
    files = sorted(glob.glob(os.path.join(input_dir, '*.parquet')))
    if not files:
        raise ValueError(f'No .parquet files found in {input_dir}')

    print(f'Loading {len(files)} raw session(s)...')
    dedup_frames = {}
    total_dupes = 0
    for f in files:
        df = pd.read_parquet(f)
        dedup = df[~df.index.duplicated(keep='first')]
        total_dupes += len(df) - len(dedup)
        dedup_frames[os.path.basename(f)] = dedup

    if freq == 'auto':
        target = detect_coarsest_freq(list(dedup_frames.values()))
        freq = pd.tseries.frequencies.to_offset(target).freqstr
        print(f'Auto-detected target frequency: {freq} (coarsest native cadence found)')

    print(f'Resampling {len(dedup_frames)} session(s) onto a {freq} grid...')
    if rain_window:
        print(f'Replacing each RH_* channel with its rolling {rain_window} sum...')
    prepared = {}
    all_columns = set()
    for name, dedup in dedup_frames.items():
        rain_cols = [c for c in dedup.columns if c.startswith('RH_')]
        other_cols = [c for c in dedup.columns if c not in rain_cols]

        parts = [dedup[other_cols].resample(freq).mean()]
        if rain_cols:
            parts.append(dedup[rain_cols].resample(freq).sum(min_count=1))
        regular = pd.concat(parts, axis=1)[dedup.columns]

        if rain_window and rain_cols:
            regular[rain_cols] = regular[rain_cols].rolling(rain_window, min_periods=1).sum()

        if len(regular) < min_rows:
            print(f'  skipping {name}: only {len(regular)} rows on the {freq} grid (< {min_rows})')
            continue
        prepared[name] = regular
        all_columns.update(regular.columns)

    channel_names = sorted(all_columns)
    print(f'{len(prepared)} session(s) kept, {len(channel_names)} channels in the union schema, '
          f'{total_dupes} duplicate row(s) dropped')

    os.makedirs(output_dir, exist_ok=True)
    for name, df in prepared.items():
        df.reindex(columns=channel_names).to_parquet(os.path.join(output_dir, name))

    concatenated = np.concatenate(
        [df.reindex(columns=channel_names).values for df in prepared.values()], axis=0)
    mean = np.nanmean(concatenated, axis=0)
    std = np.nanstd(concatenated, axis=0)
    stats = {'mean': dict(zip(channel_names, mean.tolist())),
             'std': dict(zip(channel_names, std.tolist()))}
    with open(os.path.join(output_dir, 'normalization_stats.json'), 'w') as fh:
        json.dump(stats, fh, indent=2)

    total_cells = concatenated.size
    n_missing = int(np.isnan(concatenated).sum())
    print(f'Wrote {len(prepared)} sessions to {output_dir}')
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
    args = parser.parse_args()
    prepare(args.input_dir, args.output_dir, args.freq, args.min_rows, args.rain_window)


if __name__ == '__main__':
    main()
