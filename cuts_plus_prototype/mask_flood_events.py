"""
Excludes flood-event time windows from a prepared dataset (combined_prepared/, rws_waterinfo_prepared/,
etc.), using the table from flood_events.py. Every channel's value is set to NaN for every timestamp
inside a flagged event's [start, end] window, the same occlusion mechanism prepare_rws_waterinfo.py
already uses for implausible values: NaN -> the training's observation mask treats it as missing and
never learns from it, rather than removing the rows or the sessions outright.

This treats a flood as a different causal regime from everyday conditions, which we don't want blended
into the graph learned from normal conditions - not as a data-quality problem to patch over.

By default this masks medium and high alert events only (--min-alert), since "low" alerts are frequent,
mild exceedances (a 2-year river level, a minor storm surge) rather than the kind of event this is for.
The whole event window is masked for every channel, not just the water-level stations the event was
detected from: rain and discharge during a flood are part of the same regime shift.

Writes a new directory by default, leaving --data-dir untouched; pass --in-place to overwrite it.

This masks every session the same way, train/val/score alike - fine for excluding floods from a
dataset entirely. To instead train on flood-free data while evaluating root-cause ranking against the
real flood values in held-out sessions, don't run this script at all: pass --flood-events-csv straight
to cuts_plus_rca.py, which masks train/val only in memory (see flood_labels.py).

Usage:
    python3 flood_events.py --out flood_events_output
    python3 cuts_plus_prototype/mask_flood_events.py --data-dir combined_prepared \
        --events-csv flood_events_output/events.csv --output combined_prepared_noflood
"""
import argparse
import glob
import os
import shutil

import numpy as np
import pandas as pd

ALERT_RANK = {'low': 1, 'medium': 2, 'high': 3}


def load_windows(events_csv: str, min_alert: str) -> list:
    """Returns [(start, end_exclusive), ...] for events at or above min_alert, end_exclusive being
    the day after the event's last day (so the end date itself is fully included)."""
    ev = pd.read_csv(events_csv)
    min_rank = ALERT_RANK[min_alert]
    ev = ev[ev['alert'].map(ALERT_RANK) >= min_rank]
    return [(pd.Timestamp(r.start), pd.Timestamp(r.end) + pd.Timedelta(days=1))
            for r in ev.itertuples()]


def mask_dataframe(df: pd.DataFrame, windows: list) -> tuple:
    """Sets every column to NaN for timestamps inside any (start, end_exclusive) window. Returns
    (masked_df, hit) - hit is the boolean row mask actually applied, for reporting."""
    session_start, session_end = df.index.min(), df.index.max()
    hit = np.zeros(len(df), dtype=bool)
    for start, end in windows:
        if start >= session_end or end <= session_start:
            continue
        in_window = (df.index >= start) & (df.index < end)
        hit |= in_window
    if hit.any():
        df = df.copy()
        df.loc[hit, :] = np.nan
    return df, hit


def mask_dataset(data_dir: str, output_dir: str, windows: list, in_place: bool):
    files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
    if not files:
        raise ValueError(f'No .parquet files found in {data_dir}')
    if not in_place:
        os.makedirs(output_dir, exist_ok=True)

    total_cells = 0
    masked_cells = 0
    sessions_touched = 0
    events_applied = set()
    for f in files:
        df = pd.read_parquet(f)
        total_cells += df.size
        session_start, session_end = df.index.min(), df.index.max()
        for i, (start, end) in enumerate(windows):
            if start >= session_end or end <= session_start:
                continue
            if ((df.index >= start) & (df.index < end)).any():
                events_applied.add(i)
        df, hit = mask_dataframe(df, windows)
        if hit.any():
            masked_cells += int(hit.sum()) * df.shape[1]
            sessions_touched += 1
        out_path = f if in_place else os.path.join(output_dir, os.path.basename(f))
        df.to_parquet(out_path)

    if not in_place:
        stats_src = os.path.join(data_dir, 'normalization_stats.json')
        if os.path.exists(stats_src):
            shutil.copy(stats_src, os.path.join(output_dir, 'normalization_stats.json'))

    share = masked_cells / total_cells if total_cells else 0.0
    print(f'{len(windows)} event window(s) considered, {len(events_applied)} overlapped the dataset')
    print(f'{sessions_touched} of {len(files)} session(s) touched; '
          f'{masked_cells:,} of {total_cells:,} cells masked ({share:.3%})')
    dest = data_dir if in_place else output_dir
    print(f'Wrote masked dataset to {dest}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--events-csv', default='flood_events_output/events.csv')
    p.add_argument('--min-alert', choices=['low', 'medium', 'high'], default='medium',
                   help='mask events at or above this alert level (default medium)')
    p.add_argument('--output', default=None, help='new directory for the masked copy (required unless --in-place)')
    p.add_argument('--in-place', action='store_true', help='overwrite --data-dir instead of writing a copy')
    args = p.parse_args()
    if not args.in_place and not args.output:
        p.error('--output is required unless --in-place is given')

    windows = load_windows(args.events_csv, args.min_alert)
    mask_dataset(args.data_dir, args.output, windows, args.in_place)


if __name__ == '__main__':
    main()
