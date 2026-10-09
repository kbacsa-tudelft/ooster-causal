"""
Combines the occluded rws water levels (rws_waterinfo_adapted/, WL_* columns, 5- or 10-minute) with the
KNMI hourly rainfall (knmi_rain/sessions/, RH_* columns) and, optionally, the occluded rws discharge
(rws_discharge_adapted/, Q_* columns, same cadence as water level) into one set of two-week sessions.

Rainfall is hourly, so each hour's total is spread evenly over the six 10-minute steps that make it up
(value / 6 at each step). This keeps every hourly total exact, but it assumes the rain fell evenly within
the hour; no sub-hourly rain pattern is invented. Hours missing in the KNMI data stay NaN. Discharge needs
no such spreading: it's downloaded the same way as water level (download_rws_waterlevel.py), so it already
shares water level's native cadence and session grid - it's just concatenated in like another column group.

Sessions are paired by their date range, so every input must use the same session boundaries (they do
for rws_waterinfo, rws_discharge and knmi_rain, all starting on the same 14-day grid). A session
missing from one source keeps the others, with NaN for the missing one.

Usage:
    python3 cuts_plus_prototype/combine_rws_knmi.py --wl-dir rws_waterinfo_adapted --rain-dir knmi_rain/sessions \
        --rain-locations knmi_rain/locations.csv --wl-locations rws_waterinfo_adapted/locations.csv \
        --discharge-dir rws_discharge_adapted --discharge-locations rws_discharge_adapted/locations.csv \
        --output combined_adapted
"""
import argparse
import glob
import os
import re
import shutil

import pandas as pd

RANGE_RE = re.compile(r'(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})')


def session_key(path: str):
    m = RANGE_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f'no date range in {path}')
    return m.group(1), m.group(2)


def hourly_to_10min(hourly: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Spreads each hour's total over its six 10-minute steps (value / 6), on the grid [start, end)."""
    grid = pd.date_range(start, end - pd.Timedelta(minutes=10), freq='10min')
    values = hourly.reindex(grid.floor('h')).to_numpy(dtype=float) / 6.0
    return pd.DataFrame(values, index=grid, columns=hourly.columns)


def combine(wl_dir: str, rain_dir: str, output_dir: str, wl_locations: str, rain_locations: str,
           discharge_dir: str = None, discharge_locations: str = None):
    os.makedirs(output_dir, exist_ok=True)
    rain_by_range = {session_key(f): f for f in glob.glob(os.path.join(rain_dir, '*.parquet'))}
    discharge_by_range = ({session_key(f): f for f in glob.glob(os.path.join(discharge_dir, '*.parquet'))}
                          if discharge_dir else {})
    wl_files = sorted(glob.glob(os.path.join(wl_dir, '*.parquet')))
    if not wl_files:
        raise ValueError(f'No water-level sessions in {wl_dir}')

    if rain_by_range and not any(session_key(f) in rain_by_range for f in wl_files):
        raise ValueError('no rainfall session has the same date range as a water-level session; '
                         'both datasets must be cut on the same 14-day grid (same --start)')
    if discharge_by_range and not any(session_key(f) in discharge_by_range for f in wl_files):
        raise ValueError('no discharge session has the same date range as a water-level session; '
                         'both datasets must be cut on the same 14-day grid (same --start)')

    n_with_rain = 0
    n_with_discharge = 0
    for f in wl_files:
        key = session_key(f)
        start = pd.Timestamp(key[0])
        end = pd.Timestamp(key[1])  # sessions are contiguous: this is also the next session's start
        parts = [pd.read_parquet(f)]

        if key in rain_by_range:
            hourly = pd.read_parquet(rain_by_range[key])
            parts.append(hourly_to_10min(hourly, start, end))
            n_with_rain += 1
        if key in discharge_by_range:
            parts.append(pd.read_parquet(discharge_by_range[key]))  # already on the same grid as wl
            n_with_discharge += 1

        combined = pd.concat(parts, axis=1).sort_index() if len(parts) > 1 else parts[0].sort_index()
        combined.index.name = 'timestamp'
        combined.to_parquet(os.path.join(output_dir, os.path.basename(f)))

    loc_parts = [pd.read_csv(wl_locations), pd.read_csv(rain_locations)]
    if discharge_locations:
        loc_parts.append(pd.read_csv(discharge_locations))
    loc = pd.concat(loc_parts, ignore_index=True)
    loc.drop_duplicates('name').to_csv(os.path.join(output_dir, 'locations.csv'), index=False)
    print(f'{len(wl_files)} session(s) written; {n_with_rain} with rainfall, '
          f'{n_with_discharge} with discharge')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--wl-dir', default='rws_waterinfo_adapted')
    p.add_argument('--rain-dir', default='knmi_rain/sessions')
    p.add_argument('--wl-locations', default='rws_waterinfo_adapted/locations.csv')
    p.add_argument('--rain-locations', default='knmi_rain/locations.csv')
    p.add_argument('--discharge-dir', default=None, help='optional: occluded discharge, e.g. rws_discharge_adapted')
    p.add_argument('--discharge-locations', default=None)
    p.add_argument('--output', default='combined_adapted')
    args = p.parse_args()
    combine(args.wl_dir, args.rain_dir, args.output, args.wl_locations, args.rain_locations,
            args.discharge_dir, args.discharge_locations)


if __name__ == '__main__':
    main()
