"""
Adapts rws_data/ (Rijkswaterstaat water-level export, 2005-2008, one file per two-week session) into
the same convention the rest of the pipeline expects (as used by two_week_chunks_*):
  - Every column prefixed WL_ - this export has no rainfall channels (all columns are water-level-type
    quantities: physically plausible negative values consistent with NAP-datum-relative levels, none
    of the near-zero rain-accumulation pattern seen in the KNMI RH_* channels elsewhere).
  - locations.csv deduplicated (the source file is ~211k rows for only ~2.5k unique station codes -
    it looks like one export run appended to it rather than overwriting) and renamed to the name/lat/
    lon schema plot_causal_map.py's load_locations() expects for a dataset that's already lat/lon
    (as opposed to two_week_chunks_*'s EPSG:28992 x/y, which needs a projection step).

Each session file already has a proper DatetimeIndex (irregular cadence - a union of independently
clocked stations, not a fixed grid), so prepare_data.py's frequency auto-detection handles it
unchanged; no time-axis work needed here.

Note: this dataset spans 2005-2008, with no temporal overlap with two_week_chunks_* (2021-2026) - it
is its own independent historical dataset, not additional channels to merge into the current one.

Usage:
    python3 cuts_plus_prototype/adapt_rws_data.py --input-dir rws_data --output-dir rws_data_adapted
"""
import argparse
import glob
import os

import pandas as pd


def adapt(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, '*.parquet')))
    if not files:
        raise ValueError(f'No .parquet files found in {input_dir}')

    print(f'Adapting {len(files)} session(s)...')
    for f in files:
        df = pd.read_parquet(f)
        df = df.add_prefix('WL_')
        df.to_parquet(os.path.join(output_dir, os.path.basename(f)))

    locations_csv = os.path.join(input_dir, 'locations.csv')
    loc = pd.read_csv(locations_csv)
    before = len(loc)
    loc = loc.drop_duplicates(subset='Code', keep='first')
    loc = loc[['Code', 'Lat', 'Lon']].rename(columns={'Code': 'name', 'Lat': 'lat', 'Lon': 'lon'})
    loc.to_csv(os.path.join(output_dir, 'locations.csv'), index=False)
    print(f'locations.csv: {before} rows -> {len(loc)} unique station(s)')
    print(f'Wrote {len(files)} session(s) + locations.csv to {output_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', default='rws_data')
    parser.add_argument('--output-dir', default='rws_data_adapted')
    args = parser.parse_args()
    adapt(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
