"""
Prepares a single-quantity rws download (from download_rws_waterlevel.py) for cuts_plus_rca.py, in the
same convention as two_week_chunks_* and rws_data_adapted/:
  - every column prefixed with --prefix (default WL_ for water level; use Q_ for discharge);
  - implausible values occluded: set to NaN, so the training's observation mask treats them as missing
    and never learns from them (NaN -> mask 0, filled with the channel mean);
  - locations.csv copied (already name/lat/lon).

Occlusion rule: a value is kept only if it lies in [--min-value, --max-value] and is not an error code.
The default window (-1000 to 10000, i.e. cm: -10 m to +100 m) is for water level, wide enough for every
real Dutch level, including the Limburg Maas stations that sit 30-45 m above NAP. It removes 999999999
(about 2% of all values), 9.99999e37, -99900, and values such as 33852 cm at wilhelminakanaal.sluisiii.
For discharge (m3/s), pass a window such as --min-value -500 --max-value 20000: the Rhine's recorded
maximum at Lobith is around 12,600 m3/s, so 20000 leaves headroom without keeping the same error codes.
Two error codes lie inside both windows and are removed by exact match regardless: 9999 and 99999 - the
same placeholder values turned up in a discharge sample pulled for comparison, so this looks like a
convention used across quantities, not something specific to water level.
Every window is an assumption, not a measured property of the data; change it with the flags if you
know a different range.

Resampling onto a regular grid is left to prepare_data.py, which runs afterwards (see the usage line).

Usage:
    python3 cuts_plus_prototype/prepare_rws_waterinfo.py --input-dir rws_waterinfo --output-dir rws_waterinfo_adapted
    python3 cuts_plus_prototype/prepare_data.py --input-dir rws_waterinfo_adapted --output-dir rws_waterinfo_prepared
    # discharge:
    python3 cuts_plus_prototype/prepare_rws_waterinfo.py --input-dir rws_discharge --output-dir rws_discharge_adapted \
        --prefix Q_ --min-value -500 --max-value 20000
"""
import argparse
import glob
import os
import shutil

import numpy as np
import pandas as pd

ERROR_CODES = (9999.0, 99999.0)  # placeholder values inside the plausible window


def occlude(df: pd.DataFrame, min_value: float, max_value: float):
    """Returns the occluded frame and the number of values set to NaN."""
    values = df.to_numpy(dtype=float)
    observed = np.isfinite(values)
    implausible = observed & ((values < min_value) | (values > max_value) | np.isin(values, ERROR_CODES))
    out = df.mask(pd.DataFrame(implausible, index=df.index, columns=df.columns))
    return out, int(implausible.sum())


def prepare(input_dir: str, output_dir: str, min_value: float, max_value: float, prefix: str):
    files = sorted(glob.glob(os.path.join(input_dir, 'sessions', '*.parquet')))
    if not files:
        raise ValueError(f'No session files found in {input_dir}/sessions')
    os.makedirs(output_dir, exist_ok=True)

    print(f'Occluding values outside [{min_value}, {max_value}] in {len(files)} session(s)...')
    total_observed = 0
    total_occluded = 0
    for f in files:
        df = pd.read_parquet(f)
        total_observed += int(df.notna().sum().sum())
        df, n_occ = occlude(df, min_value, max_value)
        total_occluded += n_occ
        df = df.add_prefix(prefix)
        df.to_parquet(os.path.join(output_dir, os.path.basename(f)))

    shutil.copy(os.path.join(input_dir, 'locations.csv'), os.path.join(output_dir, 'locations.csv'))
    share = total_occluded / total_observed if total_observed else 0.0
    print(f'Occluded {total_occluded:,} of {total_observed:,} observed values ({share:.2%})')
    print(f'Wrote {len(files)} session(s) + locations.csv to {output_dir}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-dir', default='rws_waterinfo')
    p.add_argument('--output-dir', default='rws_waterinfo_adapted')
    p.add_argument('--prefix', default='WL_', help='column prefix to add; Q_ for discharge')
    p.add_argument('--min-value', type=float, default=-1000.0)
    p.add_argument('--max-value', type=float, default=10000.0)
    args = p.parse_args()
    prepare(args.input_dir, args.output_dir, args.min_value, args.max_value, args.prefix)


if __name__ == '__main__':
    main()
