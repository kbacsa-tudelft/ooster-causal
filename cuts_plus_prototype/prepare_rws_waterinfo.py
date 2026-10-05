"""
Prepares rws_waterinfo/ (Rijkswaterstaat measured water levels from download_rws_waterlevel.py) for
cuts_plus_rca.py, in the same convention as two_week_chunks_* and rws_data_adapted/:
  - every column prefixed WL_ (all channels are water levels in cm, relative to NAP);
  - implausible values occluded: set to NaN, so the training's observation mask treats them as missing
    and never learns from them (NaN -> mask 0, filled with the channel mean);
  - locations.csv copied (already name/lat/lon).

Occlusion rule: a value is kept only if it lies in [--min-cm, --max-cm] and is not an error code.
The default window (-1000 to 10000 cm, i.e. -10 m to +100 m) is wide enough for every real Dutch level,
including the Limburg Maas stations that sit 30-45 m above NAP. It removes 999999999 (about 2% of all
values), 9.99999e37, -99900, and values such as 33852 cm at wilhelminakanaal.sluisiii. Two error codes
lie inside the window and are removed by exact match: 9999 (8,581 values) and 99999.
The window is an assumption, not a measured property of the data; change it with the flags if you know
a different range.

Resampling onto a regular grid is left to prepare_data.py, which runs afterwards (see the usage line).

Usage:
    python3 cuts_plus_prototype/prepare_rws_waterinfo.py --input-dir rws_waterinfo --output-dir rws_waterinfo_adapted
    python3 cuts_plus_prototype/prepare_data.py --input-dir rws_waterinfo_adapted --output-dir rws_waterinfo_prepared
"""
import argparse
import glob
import os
import shutil

import numpy as np
import pandas as pd


ERROR_CODES = (9999.0, 99999.0)  # placeholder values inside the plausible window


def occlude(df: pd.DataFrame, min_cm: float, max_cm: float):
    """Returns the occluded frame and the number of values set to NaN."""
    values = df.to_numpy(dtype=float)
    observed = np.isfinite(values)
    implausible = observed & ((values < min_cm) | (values > max_cm) | np.isin(values, ERROR_CODES))
    out = df.mask(pd.DataFrame(implausible, index=df.index, columns=df.columns))
    return out, int(implausible.sum())


def prepare(input_dir: str, output_dir: str, min_cm: float, max_cm: float):
    files = sorted(glob.glob(os.path.join(input_dir, 'sessions', '*.parquet')))
    if not files:
        raise ValueError(f'No session files found in {input_dir}/sessions')
    os.makedirs(output_dir, exist_ok=True)

    print(f'Occluding values outside [{min_cm}, {max_cm}] cm in {len(files)} session(s)...')
    total_observed = 0
    total_occluded = 0
    for f in files:
        df = pd.read_parquet(f)
        total_observed += int(df.notna().sum().sum())
        df, n_occ = occlude(df, min_cm, max_cm)
        total_occluded += n_occ
        df = df.add_prefix('WL_')
        df.to_parquet(os.path.join(output_dir, os.path.basename(f)))

    shutil.copy(os.path.join(input_dir, 'locations.csv'), os.path.join(output_dir, 'locations.csv'))
    share = total_occluded / total_observed if total_observed else 0.0
    print(f'Occluded {total_occluded:,} of {total_observed:,} observed values ({share:.2%})')
    print(f'Wrote {len(files)} session(s) + locations.csv to {output_dir}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-dir', default='rws_waterinfo')
    p.add_argument('--output-dir', default='rws_waterinfo_adapted')
    p.add_argument('--min-cm', type=float, default=-1000.0)
    p.add_argument('--max-cm', type=float, default=10000.0)
    args = p.parse_args()
    prepare(args.input_dir, args.output_dir, args.min_cm, args.max_cm)


if __name__ == '__main__':
    main()
