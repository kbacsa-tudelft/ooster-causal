"""
Removes channels observed in less than --min-availability of all rows from a prepared dataset, in place:
drops the column from every session parquet and from normalization_stats.json. Channels with no
observations at all are below any positive threshold, so they go too.

Applied after prepare_data.py (so the availability is measured on the regular grid the model sees).
Re-running is harmless: channels already removed are simply not there any more.

Usage:
    python3 cuts_plus_prototype/drop_sparse_channels.py --data-dir rws_waterinfo_prepared --min-availability 0.10
"""
import argparse
import glob
import json

import pandas as pd


def drop_sparse(data_dir: str, min_availability: float):
    files = sorted(glob.glob(f'{data_dir}/*.parquet'))
    if not files:
        raise ValueError(f'No .parquet files found in {data_dir}')

    observed = None
    rows = 0
    cols = None
    for f in files:
        d = pd.read_parquet(f)
        n = d.notna().sum().to_numpy()
        observed = n if observed is None else observed + n
        rows += len(d)
        cols = list(d.columns)

    share = observed / rows
    drop = [c for c, s in zip(cols, share) if s < min_availability]
    keep = [c for c in cols if c not in drop]
    print(f'{len(cols)} channels, {len(drop)} below {min_availability:.0%} availability, {len(keep)} kept')
    if not drop:
        return

    for f in files:
        pd.read_parquet(f)[keep].to_parquet(f)

    stats_path = f'{data_dir}/normalization_stats.json'
    with open(stats_path) as fh:
        stats = json.load(fh)
    stats = {k: {c: v for c, v in stats[k].items() if c in keep} for k in stats}
    with open(stats_path, 'w') as fh:
        json.dump(stats, fh, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--min-availability', type=float, default=0.10)
    args = p.parse_args()
    drop_sparse(args.data_dir, args.min_availability)


if __name__ == '__main__':
    main()
