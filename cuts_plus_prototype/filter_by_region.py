"""
Filters a prepared CUTS+ dataset (parquet sessions + normalization_stats.json) down to the channels
whose station falls inside a geographic bounding box, using a locations.csv for coordinates. Built to
carve out faster-to-train regional subsets (e.g. a river-rich Limburg/Maas corridor, or the mostly-
estuary Zeeland coast) from the full rws_waterinfo_prepared_1974 dataset - see sweep_fill_policy.sh,
which sweeps --fill-policy across one or more such subsets.

A channel column (e.g. "WL_borgharen.beneden") is matched to locations.csv's `name` column by
stripping its WL_/Q_ prefix. RH_* (rainfall) channels are matched by their full name instead - KNMI
locations.csv stores rain stations under their full prefixed channel name (e.g. "RH_VALKENBURG_210"),
not the bare station code (see download_knmi_rain.py / plot_causal_map.py's load_locations, which rely
on the same convention). A channel whose station isn't in locations.csv is dropped with a warning
(shouldn't happen for this pipeline's own output, but checked defensively rather than assumed).

Usage:
    python3 cuts_plus_prototype/filter_by_region.py --input-dir rws_waterinfo_prepared_1974 \
        --locations-csv rws_waterinfo_adapted_full/locations.csv --output-dir rws_waterinfo_limburg \
        --lat-min 50.75 --lat-max 51.80 --lon-min 5.60 --lon-max 6.20
"""
import argparse
import glob
import json
import os
import re

import pandas as pd

CHANNEL_PREFIX_RE = re.compile(r'^(WL|Q)_')


def station_of(channel: str) -> str:
    if channel.startswith('RH_'):
        return channel
    return CHANNEL_PREFIX_RE.sub('', channel)


def select_channels(columns: list, locations: pd.DataFrame,
                     lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> list:
    loc_by_name = locations.set_index('name')
    keep = []
    unmatched = []
    for c in columns:
        station = station_of(c)
        if station not in loc_by_name.index:
            unmatched.append(c)
            continue
        lat, lon = loc_by_name.loc[station, 'lat'], loc_by_name.loc[station, 'lon']
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            keep.append(c)
    if unmatched:
        shown = unmatched[:10]
        print(f'{len(unmatched)} channel(s) have no matching station in locations.csv, skipped: '
              f'{shown}{"..." if len(unmatched) > len(shown) else ""}')
    return keep


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-dir', required=True)
    p.add_argument('--locations-csv', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--lat-min', type=float, required=True)
    p.add_argument('--lat-max', type=float, required=True)
    p.add_argument('--lon-min', type=float, required=True)
    p.add_argument('--lon-max', type=float, required=True)
    args = p.parse_args()

    locations = pd.read_csv(args.locations_csv)
    files = sorted(glob.glob(os.path.join(args.input_dir, '*.parquet')))
    if not files:
        raise ValueError(f'No .parquet files found in {args.input_dir}')

    columns = list(pd.read_parquet(files[0]).columns)
    keep = select_channels(columns, locations, args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    if not keep:
        raise ValueError('No channels fall inside the given bounding box - check --lat-min/--lat-max/'
                          '--lon-min/--lon-max and --locations-csv.')
    print(f'{len(keep)} of {len(columns)} channels fall inside the bounding box:')
    for c in keep:
        print(f'  {c}')

    os.makedirs(args.output_dir, exist_ok=True)
    for f in files:
        df = pd.read_parquet(f)
        df[keep].to_parquet(os.path.join(args.output_dir, os.path.basename(f)))

    stats_src = os.path.join(args.input_dir, 'normalization_stats.json')
    if os.path.exists(stats_src):
        with open(stats_src) as fh:
            stats = json.load(fh)
        filtered_stats = {stat_name: {c: v[c] for c in keep if c in v} for stat_name, v in stats.items()}
        with open(os.path.join(args.output_dir, 'normalization_stats.json'), 'w') as fh:
            json.dump(filtered_stats, fh, indent=2)

    print(f'Wrote {len(files)} session(s), {len(keep)} channel(s), to {args.output_dir}')


if __name__ == '__main__':
    main()
