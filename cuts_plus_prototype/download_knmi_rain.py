"""
Downloads hourly precipitation (RH, mm) from every KNMI automatic weather station that reports it, for
the same period as rws_waterinfo/, and writes it in the same two-week session layout: one parquet per
session on the same boundaries as the rws sessions, a timestamp index in Dutch local time (naive), and one
column per station named like the existing rain channels, RH_<NAME>_<CODE> (e.g. RH_DE-BILT_260).

Data: KNMI daggegevens API, uurgegevens, stns=ALL, vars=RH (public, no key). Each value is the rainfall
over the hour that starts at the timestamp (KNMI labels hours 1-24 in UTC; hour h covers
[h-1, h) UTC). Values are tenths of a millimetre in the API and are stored in mm; missing stays NaN.

Station names and coordinates come from the metadata file bundled with hydropandas
(data/knmi_meteostation.json), so hydropandas must be installed. Only stations present in the data are
kept. Writes locations.csv (name, lat, lon) with the same column names as the rws locations file, so
plot_causal_map.py can place the rain nodes.

Usage:
    pip install hydropandas pandas pyarrow
    python3 cuts_plus_prototype/download_knmi_rain.py --out knmi_rain --start 2005-01-01 --end 2025-01-01
"""
import argparse
import json
import os
import urllib.parse
import urllib.request

import pandas as pd

API = 'https://www.daggegevens.knmi.nl/klimatologie/uurgegevens'
SESSION_DAYS = 14


def fetch_month(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Hourly RH for every station between start and end (both inclusive, whole hours), as a wide frame:
    index = UTC naive hour start, columns = KNMI station code (int), values = mm."""
    body = urllib.parse.urlencode({'stns': 'ALL', 'vars': 'RH', 'start': start.strftime('%Y%m%d%H'),
                                   'end': end.strftime('%Y%m%d%H'), 'fmt': 'json'}).encode()
    with urllib.request.urlopen(urllib.request.Request(API, data=body), timeout=900) as resp:
        rows = json.load(resp)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    df['time_utc'] = (pd.to_datetime(df['date']).dt.tz_localize(None)
                      + pd.to_timedelta(df['hour'] - 1, unit='h'))
    # KNMI codes trace rainfall (below 0.05 mm) as -1, i.e. -0.1 mm here: store it as 0 mm
    df['mm'] = (pd.to_numeric(df['RH'], errors='coerce') / 10.0).clip(lower=0)
    return df.pivot_table(index='time_utc', columns='station_code', values='mm', aggfunc='first')


def fetch_year(year: int) -> pd.DataFrame:
    """One calendar year, fetched month by month: the API rejects a full-year request."""
    months = []
    for m in range(1, 13):
        first = pd.Timestamp(year=year, month=m, day=1)
        last = first + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23)
        months.append(fetch_month(first, last))
    months = [w for w in months if not w.empty]
    if not months:
        return pd.DataFrame()
    return pd.concat(months).sort_index()


def station_table(codes) -> pd.DataFrame:
    """Name and coordinates for the station codes, from the hydropandas metadata file."""
    import hydropandas
    meta_path = os.path.join(os.path.dirname(hydropandas.__file__), 'data', 'knmi_meteostation.json')
    with open(meta_path) as fh:
        meta = json.load(fh)
    rows = []
    for code in codes:
        k = str(code)
        if k not in meta['lat']:
            raise ValueError(f'station {code} has no coordinates in the hydropandas metadata')
        rows.append({'code': int(code), 'name': meta['name'][k], 'lat': meta['lat'][k], 'lon': meta['lon'][k]})
    return pd.DataFrame(rows)


def column_name(name: str, code: int) -> str:
    return f"RH_{name.upper().replace(' ', '-')}_{code}"


def build_sessions(raw_dir: str, out_dir: str, start: str, end: str, rename: dict):
    os.makedirs(out_dir, exist_ok=True)
    years = sorted({int(f[:4]) for f in os.listdir(raw_dir) if f.endswith('.parquet')})
    if not years:
        raise ValueError(f'no yearly files in {raw_dir}')
    full = pd.concat([pd.read_parquet(os.path.join(raw_dir, f'{y}.parquet')) for y in years]).sort_index()
    full = full[~full.index.duplicated(keep='first')]

    # Dutch local time, naive, to match the rws sessions. The 02:00 hour that repeats at the autumn
    # DST change is kept once (first occurrence).
    local = full.copy()
    local.index = local.index.tz_localize('UTC').tz_convert('Europe/Amsterdam').tz_localize(None)
    local = local[~local.index.duplicated(keep='first')]
    local.index.name = 'timestamp'
    local = local.rename(columns=rename)

    s = pd.Timestamp(start)
    stop = pd.Timestamp(end)
    written = 0
    while s < stop:
        e = s + pd.Timedelta(days=SESSION_DAYS)
        path = os.path.join(out_dir, f'knmi_rh_{s.date()}_{e.date()}.parquet')
        if not os.path.exists(path):
            sl = local.loc[(local.index >= s) & (local.index < e)].dropna(how='all', axis=1).dropna(how='all')
            if len(sl):
                sl.to_parquet(path)
                written += 1
        s = e
    print(f'sessions written this run: {written}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='knmi_rain')
    p.add_argument('--start', default='2005-01-01')
    p.add_argument('--end', default='2025-01-01')
    args = p.parse_args()

    raw_dir = os.path.join(args.out, 'raw')
    sess_dir = os.path.join(args.out, 'sessions')
    os.makedirs(raw_dir, exist_ok=True)

    last_year = (pd.Timestamp(args.end) - pd.Timedelta(days=1)).year
    print('=== pass 1: download by year ===')
    for year in range(pd.Timestamp(args.start).year, last_year + 1):
        path = os.path.join(raw_dir, f'{year}.parquet')
        if os.path.exists(path):
            print(f'  {year}: already downloaded, skipping')
            continue
        wide = fetch_year(year)
        if wide.empty:
            print(f'  {year}: no data returned')
            continue
        wide.to_parquet(path)
        print(f'  {year}: saved {wide.shape[0]} hours x {wide.shape[1]} stations')

    # station metadata for every code that has data in the raw files
    codes = sorted({int(c) for f in os.listdir(raw_dir) if f.endswith('.parquet')
                    for c in pd.read_parquet(os.path.join(raw_dir, f)).columns})
    st = station_table(codes)
    st['channel'] = [column_name(n, c) for n, c in zip(st['name'], st['code'])]
    st.to_csv(os.path.join(args.out, 'stations.csv'), index=False)
    st[['channel', 'lat', 'lon']].rename(columns={'channel': 'name'}).to_csv(
        os.path.join(args.out, 'locations.csv'), index=False)

    print('=== pass 2: two-week sessions ===')
    rename = dict(zip(st['code'], st['channel']))
    build_sessions(raw_dir, sess_dir, args.start, args.end, rename)
    print(f'{len(st)} stations with coordinates (stations.csv, locations.csv written to {args.out})')


if __name__ == '__main__':
    main()
