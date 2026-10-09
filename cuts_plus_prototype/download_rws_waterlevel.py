"""
Downloads a single Rijkswaterstaat quantity (grootheid) with the rws-waterinfo package and rebuilds it
into two-week session files in the same layout as rws_data (one parquet per session, timestamp index,
one column per location code), so the existing adapt/prepare/train pipeline can read it. Water level
(WATHTE, cm) is the default; discharge (Q, m3/s) uses the same script with --grootheid/--eenheid.

Why one quantity at a time: the earlier rws_data export mixed many quantities (concentrations, counts,
temperature, wind, discharge) under the same location names with no way to tell them apart. Selecting
the quantity explicitly avoids that - run this script once per quantity, into separate --out directories.

Two passes, each resumable:
  1. download: one wide yearly table per year (raw/YYYY.parquet), requests batched through get_data().
  2. sessions: two-week parquet files (sessions/rws_YYYY-MM-DD_YYYY-MM-DD.parquet), cut from the yearly tables.
A channels.csv (provenance: code, name, quantity, unit, compartment, measuring device) and a locations.csv
(name, lat, lon; ETRS89, which is WGS84 for this purpose) are written next to them.

Install first:  pip install rws-waterinfo

Historical coverage varies by station and is picked up automatically: a location's catalog entry
often lists several measuring-device codes from different eras (e.g. Vlissingen has four, the oldest
giving 3-hourly readings back to at least 1950, the newest giving fine-grained readings from the
1990s on), and the per-year download already tries every device for a code until one has data, with
no extra configuration needed. Coverage and cadence both improve in later decades as more automatic
stations came online - don't expect the 1950s-60s to look like 2020 in station count or resolution.

Known limitation: if a station's measuring device changes mid-year, only the first device (by catalog
order) that returns any data for that year is used for the whole year - a real mid-year splice between
two devices isn't reconstructed. This predates this rewrite and wasn't in scope to fix here.

Usage:
    python3 cuts_plus_prototype/download_rws_waterlevel.py --out rws_waterinfo \
        --start 1950-01-01 --end 2025-01-01 --workers 10
    # discharge, same date range:
    python3 cuts_plus_prototype/download_rws_waterlevel.py --out rws_discharge \
        --grootheid Q --eenheid m3/s --start 1950-01-01 --end 2025-01-01 --workers 10
    # quick test on a few stations and one month:
    python3 cuts_plus_prototype/download_rws_waterlevel.py --out rws_test \
        --start 2024-01-01 --end 2024-02-01 --limit-codes 3
"""
import argparse
import functools
import glob
import os

import pandas as pd
import rws_waterinfo as rw

SESSION_DAYS = 14


def select_locations(catalog: pd.DataFrame, grootheid: str, compartiment: str, eenheid: str, proces: str,
                     limit: int | None) -> pd.DataFrame:
    """One row per (location, measuring device) that measures the given quantity in the requested unit."""
    sel = catalog[(catalog['Grootheid.Code'] == grootheid) & (catalog['ProcesType'] == proces)
                  & (catalog['Eenheid.Code'] == eenheid) & (catalog['Compartiment.Code'] == compartiment)]
    sel = sel.drop_duplicates(['Code', 'MeetApparaat.Code']).sort_values(['Code', 'MeetApparaat.Code'])
    if limit:
        sel = sel[sel['Code'].isin(sel['Code'].drop_duplicates().head(limit))]
    return sel.reset_index(drop=True)


def request_params(sel: pd.DataFrame, start: str, end: str, proces: str, grootheid: str) -> list:
    return [{
        'locatie_code': r['Code'],
        'compartiment_code': r['Compartiment.Code'],
        'eenheid_code': r['Eenheid.Code'],
        'meetapparaat_code': int(r['MeetApparaat.Code']),  # numpy int64 is not JSON-serializable
        'grootheid_code': grootheid,
        'start_date': start,
        'end_date': end,
        'proces_type': proces,
    } for _, r in sel.iterrows()]


def download_year(sel: pd.DataFrame, year: int, proces: str, workers: int, batch: int, grootheid: str) -> pd.DataFrame:
    """Wide table for one year: index = local naive timestamp, one column per location code."""
    start, end = f'{year}-01-01', f'{year + 1}-01-01'
    params = request_params(sel, start, end, proces, grootheid)
    pieces, seen = [], set()
    for i in range(0, len(params), batch):
        chunk = params[i:i + batch]
        df = rw.get_data(chunk, return_df=True, parallel=True, max_workers=workers, proces_type=proces)
        if df is None or len(df) == 0:
            continue
        df = df[['Code', 'Tijdstip', 'Meetwaarde.Waarde_Numeriek']].copy()
        df['Tijdstip'] = pd.to_datetime(df['Tijdstip'], utc=True).dt.tz_convert('Europe/Amsterdam').dt.tz_localize(None)
        df['value'] = pd.to_numeric(df['Meetwaarde.Waarde_Numeriek'], errors='coerce')
        df = df.dropna(subset=['value']).drop_duplicates(['Code', 'Tijdstip'])
        df = df[~df['Code'].isin(seen)]  # first measuring device per location wins
        if df.empty:
            continue
        seen.update(df['Code'].unique())
        wide = df.pivot(index='Tijdstip', columns='Code', values='value')
        pieces.append(wide)
        print(f'  {year}: {min(i + batch, len(params))}/{len(params)} requests, {len(seen)} locations with data')
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, axis=1).sort_index()


def build_sessions(raw_dir: str, out_dir: str, start: str, end: str):
    """Cuts the yearly tables into two-week session files, named like rws_data's (rws_start_end.parquet)."""
    os.makedirs(out_dir, exist_ok=True)

    @functools.lru_cache(maxsize=2)
    def year_table(y):
        path = os.path.join(raw_dir, f'{y}.parquet')
        return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()

    s = pd.Timestamp(start)
    stop = pd.Timestamp(end)
    written = 0
    while s < stop:
        e = s + pd.Timedelta(days=SESSION_DAYS)
        name = f'rws_{s.date()}_{e.date()}.parquet'
        path = os.path.join(out_dir, name)
        if not os.path.exists(path):
            parts = [year_table(y) for y in range(s.year, e.year + 1)]
            parts = [p for p in parts if len(p)]
            if parts:
                full = pd.concat(parts).sort_index()
                full = full[~full.index.duplicated()]
                sl = full.loc[(full.index >= s) & (full.index < e)].dropna(how='all', axis=1)
                sl = sl.dropna(how='all')
                if len(sl):
                    sl.index.name = 'timestamp'
                    sl.to_parquet(path)
                    written += 1
        s = e
    print(f'sessions written this run: {written}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='rws_waterinfo')
    p.add_argument('--start', default='1950-01-01')
    p.add_argument('--end', default='2025-01-01')
    p.add_argument('--grootheid', default='WATHTE', help="quantity code, e.g. 'WATHTE' (water level) or 'Q' (discharge)")
    p.add_argument('--compartiment', default='OW', help='surface water = OW')
    p.add_argument('--eenheid', default='cm', help="unit code matching --grootheid, e.g. 'cm' or 'm3/s'")
    p.add_argument('--proces', default='meting', help='measured values (not forecasts/astronomical)')
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--batch', type=int, default=100, help='requests per get_data call')
    p.add_argument('--limit-codes', type=int, default=None, help='only the first N locations (for testing)')
    p.add_argument('--catalog', default=None, help='CSV cache of rw.get_catalog(); fetched if omitted')
    args = p.parse_args()

    raw_dir = os.path.join(args.out, 'raw')
    sess_dir = os.path.join(args.out, 'sessions')
    os.makedirs(raw_dir, exist_ok=True)

    if args.catalog and os.path.exists(args.catalog):
        catalog = pd.read_csv(args.catalog, low_memory=False)
    else:
        catalog = rw.get_catalog()
        if args.catalog:
            catalog.to_csv(args.catalog, index=False)
    sel = select_locations(catalog, args.grootheid, args.compartiment, args.eenheid, args.proces, args.limit_codes)
    print(f'{sel["Code"].nunique()} locations, {len(sel)} location/device requests per year')

    print('=== pass 1: download by year ===')
    last_year = (pd.Timestamp(args.end) - pd.Timedelta(days=1)).year  # --end is exclusive
    for year in range(pd.Timestamp(args.start).year, last_year + 1):
        path = os.path.join(raw_dir, f'{year}.parquet')
        if os.path.exists(path):
            print(f'  {year}: already downloaded, skipping')
            continue
        wide = download_year(sel, year, args.proces, args.workers, args.batch, args.grootheid)
        if len(wide):
            wide.to_parquet(path)
            print(f'  {year}: saved {wide.shape[0]} timestamps x {wide.shape[1]} locations')
        else:
            print(f'  {year}: no data returned')

    print('=== pass 2: two-week sessions ===')
    build_sessions(raw_dir, sess_dir, args.start, args.end)

    present = set()
    for f in glob.glob(os.path.join(sess_dir, '*.parquet')):
        present.update(pd.read_parquet(f).columns)
    kept = sel[sel['Code'].isin(present)].drop_duplicates('Code')
    kept[['Code', 'Naam', 'Grootheid.Code', 'Eenheid.Code', 'Compartiment.Code', 'MeetApparaat.Code']].to_csv(
        os.path.join(args.out, 'channels.csv'), index=False)
    kept[['Code', 'Lat', 'Lon']].rename(columns={'Code': 'name', 'Lat': 'lat', 'Lon': 'lon'}).to_csv(
        os.path.join(args.out, 'locations.csv'), index=False)
    print(f'channels with data: {len(kept)} (channels.csv, locations.csv written to {args.out})')


if __name__ == '__main__':
    main()
