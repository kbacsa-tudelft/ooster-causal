"""
EDA comparing two real datasets that should yield similar causal graphs but don't:
  - two_week_chunks_full: 2021-2024, 5 water-level + 2 rainfall channels (WL_* / RH_*)
  - rws_data: 2005-2025, ~1,500 water-level stations, no rainfall channels

Writes one self-contained HTML report (matplotlib figures embedded as base64 PNG, so it opens offline)
plus a per-channel statistics CSV to --output.

Station matching is deliberately conservative and reported as such:
  1. name normalization (strip WL_/RH_ prefix and _observed_waterlevel, lowercase, '_'/'-' -> '.')
  2. a small hand-written alias table (ALIASES) for names that differ by more than punctuation
  3. coordinate-based candidates (nearest rws station within --match-radius-m), flagged as
     *candidates only* - a nearby station is not proof of the same measurement.
Rainfall channels have no water-level counterpart in rws_data by construction; they are reported
with their nearest rws site for context only and never compared as values.

Usage:
    python3 cuts_plus_prototype/eda_compare_datasets.py \
        --full-dir two_week_chunks_full --rws-dir rws_data \
        --full-locations two_week_chunks/locations.csv \
        --rws-locations rws_data_adapted/locations.csv \
        --output cuts_plus_prototype/scratch/eda
"""
import argparse
import base64
import glob
import io
import json
import os
import re
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from pyproj import Transformer  # noqa: E402

# Full-dataset water-level name (after normalization) -> rws name(s) known to be the same station.
# Hand-checked against the rws column list; keep this table small and documented.
ALIASES = {
    'roompot.buiten': ['oosterschelde.roompotsluis.buiten'],
    'haringvliet.10': ['haringvliet.10'],
}
DISCHARGE_LIKE_MAX = 1000.0  # heuristic: values this large are unlikely to be a water level in m or cm


def normalize_station(channel: str) -> str:
    """'WL_Haringvliet_10_observed_waterlevel' -> 'haringvliet.10'. RH_ channels also lose their
    trailing rain-gauge id (e.g. '_330'). Dataset-agnostic on purpose: only string rules here."""
    name = channel
    is_rain = name.startswith('RH_')
    for prefix in ('WL_', 'RH_'):
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = re.sub(r'_observed_waterlevel$', '', name)
    if is_rain:
        name = re.sub(r'_\d+$', '', name)
    return name.lower().replace('_', '.').replace('-', '.')


def session_bounds(path: str):
    """Start/end timestamps parsed from a session filename (rws_YYYY-MM-DD_YYYY-MM-DD or
    YYYY-MM-DD_to_YYYY-MM-DD)."""
    dates = re.findall(r'\d{4}-\d{2}-\d{2}', os.path.basename(path))
    return pd.Timestamp(dates[0]), pd.Timestamp(dates[1])


def coverage(files: list, label: str) -> dict:
    """Session count, date range, modal native cadence per session, timezone, duplicates, and
    column count per session (schema evolution)."""
    cadences, tzs, dupes, col_counts, years = Counter(), set(), 0, [], Counter()
    starts = []
    for f in files:
        df = pd.read_parquet(f)
        idx = df.index
        if getattr(idx, 'tz', None) is not None:
            tzs.add(str(idx.tz))
        else:
            tzs.add('naive')
        dupes += int(idx.duplicated().sum())
        if len(idx) > 1:
            diffs = pd.Series(idx).diff().dropna()
            if len(diffs):
                cadences[str(diffs.value_counts().idxmax())] += 1
        col_counts.append(df.shape[1])
        start, _ = session_bounds(f)
        starts.append(start)
        years[start.year] += 1
    return {
        'label': label,
        'sessions': len(files),
        'first': str(min(starts).date()) if starts else '',
        'last': str(max(starts).date()) if starts else '',
        'cadence_per_session': dict(cadences),
        'timezones': sorted(tzs),
        'duplicate_timestamps': dupes,
        'columns_min': min(col_counts) if col_counts else 0,
        'columns_max': max(col_counts) if col_counts else 0,
        'columns_median': float(np.median(col_counts)) if col_counts else 0.0,
        'sessions_per_year': dict(sorted(years.items())),
        'col_counts': col_counts,
        'starts': starts,
    }


def per_channel_stats(files: list, label: str) -> pd.DataFrame:
    """Streams one session at a time (rws_data has OOM'd when loaded whole). For each channel:
    observed-value count, missing fraction, mean, std, min, max, and share of sessions in which the
    channel is present but constant."""
    acc = {}
    total_rows = 0
    for f in files:
        df = pd.read_parquet(f)
        total_rows += len(df)
        for c in df.columns:
            s = df[c]
            vals = s.dropna().to_numpy(dtype=float)
            a = acc.setdefault(c, {'n_obs': 0, 'sum': 0.0, 'sumsq': 0.0, 'min': np.inf, 'max': -np.inf,
                                   'sessions_present': 0, 'sessions_constant': 0})
            if vals.size == 0:
                continue
            a['n_obs'] += vals.size
            a['sum'] += vals.sum()
            a['sumsq'] += (vals * vals).sum()
            a['min'] = min(a['min'], vals.min())
            a['max'] = max(a['max'], vals.max())
            a['sessions_present'] += 1
            if np.unique(vals).size <= 1:
                a['sessions_constant'] += 1
    rows = []
    for c, a in acc.items():
        n = a['n_obs']
        mean = a['sum'] / n if n else np.nan
        std = np.sqrt(max(a['sumsq'] / n - mean * mean, 0.0)) if n else np.nan
        rows.append({
            'dataset': label, 'channel': c, 'normalized': normalize_station(c),
            'availability': n / total_rows if total_rows else 0.0,
            'mean': mean, 'std': std,
            'min': a['min'] if n else np.nan, 'max': a['max'] if n else np.nan,
            'sessions_present': a['sessions_present'],
            'constant_session_share': (a['sessions_constant'] / a['sessions_present'])
            if a['sessions_present'] else np.nan,
            'discharge_like': bool(n and a['max'] >= DISCHARGE_LIKE_MAX),
        })
    return pd.DataFrame(rows)


def load_locations(path: str, kind: str) -> pd.DataFrame:
    """Returns name, lat, lon for either locations file (full: EPSG:28992 x/y; rws: lat/lon)."""
    loc = pd.read_csv(path)
    if kind == 'rd':
        tr = Transformer.from_crs('EPSG:28992', 'EPSG:4326', always_xy=True)
        lon, lat = tr.transform(loc['x'].values, loc['y'].values)
        out = pd.DataFrame({'name': loc['name'].astype(str), 'lat': lat, 'lon': lon})
    else:
        out = loc[['name', 'lat', 'lon']].drop_duplicates('name').copy()
        out['name'] = out['name'].astype(str)
    return out.reset_index(drop=True)


def nearest_rws(full_loc: pd.DataFrame, rws_loc: pd.DataFrame, radius_m: float) -> pd.DataFrame:
    """For each full-dataset location, the nearest rws site and its distance, as a *candidate only*."""
    lat0 = np.deg2rad(np.mean(np.r_[full_loc['lat'], rws_loc['lat']]))
    kx, ky = 111320 * np.cos(lat0), 110574.0
    rx = rws_loc['lon'].to_numpy() * kx
    ry = rws_loc['lat'].to_numpy() * ky
    rows = []
    for _, r in full_loc.iterrows():
        d = np.hypot(rx - r['lon'] * kx, ry - r['lat'] * ky)
        j = int(np.argmin(d))
        rows.append({'full_name': r['name'], 'nearest_rws': rws_loc.iloc[j]['name'],
                     'distance_m': float(d[j]), 'within_radius': bool(d[j] <= radius_m)})
    return pd.DataFrame(rows)


def value_agreement(full_files: list, rws_files: list, full_chan: str, rws_chan: str) -> dict:
    """Aligns one full-dataset channel with one rws channel on a common 10-min grid over their
    overlapping time window, and reports correlation, RMSE, and bias of the full series minus rws."""
    full_s = []
    for f in full_files:
        if full_chan in pq.read_schema(f).names:
            full_s.append(pd.read_parquet(f, columns=[full_chan])[full_chan])
    rws_s = []
    for f in rws_files:
        if rws_chan in pq.read_schema(f).names:
            rws_s.append(pd.read_parquet(f, columns=[rws_chan])[rws_chan])
    if not full_s or not rws_s:
        return {'n': 0}
    a = pd.concat(full_s).sort_index()
    b = pd.concat(rws_s).sort_index()
    # rws_data carries a fixed UTC+01:00 offset while two_week_chunks_full is naive; compare wall-clock time
    if getattr(a.index, 'tz', None) is not None:
        a.index = a.index.tz_localize(None)
    if getattr(b.index, 'tz', None) is not None:
        b.index = b.index.tz_localize(None)
    a = a[~a.index.duplicated()].resample('10min').mean()
    b = b[~b.index.duplicated()].resample('10min').mean()
    joined = pd.concat([a.rename("full"), b.rename("rws")], axis=1, sort=True).dropna()
    if len(joined) < 2:
        return {'n': len(joined)}
    diff = joined['full'] - joined['rws']
    return {
        'n': int(len(joined)),
        'corr': float(joined['full'].corr(joined['rws'])),
        'rmse': float(np.sqrt((diff ** 2).mean())),
        'bias': float(diff.mean()),
        'joined': joined,
    }


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def figures(full_cov: dict, rws_cov: dict, agreements: list) -> list:
    out = []
    fig, ax = plt.subplots(figsize=(8, 3))
    yrs = sorted(set(full_cov['sessions_per_year']) | set(rws_cov['sessions_per_year']))
    w = 0.4
    ax.bar(np.array(yrs) - w / 2, [full_cov['sessions_per_year'].get(y, 0) for y in yrs], w,
           label='two_week_chunks_full')
    ax.bar(np.array(yrs) + w / 2, [rws_cov['sessions_per_year'].get(y, 0) for y in yrs], w, label='rws_data')
    ax.set_title('Sessions per year'); ax.legend()
    out.append(('Sessions per year', fig_to_b64(fig)))

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(rws_cov['starts'], rws_cov['col_counts'], label='rws_data columns per session')
    ax.plot(full_cov['starts'], full_cov['col_counts'], label='two_week_chunks_full columns per session')
    ax.set_title('Schema over time (columns per session)'); ax.legend()
    out.append(('Schema over time', fig_to_b64(fig)))

    for ag in agreements:
        if ag.get('joined') is None:
            continue
        j = ag['joined']
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(j.index, j['full'], lw=0.6, label='two_week_chunks_full')
        ax.plot(j.index, j['rws'], lw=0.6, label='rws_data')
        ax.set_title(f"{ag['full_chan']}  vs  {ag['rws_chan']}  (corr {ag['corr']:.3f})")
        ax.legend()
        out.append((f"Overlap: {ag['full_chan']}", fig_to_b64(fig)))
    return out


def html_report(sections: list, figs: list) -> str:
    parts = ['<!doctype html><html><head><meta charset="utf-8"><title>EDA: rws_data vs '
             'two_week_chunks_full</title><style>body{font-family:sans-serif;max-width:1100px;margin:2em auto;'
             'padding:0 1em}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;'
             'padding:3px 6px}img{max-width:100%}h2{border-bottom:1px solid #ddd}</style></head><body>',
             '<h1>EDA: rws_data vs two_week_chunks_full</h1>']
    for title, body in sections:
        parts.append(f'<h2>{title}</h2>{body}')
    for title, b64 in figs:
        parts.append(f'<h3>{title}</h3><img src="data:image/png;base64,{b64}"/>')
    parts.append('</body></html>')
    return '\n'.join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--full-dir', default='two_week_chunks_full')
    p.add_argument('--rws-dir', default='rws_data')
    p.add_argument('--full-locations', default='two_week_chunks/locations.csv')
    p.add_argument('--rws-locations', default='rws_data_adapted/locations.csv')
    p.add_argument('--output', default='cuts_plus_prototype/scratch/eda')
    p.add_argument('--match-radius-m', type=float, default=2000.0)
    p.add_argument('--availability-thresholds', type=float, nargs='+', default=[0.10, 0.5])
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)

    full_files = sorted(glob.glob(os.path.join(args.full_dir, '*.parquet')))
    rws_files = sorted(glob.glob(os.path.join(args.rws_dir, '*.parquet')))
    print(f'full: {len(full_files)} sessions, rws: {len(rws_files)} sessions')

    full_label = os.path.basename(os.path.normpath(args.full_dir))
    full_cov = coverage(full_files, full_label)
    rws_cov = coverage(rws_files, 'rws_data')

    full_stats = per_channel_stats(full_files, full_label)
    rws_stats = per_channel_stats(rws_files, 'rws_data')
    stats = pd.concat([full_stats, rws_stats], ignore_index=True)
    stats.to_csv(os.path.join(args.output, 'per_channel_stats.csv'), index=False)

    # Overlap window = the full dataset's span; only rws sessions inside it are compared.
    t0, t1 = pd.Timestamp(full_cov['first']), pd.Timestamp(full_cov['last']) + pd.Timedelta(days=14)
    rws_overlap = [f for f in rws_files if session_bounds(f)[1] >= t0 and session_bounds(f)[0] <= t1]

    full_wl = [c for c in full_stats['channel'] if c.startswith('WL_')]
    full_rh = [c for c in full_stats['channel'] if c.startswith('RH_')]
    rws_norm = {normalize_station(c): c for c in rws_stats['channel']}

    matches, unmatched = [], []
    for c in full_wl:
        key = normalize_station(c)
        hit = rws_norm.get(key)
        how = 'name'
        if hit is None:
            for alias in ALIASES.get(key, []):
                if alias in rws_norm:
                    hit, how = rws_norm[alias], 'alias'
                    break
        if hit is None:
            unmatched.append(c)
        else:
            matches.append({'full_chan': c, 'rws_chan': hit, 'how': how})

    full_loc = load_locations(args.full_locations, 'rd')
    rws_loc = load_locations(args.rws_locations, 'latlon')
    def loc_name(channel: str) -> str:
        # locations.csv keeps the suffix for water level (e.g. 'BG2_observed_waterlevel') and uses
        # rain channel names as-is, matching plot_causal_map.load_locations.
        return channel[len('WL_'):] if channel.startswith('WL_') else channel

    wanted = {loc_name(c) for c in unmatched + full_rh}
    candidates = nearest_rws(full_loc[full_loc['name'].isin(wanted)], rws_loc, args.match_radius_m)

    agreements = []
    for m in matches:
        ag = value_agreement(full_files, rws_overlap, m['full_chan'], m['rws_chan'])
        ag.update(full_chan=m['full_chan'], rws_chan=m['rws_chan'], how=m['how'])
        agreements.append(ag)

    thr_rows = []
    for t in args.availability_thresholds:
        for label, st in [(full_label, full_stats), ('rws_data', rws_stats)]:
            kept = int((st['availability'] >= t).sum())
            thr_rows.append({'threshold': t, 'dataset': label, 'channels_kept': kept, 'channels_total': len(st)})

    ag_table = pd.DataFrame([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in a.items()
                              if k in ('full_chan', 'rws_chan', 'how', 'n', 'corr', 'rmse', 'bias')}
                             for a in agreements]) if agreements else pd.DataFrame()

    sections = []
    sections.append(('1. Coverage and schema', pd.DataFrame([
        {k: v for k, v in c.items() if k in ('label', 'sessions', 'first', 'last', 'columns_min',
                                             'columns_max', 'columns_median', 'duplicate_timestamps')}
        | {'cadence': json.dumps(c['cadence_per_session']), 'timezones': ','.join(c['timezones'])}
        for c in (full_cov, rws_cov)]).to_html(index=False)))
    sections.append(('2. Station overlap', (
        '<h3>Matched water-level channels</h3>' + (pd.DataFrame(matches).to_html(index=False)
                                                  if matches else '<p>none</p>') +
        '<h3>Unmatched full-dataset water-level channels</h3>' + (
            '<p>' + ', '.join(unmatched) + '</p>' if unmatched else '<p>none</p>') +
        '<h3>Rainfall channels (no rws counterpart by construction)</h3><p>' + ', '.join(full_rh) + '</p>' +
        '<h3>Nearest rws site by coordinates (candidates only, not matches)</h3>' +
        candidates.to_html(index=False))))
    sections.append(('3. Per-channel statistics', (
        '<p>Full table in per_channel_stats.csv. Discharge-like = max value &ge; '
        f'{DISCHARGE_LIKE_MAX:g}.</p>' +
        '<h3>Summary by dataset</h3>' + stats.groupby('dataset').agg(
            channels=('channel', 'count'),
            median_availability=('availability', 'median'),
            median_constant_share=('constant_session_share', 'median'),
            discharge_like=('discharge_like', 'sum')).reset_index().to_html(index=False) +
        f'<h3>{full_label} channels</h3>' + full_stats.round(4).to_html(index=False))))
    sections.append(('4. Value agreement on matched stations (overlap window)', (
        '<p>Full series minus rws series after resampling both to a 10-min mean. corr is over the joined '
        'timestamps; bias is mean(full - rws).</p>' +
        (ag_table.to_html(index=False) if len(ag_table) else '<p>no matched stations with overlapping data</p>'))))
    sections.append(('5. Dependence structure', (
        '<p>Pairwise correlations among matched stations are shown in section 4 for the full/rws pair. '
        'Compare the strongest WL-WL edges in each trained graph against these.</p>')))
    sections.append(('6. Effect of the channel availability filter', pd.DataFrame(thr_rows).to_html(index=False)))

    figs = figures(full_cov, rws_cov, agreements)
    with open(os.path.join(args.output, 'eda_report.html'), 'w') as fh:
        fh.write(html_report(sections, figs))
    print(f"Wrote {os.path.join(args.output, 'eda_report.html')}")
    print(f'matched WL: {len(matches)} ({[m["full_chan"] for m in matches]}), unmatched WL: {len(unmatched)}, '
          f'rain channels: {len(full_rh)}')


if __name__ == '__main__':
    main()
