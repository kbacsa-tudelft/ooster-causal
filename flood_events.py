#!/usr/bin/env python3
"""
flood_events.py - Build a table of low / medium / high alert flood events in the
Netherlands from Rijkswaterstaat water level measurements.

Two kinds of stations are handled:

  coast  Storm surges. Classified with Rijkswaterstaat's official storm surge
         classes (stormvloedklassen, situation 1-1-2015, as printed in every
         "Stormvloedflits" report):
             low    = lage stormvloed
             medium = middelbare stormvloed
             high   = hoge, zeer hoge or extreme stormvloed

  river  River high water (and canals). Rijkswaterstaat has no equivalent
         classes, so levels are classified by return period, estimated from the
         station's own annual maxima with a Gumbel fit:
             low    >= 2-year level
             medium >= 10-year level
             high   >= 50-year level
         The return periods can be changed with --river-periods.

Data sources
  * Local (default): the occluded rws water-level dataset this project already
    downloads and cleans (cuts_plus_prototype/download_rws_waterlevel.py +
    prepare_rws_waterinfo.py), via --data-dir (default rws_waterinfo_adapted,
    columns WL_<code>). No network call - this script used to hit the
    Rijkswaterstaat DDL API (waterwebservices.rijkswaterstaat.nl), but that
    service has been decommissioned (it now 301-redirects to a page that is
    itself a 404), so the online path was removed rather than chasing a
    replacement endpoint.
  * Offline: CSV exports from https://waterinfo.rws.nl (or any CSV with a
    date/time column and a water level column in cm NAP), via --csv. Useful
    for history before rws_waterinfo_adapted's own coverage starts.

Caveat: the Gumbel return-period fit for river stations wants a long record
(its annual maxima are the whole signal). Coverage varies by station - the
local dataset can reach back to the 1950s for some, but others only have a
few decades, or come online much later - so check a station's actual span
(printed in the coverage line) before trusting its 50-year level; a short
record makes that level a real extrapolation, not an official RWS figure.
Storm surge classification is unaffected: it uses RWS's fixed cm NAP
thresholds, not a fit.

Examples
  # Full table over the local dataset's coverage, all default stations
  python flood_events.py

  # Only the coast, written to a different folder
  python flood_events.py --stations coast --out results

  # Add your own station from the local dataset (code from its channels.csv)
  python flood_events.py --extra some.station.code:river

  # Offline, from Waterinfo exports (e.g. for history before the local dataset's coverage)
  python flood_events.py --csv lobith.csv:LOBH:river --csv delfzijl.csv:DELFZL:coast

Output (in --out, default ./flood_events_output)
  events.csv       one row per national event, with its alert class
  events.md        the same table in Markdown, split into high / medium / low
  station_events.csv  one row per station per event (peak level, class)

Requires: Python 3.9+, pandas, pyarrow
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MISSING_SENTINEL = 1e8  # RWS uses 999999999 for missing values; kept as a safety net for --csv

# Official storm surge classes in cm NAP (lower bound of each class).
# Source: RWS Stormvloedflits reports, classification table, situation 1-1-2015.
# "zeer hoge" starts where "hoge" ends; "extreme" bound as published.
# Keyed by the local dataset's station code (see DEFAULT_STATIONS) rather than the old DDL code.
SURGE_CLASSES = {
    #                                    lage  middelbare  hoge  zeer_hoog  extreme
    "vlissingen":                       (350, 385, 440, 490, 550),
    "hoekvanholland":                   (260, 300, 360, 430, 510),
    "dordrecht.oudemaas.benedenmerwede": (215, 245, 275, 295, 315),
    "denhelder.marsdiep":               (230, 275, 340, 400, 450),
    "harlingen.waddenzee":              (305, 355, 415, 470, 510),
    "delfzijl":                         (355, 420, 505, 580, 640),
}
SURGE_LABELS = ["lage", "middelbare", "hoge", "zeer hoge", "extreme"]

# Default stations, as codes in the local rws_waterinfo_adapted dataset (columns are WL_<code>).
# Picked by name match against channels.csv; where a name matches several gauges, the open-water/
# main-stem one was picked over a harbor, canal, or inlet variant (e.g. Den Helder's open-sea
# Marsdiep gauge over its sheltered ferry-berth one; Lobith's and Borgharen's main-river gauges,
# which also turned up as the discharge stations sampled earlier in this project - a good sign
# they're the standard ones RWS treats as "Lobith"/"Borgharen").
DEFAULT_STATIONS = {
    "coast": [
        ("vlissingen", "Vlissingen"),
        ("hoekvanholland", "Hoek van Holland"),
        ("dordrecht.oudemaas.benedenmerwede", "Dordrecht"),
        ("denhelder.marsdiep", "Den Helder"),
        ("harlingen.waddenzee", "Harlingen"),
        ("delfzijl", "Delfzijl"),
    ],
    "river": [
        ("lobith.bovenrijn.tolkamer", "Lobith"),
        ("maastricht.borgharen.maas.beneden", "Borgharen"),
        ("venlo", "Venlo"),
        ("nijmegen.waal", "Nijmegen"),
        ("deventer", "Deventer"),
    ],
}

# Events at different stations are merged into one national event when their
# peaks are this many days apart or less.
MERGE_DAYS = {"coast": 2, "river": 10}
# Minimum gap between two separate events at the same station.
DECLUSTER_DAYS = {"coast": 2, "river": 7}

CLASS_ORDER = {"low": 1, "medium": 2, "high": 3}


@dataclass
class Station:
    code: str
    name: str
    kind: str          # "coast" or "river"


# --------------------------------------------------------------------------
# Reading from the local rws water-level dataset (WL_<code> columns)
# --------------------------------------------------------------------------

def load_local_series(code: str, data_dir: Path) -> pd.Series:
    """Concatenates one station's WL_<code> column across every session parquet in data_dir
    (e.g. rws_waterinfo_adapted/). Uses pyarrow to check each file's schema first, so sessions
    that don't have this column (the station wasn't active then) are skipped without reading
    their data - the same cheap-schema-check style prepare_data.py uses."""
    col = f"WL_{code}"
    parts = []
    for f in sorted(data_dir.glob("*.parquet")):
        if col not in pq.ParquetFile(f).schema_arrow.names:
            continue
        parts.append(pd.read_parquet(f, columns=[col])[col])
    if not parts:
        return pd.Series(dtype=float)
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated(keep="first")].dropna()
    # the data is Dutch local time but stored tz-naive; localize it so it compares correctly
    # against the rest of this script's tz-aware timestamps (same convention as read_csv_series)
    s.index = pd.DatetimeIndex(s.index).tz_localize("Europe/Amsterdam", ambiguous="NaT", nonexistent="NaT")
    return clean(s[s.index.notna()])


# --------------------------------------------------------------------------
# Reading offline CSVs (Waterinfo exports or generic)
# --------------------------------------------------------------------------

def read_csv_series(path: str) -> pd.Series:
    """Read a Waterinfo export (';' separated, Dutch headers) or a generic CSV
    with a datetime column and a numeric level column in cm NAP."""
    raw = pd.read_csv(path, sep=None, engine="python", decimal=",", encoding="latin-1")
    cols = {c.upper(): c for c in raw.columns}
    if "WAARNEMINGDATUM" in cols:  # Waterinfo export
        date = raw[cols["WAARNEMINGDATUM"]].astype(str)
        tcol = next((cols[c] for c in cols if c.startswith("WAARNEMINGTIJD")), None)
        stamp = date + " " + raw[tcol].astype(str) if tcol else date
        times = pd.to_datetime(stamp, dayfirst=True, errors="coerce")
        values = pd.to_numeric(raw[cols.get("NUMERIEKEWAARDE", raw.columns[-1])], errors="coerce")
    else:  # generic: first parseable datetime column, last numeric column
        tcol = next(c for c in raw.columns
                    if pd.to_datetime(raw[c], errors="coerce").notna().mean() > 0.9)
        vcol = [c for c in raw.columns if c != tcol
                and pd.to_numeric(raw[c], errors="coerce").notna().mean() > 0.9][-1]
        times = pd.to_datetime(raw[tcol], errors="coerce")
        values = pd.to_numeric(raw[vcol], errors="coerce")
    s = pd.Series(values.values, index=times).dropna()
    s.index = s.index.tz_localize("Europe/Amsterdam", ambiguous="NaT", nonexistent="NaT") \
        if s.index.tz is None else s.index
    return clean(s[s.index.notna()])


def clean(s: pd.Series) -> pd.Series:
    s = s[(s.abs() < MISSING_SENTINEL)]
    return s.sort_index()


# --------------------------------------------------------------------------
# Event detection and classification
# --------------------------------------------------------------------------

def peaks_over_threshold(s: pd.Series, threshold: float, gap_days: int) -> pd.DataFrame:
    """Independent peaks above threshold: values above it are grouped into one
    event as long as consecutive exceedances are less than gap_days apart."""
    above = s[s >= threshold]
    if above.empty:
        return pd.DataFrame(columns=["start", "end", "peak_time", "peak_level"])
    new_event = above.index.to_series().diff() > pd.Timedelta(days=gap_days)
    event_id = new_event.cumsum()
    out = []
    for _, grp in above.groupby(event_id.values):
        out.append({
            "start": grp.index.min(),
            "end": grp.index.max(),
            "peak_time": grp.idxmax(),
            "peak_level": float(grp.max()),
        })
    return pd.DataFrame(out)


def classify_surge(code: str, level: float) -> tuple[str | None, str | None]:
    bounds = SURGE_CLASSES[code]
    label = None
    for b, name in zip(bounds, SURGE_LABELS):
        if level >= b:
            label = name
    if label is None:
        return None, None
    alert = {"lage": "low", "middelbare": "medium"}.get(label, "high")
    return alert, label + " stormvloed"


def gumbel_levels(s: pd.Series, periods: list[float]) -> dict[float, float]:
    """Return levels for the given return periods (years) from a Gumbel fit to
    annual maxima (hydrological year starting 1 October). Years with less than
    70% coverage are dropped."""
    shifted = s.copy()
    # This grouping only needs wall-clock calendar dates, never an absolute instant, so the index
    # is made tz-naive first: subtracting DateOffset from a tz-aware index re-localizes into its own
    # timezone afterwards, which raises if the shifted date lands on that timezone's DST transition
    # (e.g. shifting a 2005 date back 9 months can land on 2004-10-31, the Netherlands' DST fall-back
    # that year) - a latent bug, not something specific to this project's data.
    idx = pd.DatetimeIndex(shifted.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    shifted.index = idx - pd.DateOffset(months=9)  # Oct -> Jan
    yearly = shifted.groupby(shifted.index.year)
    counts = yearly.size()
    typical = counts.median()
    am = yearly.max()[counts >= 0.7 * typical]
    if len(am) < 10:
        raise ValueError(f"only {len(am)} complete years; need at least 10 for return periods")
    mean, std = am.mean(), am.std(ddof=1)
    beta = std * math.sqrt(6) / math.pi
    mu = mean - 0.5772 * beta
    return {T: mu - beta * math.log(-math.log(1 - 1 / T)) for T in periods}


def station_events(st: Station, s: pd.Series, river_periods: list[float]) -> pd.DataFrame:
    if s.empty:
        return pd.DataFrame()
    if st.kind == "coast":
        if st.code not in SURGE_CLASSES:
            raise ValueError(f"No official storm surge classes for {st.code}; use kind 'river'.")
        threshold = SURGE_CLASSES[st.code][0]
        ev = peaks_over_threshold(s, threshold, DECLUSTER_DAYS["coast"])
        if ev.empty:
            return ev
        cls = ev["peak_level"].apply(lambda v: classify_surge(st.code, v))
        ev["alert"] = [c[0] for c in cls]
        ev["detail"] = [c[1] for c in cls]
    else:
        levels = gumbel_levels(s, river_periods)
        t_low, t_med, t_high = river_periods
        ev = peaks_over_threshold(s, levels[t_low], DECLUSTER_DAYS["river"])
        if ev.empty:
            return ev

        def cls(v: float) -> str:
            if v >= levels[t_high]:
                return "high"
            if v >= levels[t_med]:
                return "medium"
            return "low"
        ev["alert"] = ev["peak_level"].apply(cls)
        ev["detail"] = ev["peak_level"].apply(
            lambda v: f">= {max(T for T in river_periods if v >= levels[T]):g}-year level")
    ev.insert(0, "station", st.name)
    ev.insert(1, "code", st.code)
    ev.insert(2, "kind", st.kind)
    return ev


def merge_national(ev: pd.DataFrame) -> pd.DataFrame:
    """Merge station events of the same kind whose peaks are close in time."""
    rows = []
    for kind, grp in ev.groupby("kind"):
        grp = grp.sort_values("peak_time")
        gap = grp["peak_time"].diff() > pd.Timedelta(days=MERGE_DAYS[kind])
        for _, g in grp.groupby(gap.cumsum().values):
            top = g.loc[g["alert"].map(CLASS_ORDER).idxmax()]
            rows.append({
                "start": g["start"].min().date(),
                "end": g["end"].max().date(),
                "peak_date": top["peak_time"].date(),
                "type": "storm surge" if kind == "coast" else "river high water",
                "alert": top["alert"],
                "worst_station": top["station"],
                "worst_detail": top["detail"],
                "worst_level_cm_nap": round(top["peak_level"]),
                "stations": ", ".join(
                    f"{r.station} {round(r.peak_level)} ({r.alert})"
                    for r in g.sort_values("peak_level", ascending=False).itertuples()),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_rank"] = out["alert"].map(CLASS_ORDER)
    return out.sort_values(["_rank", "peak_date"], ascending=[False, True]).drop(columns="_rank")


def write_markdown(table: pd.DataFrame, path: Path, coverage: dict[str, str]) -> None:
    lines = ["# Flood events in the Netherlands by alert class", ""]
    lines.append("Station data coverage: " + "; ".join(f"{k} {v}" for k, v in coverage.items()))
    lines.append("")
    cols = ["peak_date", "start", "end", "type", "worst_station", "worst_level_cm_nap",
            "worst_detail", "stations"]
    for alert in ["high", "medium", "low"]:
        part = table[table["alert"] == alert].sort_values("peak_date")
        lines.append(f"## {alert.capitalize()} alert ({len(part)} events)")
        lines.append("")
        if part.empty:
            lines.append("None found.")
            lines.append("")
            continue
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for r in part[cols].itertuples(index=False):
            lines.append("| " + " | ".join(str(v) for v in r) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="1950-01-01",
                   help="first date (default 1950-01-01; a station's own data may start later - "
                        "see its coverage line in the output)")
    p.add_argument("--end", default=pd.Timestamp.now().strftime("%Y-%m-%d"), help="last date (default today)")
    p.add_argument("--stations", choices=["all", "coast", "river", "none"], default="all",
                   help="which default stations to include (default all)")
    p.add_argument("--extra", action="append", default=[], metavar="CODE:KIND",
                   help="extra local station, e.g. some.station.code:river (repeatable)")
    p.add_argument("--csv", action="append", default=[], metavar="FILE:CODE:KIND",
                   help="offline data file instead of the local dataset (repeatable)")
    p.add_argument("--data-dir", default="rws_waterinfo_adapted",
                   help="occluded local water-level dataset to read from (default rws_waterinfo_adapted)")
    p.add_argument("--river-periods", default="2,10,50",
                   help="return periods in years for low,medium,high river alerts (default 2,10,50)")
    p.add_argument("--out", default="flood_events_output", help="output folder")
    args = p.parse_args(argv)

    start, end = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)
    periods = [float(x) for x in args.river_periods.split(",")]
    if len(periods) != 3 or not periods[0] < periods[1] < periods[2]:
        p.error("--river-periods needs three increasing numbers, e.g. 2,10,50")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    series: list[tuple[Station, pd.Series]] = []

    # Offline files
    for spec in args.csv:
        try:
            file, code, kind = spec.rsplit(":", 2)
        except ValueError:
            p.error(f"--csv needs FILE:CODE:KIND, got {spec!r}")
        st = Station(code.upper(), code.upper(), kind)
        series.append((st, read_csv_series(file)))

    # Local dataset stations
    wanted: list[Station] = []
    if not args.csv and args.stations != "none":
        kinds = ["coast", "river"] if args.stations == "all" else [args.stations]
        for k in kinds:
            wanted += [Station(c, n, k) for c, n in DEFAULT_STATIONS[k]]
    for spec in args.extra:
        code, _, kind = spec.partition(":")
        wanted.append(Station(code, code, kind or "river"))

    if wanted and not data_dir.exists():
        print(f"{data_dir} does not exist - run download_rws_waterlevel.py and prepare_rws_waterinfo.py "
              "first, or pass --data-dir, or use --csv instead.")
        return 2
    for st in wanted:
        print(f"Loading {st.name} ({st.code}) from {data_dir}...")
        s = load_local_series(st.code, data_dir)
        if s.empty:
            print(f"  no local data for {st.code} - skipping")
            continue
        series.append((st, s))

    all_events, coverage = [], {}
    for st, s in series:
        s = s[(s.index >= start) & (s.index < end)]
        if s.empty:
            print(f"  no data for {st.name} in the period")
            continue
        coverage[st.name] = f"{s.index.min():%Y}-{s.index.max():%Y}"
        try:
            ev = station_events(st, s, periods)
        except ValueError as err:
            print(f"  skipping {st.name}: {err}")
            continue
        if not ev.empty:
            all_events.append(ev)

    if not all_events:
        print("No events found.")
        return 1

    per_station = pd.concat(all_events, ignore_index=True)
    per_station.to_csv(out / "station_events.csv", index=False)
    table = merge_national(per_station)
    table.to_csv(out / "events.csv", index=False)
    write_markdown(table, out / "events.md", coverage)

    counts = table["alert"].value_counts()
    print(f"\nDone: {len(table)} events "
          f"(high {counts.get('high', 0)}, medium {counts.get('medium', 0)}, low {counts.get('low', 0)}).")
    print(f"Results in {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
