"""
Extends flood_events.py's anomaly detection from its 11 hand-curated gauge stations to every plausible
open-water station in the raw dataset (up to 595), joined to its own municipality, so flood events can
be rated per municipality instead of only nationally from the 11 curated gauges.

Purely additive: flood_events.py's own 11-station national detection and its events.csv/
station_events.csv output are completely untouched - this writes 3 new files alongside them
(candidate_stations.csv, municipality_station_events.csv, municipality_events.csv). None of the 3
confirmed downstream consumers (mask_flood_events.py, flood_labels.py, cuts_plus_rca.py) read anything
this script produces, so nothing about them changes either.

Station classification: channels.csv carries no field distinguishing an open-water station from a
regulated one (a lock/canal/sluice/weir whose level is actively managed by operators, not a flood
signal) - the only usable signal is the station's own free-text name. EXCLUDE_SUBSTRINGS and
BOVEN_BENEDEN_RE below are a conservative, approximate heuristic (confirmed against the real 595 names:
408 candidates / 187 excluded), not an authoritative classification - known false negatives/positives
exist (e.g. "ijmuiden.buitenhaven" is genuinely storm-surge-relevant but excluded by "haven";
"texel.oudeschild" is a harbor not caught by any substring and wrongly included). The 11
DEFAULT_STATIONS are always included unfiltered, bypassing this heuristic entirely - two of them
(dordrecht.oudemaas.benedenmerwede, maastricht.borgharen.maas.beneden) would otherwise be caught by the
boven/beneden pattern, confirmed by direct testing.

Every included non-default station is classified with kind="river" (the Gumbel/return-period path
flood_events.py already uses for its 5 river stations) regardless of true coastal/river nature, since
no official SURGE_CLASSES exist for any station beyond the original 6 coastal ones.

Usage:
    python3 flood_events_municipality.py --out flood_events_output
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape

import flood_geometry as fg
from flood_events import DEFAULT_STATIONS, Station, load_local_series, merge_national, station_events

EXCLUDE_SUBSTRINGS = ["sluis", "kanaal", "haven", "gemaal", "stuw", "spui", "inlaat", "uitlaat"]
# Token-boundary match only (e.g. "beneden" as its own dotted segment) - scoped to new candidates only,
# never applied to the 11 DEFAULT_STATIONS (see module docstring for the 2 real counterexamples).
BOVEN_BENEDEN_RE = re.compile(r"(^|\.)(boven|beneden)(\.|$)")

# flood_events.clean() already drops flagged-missing readings (|value| >= 1e8, RWS's 999999999
# sentinel); confirmed by direct inspection that at least 4 of the new candidate stations
# (huibertgat, dodewaard, mook, rhenen.grebbeberg) use a second, much smaller sentinel - exactly
# 999 - that clean() doesn't catch (huibertgat alone: 687/2.4M readings are exactly 999.0, next to an
# otherwise plausible ~13cm median - a stuck/fault value, not a real reading; it drove 7 of
# huibertgat's "high" alerts in an early run of this script). None of the 11 DEFAULT_STATIONS exhibit
# this, so the filter is scoped here rather than added to flood_events.py's shared clean().
LOCAL_SENTINEL = 999.0


def select_open_water_candidates(locations: pd.DataFrame, channels: pd.DataFrame) -> tuple:
    """Returns (stations, audit_df). stations: every station considered open-water (the 11
    DEFAULT_STATIONS unfiltered, plus every other station not matched by the exclusion heuristic).
    audit_df: all 595 rows, with code/name/included/exclusion_reason - the full audit trail."""
    default_kind = {code: kind for kind, pairs in DEFAULT_STATIONS.items() for code, _ in pairs}
    display_name = channels.set_index("Code")["Naam"].to_dict()

    stations, rows = [], []
    for code in locations["name"]:
        name = display_name.get(code, code)
        if code in default_kind:
            stations.append(Station(code, name, default_kind[code]))
            rows.append({"code": code, "name": name, "included": True,
                         "exclusion_reason": "already_default"})
            continue
        lower = code.lower()
        reason = next((f"substring:{w}" for w in EXCLUDE_SUBSTRINGS if w in lower), None)
        if reason is None and BOVEN_BENEDEN_RE.search(lower):
            reason = "boven_beneden"
        if reason is None:
            stations.append(Station(code, name, "river"))
        rows.append({"code": code, "name": name, "included": reason is None, "exclusion_reason": reason})
    return stations, pd.DataFrame(rows)


def _nearest_with_distance_km(anchor_lonlat: tuple, municipalities: list) -> tuple:
    anchor_rd = fg._reproject(Point(anchor_lonlat), fg.TO_RD)
    best, best_dist = None, float("inf")
    for feature in municipalities:
        dist = fg._reproject(shape(feature["geometry"]), fg.TO_RD).distance(anchor_rd)
        if dist < best_dist:
            best, best_dist = feature, dist
    return best, best_dist / 1000


def assign_municipalities(codes: list, locations: pd.DataFrame, municipalities: list,
                           max_join_km: float) -> pd.DataFrame:
    """One row per code: municipality_code, municipality_naam, join_distance_km. Beyond max_join_km
    (offshore platforms, foreign stations like Antwerp) gets exclusion_reason="offshore_or_outside_nl"
    and no municipality - nearest_municipalities() always returns *some* polygon regardless of
    distance, so without this cutoff an offshore station would get force-matched to a wrong one."""
    locations_indexed = locations.set_index("name")
    rows = []
    for code in codes:
        lonlat = fg.station_lonlat(code, locations_indexed)
        muni, dist_km = _nearest_with_distance_km(lonlat, municipalities)
        offshore = dist_km > max_join_km
        rows.append({
            "code": code,
            "municipality_code": None if offshore else muni["properties"]["code"],
            "municipality_naam": None if offshore else muni["properties"]["naam"],
            "join_distance_km": round(dist_km, 2),
            "offshore_or_outside_nl": offshore,
        })
    return pd.DataFrame(rows).set_index("code")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locations-csv", default="rws_waterinfo/locations.csv")
    parser.add_argument("--channels-csv", default="rws_waterinfo/channels.csv")
    parser.add_argument("--data-dir", default="rws_waterinfo_adapted")
    parser.add_argument("--municipalities", default="reference_data/municipalities_2024_national.geojson")
    parser.add_argument("--max-join-km", type=float, default=15.0,
                         help="beyond this distance to the nearest municipality, a station is "
                              "treated as offshore/foreign and dropped (catches e.g. the a12 North "
                              "Sea platform, the Antwerp stations)")
    parser.add_argument("--merge-days", type=int, default=10,
                         help="flat gap (days) for merging same-municipality station events into one "
                              "municipality event - a municipality may mix coastal+river stations, so "
                              "there's no single kind to look up a per-kind gap for")
    parser.add_argument("--river-periods", default="2,10,50")
    parser.add_argument("--start", default="1950-01-01")
    parser.add_argument("--end", default=pd.Timestamp.now().strftime("%Y-%m-%d"))
    parser.add_argument("--out", default="flood_events_output")
    args = parser.parse_args()

    periods = [float(x) for x in args.river_periods.split(",")]
    if len(periods) != 3 or not periods[0] < periods[1] < periods[2]:
        parser.error("--river-periods needs three increasing numbers, e.g. 2,10,50")
    start = pd.Timestamp(args.start, tz="Europe/Amsterdam")
    end = pd.Timestamp(args.end, tz="Europe/Amsterdam")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    locations = pd.read_csv(args.locations_csv)
    channels = pd.read_csv(args.channels_csv)
    stations, audit = select_open_water_candidates(locations, channels)
    print(f"{len(stations)}/{len(locations)} stations selected as open-water candidates "
          f"(including the 11 DEFAULT_STATIONS)")

    with open(args.municipalities) as f:
        municipalities = json.load(f)["features"]
    muni = assign_municipalities(locations["name"].tolist(), locations, municipalities, args.max_join_km)

    audit = audit.set_index("code").join(muni).reset_index()
    audit.loc[audit["offshore_or_outside_nl"], "exclusion_reason"] = "offshore_or_outside_nl"
    audit.loc[audit["offshore_or_outside_nl"], "included"] = False
    audit.to_csv(out / "candidate_stations.csv", index=False)
    print(f"Wrote {len(audit)}-row station audit -> {out / 'candidate_stations.csv'}")

    detection_stations = [st for st in stations if not muni.loc[st.code, "offshore_or_outside_nl"]]
    print(f"Running peak detection on {len(detection_stations)} station(s) "
          f"({len(stations) - len(detection_stations)} dropped as offshore/outside NL)...")

    all_events = []
    for i, st in enumerate(detection_stations, 1):
        print(f"  [{i}/{len(detection_stations)}] {st.name} ({st.code})...")
        s = load_local_series(st.code, data_dir)
        if s.empty:
            continue
        s = s[(s.index >= start) & (s.index < end)]
        s = s[s != LOCAL_SENTINEL]
        if s.empty:
            continue
        try:
            ev = station_events(st, s, periods)
        except ValueError as err:
            print(f"    skipping: {err}")
            continue
        if ev.empty:
            continue
        ev["municipality_code"] = muni.loc[st.code, "municipality_code"]
        ev["municipality_naam"] = muni.loc[st.code, "municipality_naam"]
        ev["join_distance_km"] = muni.loc[st.code, "join_distance_km"]
        all_events.append(ev)

    if not all_events:
        print("No events detected at any station.")
        return 0
    per_station = pd.concat(all_events, ignore_index=True)
    per_station.to_csv(out / "municipality_station_events.csv", index=False)
    print(f"Wrote {len(per_station)} station-level exceedance(s) -> "
          f"{out / 'municipality_station_events.csv'}")

    table = merge_national(per_station, group_col="municipality_code", gap_days=args.merge_days)
    # merge_national() only emits the group_col itself (municipality_code) - municipality_naam is a
    # per-station attribute (same for every row within one group, since they're all in the same
    # municipality), so it never gets carried into the merged output on its own. Attach it back via a
    # simple 1:1 lookup rather than teaching the shared merge_national() about this one companion
    # column.
    naam_by_code = per_station.drop_duplicates("municipality_code").set_index(
        "municipality_code")["municipality_naam"]
    table.insert(table.columns.get_loc("municipality_code") + 1, "municipality_naam",
                 table["municipality_code"].map(naam_by_code))
    table.to_csv(out / "municipality_events.csv", index=False)
    counts = table["alert"].value_counts()
    print(f"Wrote {len(table)} municipality-level event(s) across "
          f"{table['municipality_code'].nunique()} municipalit(y/ies) -> {out / 'municipality_events.csv'}")
    print(f"  alert mix: high={counts.get('high', 0)}, medium={counts.get('medium', 0)}, "
          f"low={counts.get('low', 0)}")

    write_training_tables(table, per_station, out / "municipality_high")
    return 0


def _day_coverage(df: pd.DataFrame) -> float:
    """Fraction of the full min(start)..max(end) span covered by the union of this table's own
    [start, end] windows - used to sanity-check how much of the dataset a given alert threshold would
    actually mask/reroute if fed into mask_flood_events.py/cuts_plus_rca.py's --flood-aware-split.
    Confirmed this session: municipality_events.csv covers 100% of all days at low+, 27.8% at medium+,
    but only 6.8% restricted to high - which is why the training-pipeline swap uses high-alert only."""
    if df.empty:
        return 0.0
    start0 = df["start"].min()
    span_days = (df["end"].max() - start0).days + 1
    hits = np.zeros(span_days, dtype=bool)
    for s, e in zip(df["start"], df["end"]):
        hits[(s - start0).days:(e - start0).days + 1] = True
    return float(hits.mean())


def write_training_tables(municipality_events: pd.DataFrame, municipality_station_events: pd.DataFrame,
                           out_dir: Path) -> None:
    """Filters both municipality-level tables to alert=="high" and writes them under the exact
    filenames cuts_plus_rca.py's --flood-events-csv expects (events.csv + a sibling station_events.csv,
    its path hardcoded as os.path.join(dirname(flood_events_csv), 'station_events.csv')) - so the
    training pipeline needs zero code changes to read this directly in place of the original
    11-station national table. Restricted to "high" because lower thresholds cover an implausibly
    large fraction of the whole dataset (confirmed this session: 27.8% at medium+, 100% at low+, vs.
    the national table's 1.3%/4.6% - see _day_coverage); "high" alone (6.8%) is the closest match to
    the national table's existing scale."""
    out_dir.mkdir(parents=True, exist_ok=True)
    high_events = municipality_events[municipality_events["alert"] == "high"]
    high_stations = municipality_station_events[municipality_station_events["alert"] == "high"]
    high_events.to_csv(out_dir / "events.csv", index=False)
    high_stations.to_csv(out_dir / "station_events.csv", index=False)

    events_for_coverage = high_events.copy()
    events_for_coverage["start"] = pd.to_datetime(events_for_coverage["start"])
    events_for_coverage["end"] = pd.to_datetime(events_for_coverage["end"])
    coverage = _day_coverage(events_for_coverage)
    print(f"Wrote {len(high_events)} high-alert municipality event(s) "
          f"({len(high_stations)} station-level row(s)) -> {out_dir}/events.csv + station_events.csv")
    print(f"  day-coverage at high-alert: {coverage:.1%} (sanity check - should stay well under the "
          f"27.8%/100% seen at medium+/low+, not balloon back toward them)")


if __name__ == "__main__":
    raise SystemExit(main())
