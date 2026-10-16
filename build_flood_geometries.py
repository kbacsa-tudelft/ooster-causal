"""
Builds a flood-extent GeoJSON for every event in flood_events.py's computed national catalog
(events.csv), by either reusing a real digitized ROR polygon (for the 4 events that directly correspond
to one of the ROR report's 10 historical floods) or synthesizing one from real municipality boundaries,
scaled by severity relative to the nearest matching ROR event (see flood_geometry.py's docstring for
the full method).

Only 6 of the 10 ROR events are valid matching candidates: the other 4 ("Regionaal watersysteem" -
canal breaks, a peat-dike slide, a water-main break) are a different flood mechanism than anything the
11-station national gauge network this project's events.csv is built from can ever detect, confirmed by
their centroids being small, locally-unrelated patches unrelated to the national river/coastal system.

Each of the 6 candidate ROR events' own reach profile (flood_geometry.learn_reach_profile) is measured
against the municipalities that actually existed in that event's own year (1916/1926/1953/1993/1995 -
see fetch_municipalities.py), not today's much-merged ones, since the real polygon's reach should be
read against the administrative units that existed when it happened. Every synthesized (non-template)
event instead uses a single fixed year (1820 by default, the most municipality-dense year available -
see fetch_municipalities.py's docstring) for all ~257 of them uniformly: using each event's own
contemporary year here would need ~70 separate boundary files for marginal gain, and visualization
granularity matters more than per-event historical accuracy for this part (explicit user choice).

Usage:
    python3 build_flood_geometries.py --events-csv flood_events_output/events.csv \
        --ror-events-geojson flood_events_output/ror_historical_events.geojson \
        --synthesis-municipalities reference_data/municipalities_1820.geojson \
        --municipalities-dir reference_data \
        --out flood_events_output/flood_events.geojson
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

import flood_geometry as fg
from flood_events import DEFAULT_STATIONS


def load_ror_candidates(ror_events_geojson: str) -> dict:
    with open(ror_events_geojson) as f:
        collection = json.load(f)
    candidate_nrs = set(fg.STORM_SURGE_CANDIDATES) | set(fg.RIVER_CANDIDATES)
    by_nr = {}
    for feature in collection["features"]:
        nr = feature["properties"]["nr"]
        if nr in candidate_nrs:
            by_nr[nr] = feature
    missing = candidate_nrs - by_nr.keys()
    if missing:
        raise ValueError(f"ROR events geojson is missing candidate nr(s): {sorted(missing)}")
    return by_nr


def event_type_for_nr(nr: int) -> str:
    return "storm surge" if nr in fg.STORM_SURGE_CANDIDATES else "river high water"


def find_direct_match_rows(events: pd.DataFrame) -> dict:
    """nr -> row index of the events.csv row that IS this same real historical flood (same year+type;
    when several rows share a year+type, e.g. a minor unrelated event earlier the same year, the
    highest-severity one is the real ROR-documented flood - the ROR report only covers events with
    'significante negatieve effecten')."""
    row_for_nr = {}
    for nr, (year, event_type) in fg.DIRECT_MATCH_YEAR_TYPE.items():
        candidates = events[(events["peak_date"].dt.year == year) & (events["type"] == event_type)]
        if candidates.empty:
            print(f"  nr {nr} ({year} {event_type}): no matching events.csv row found - will use the "
                  f"fallback severity reference instead")
            continue
        best_idx = candidates["worst_level_cm_nap"].idxmax()
        row_for_nr[nr] = best_idx
    return row_for_nr


def fallback_reference_level(nr: int, ror_centroids: dict, events: pd.DataFrame, locations,
                              name_to_code: dict) -> float:
    """nr 1/2 predate events.csv's 1950-04-24 coverage, so no directly-matching row exists - fall back
    to the nearest candidate station's own worst-ever recorded level for that event type."""
    event_type = event_type_for_nr(nr)
    station_codes = [c for c, _ in DEFAULT_STATIONS["coast" if event_type == "storm surge" else "river"]]
    anchor_code = fg.nearest_station(
        (ror_centroids[nr].x, ror_centroids[nr].y), station_codes, locations)
    anchor_name = {code: name for name, code in name_to_code.items()}[anchor_code]
    same_type = events[events["type"] == event_type]
    at_station = same_type[same_type["worst_station"] == anchor_name]
    if not at_station.empty:
        return float(at_station["worst_level_cm_nap"].max())
    return float(same_type["worst_level_cm_nap"].max())  # station never recorded worst - use type max


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-csv", default="flood_events_output/events.csv")
    parser.add_argument("--ror-events-geojson", default="flood_events_output/ror_historical_events.geojson")
    parser.add_argument("--synthesis-municipalities", default="reference_data/municipalities_1820.geojson",
                         help="the single municipality set used for every synthesized (non-template) "
                              "event's extent")
    parser.add_argument("--municipalities-dir", default="reference_data",
                         help="directory holding municipalities_<year>.geojson for each of the 6 "
                              "candidate ROR events' own years, used only for reach-profile learning")
    parser.add_argument("--locations-csv", default="rws_waterinfo/locations.csv")
    parser.add_argument("--out", default="flood_events_output/flood_events.geojson")
    args = parser.parse_args()

    events = pd.read_csv(args.events_csv)
    events["peak_date"] = pd.to_datetime(events["peak_date"])
    locations = pd.read_csv(args.locations_csv, index_col="name")
    with open(args.synthesis_municipalities) as f:
        synthesis_municipalities = json.load(f)["features"]
    ror_by_nr = load_ror_candidates(args.ror_events_geojson)
    ror_geom_by_nr = {nr: shape(f["geometry"]) for nr, f in ror_by_nr.items()}
    ror_centroids = {nr: g.centroid for nr, g in ror_geom_by_nr.items()}
    name_to_code = fg.station_name_to_code()

    print("Matching ROR events to computed events.csv rows:")
    direct_match_row = find_direct_match_rows(events)
    # Invert: which nr(s) does each matched row correspond to (row 38 maps to both nr 5 and 6, since
    # the real Jan 1995 flood hit both the Maas and the Rhine and flood_events.py's merge_national()
    # groups all "river" kind stations peaking close in time into one national event).
    nrs_for_row = {}
    for nr, row_idx in direct_match_row.items():
        nrs_for_row.setdefault(row_idx, []).append(nr)
    for row_idx, nrs in nrs_for_row.items():
        print(f"  row {row_idx} (peak_date={events.loc[row_idx, 'peak_date'].date()}) "
              f"<- ROR nr {nrs}")

    print("Reference severity levels and reach profiles for the 6 candidate ROR events:")
    reference_level, reach_profile, anchor_code_for_nr = {}, {}, {}
    for nr in sorted(ror_by_nr):
        event_type = event_type_for_nr(nr)
        if nr in direct_match_row:
            reference_level[nr] = float(events.loc[direct_match_row[nr], "worst_level_cm_nap"])
        else:
            reference_level[nr] = fallback_reference_level(
                nr, ror_centroids, events, locations, name_to_code)
        station_codes = [c for c, _ in DEFAULT_STATIONS["coast" if event_type == "storm surge" else "river"]]
        anchor_code_for_nr[nr] = fg.nearest_station(
            (ror_centroids[nr].x, ror_centroids[nr].y), station_codes, locations)
        anchor_lonlat = fg.station_lonlat(anchor_code_for_nr[nr], locations)

        # This nr's own historical year's boundaries, not today's much-merged ones (see module
        # docstring) - the real ROR polygon's reach should be measured against the administrative
        # units that existed when that flood actually happened.
        template_year = ror_by_nr[nr]["properties"]["jaar"]
        municipalities_path = os.path.join(args.municipalities_dir, f"municipalities_{template_year}.geojson")
        with open(municipalities_path) as f:
            template_municipalities = json.load(f)["features"]

        reach_profile[nr] = fg.learn_reach_profile(
            ror_geom_by_nr[nr], template_municipalities, event_type, anchor_lonlat, locations)
        print(f"  nr {nr} ({template_year}): reference_level={reference_level[nr]:.0f}, "
              f"anchor={anchor_code_for_nr[nr]}, profile={reach_profile[nr]}")

    features = []
    n_digitized, n_derived = 0, 0
    fractions = []
    for row_idx, row in events.iterrows():
        props = row.to_dict()
        props["peak_date"] = str(row["peak_date"].date())

        if row_idx in nrs_for_row:
            nrs = nrs_for_row[row_idx]
            geometry = ror_geom_by_nr[nrs[0]] if len(nrs) == 1 else \
                unary_union([ror_geom_by_nr[nr] for nr in nrs])
            props["matched_nr"] = ";".join(str(nr) for nr in nrs)
            props["extent_source"] = "digitized"
            props["severity_fraction"] = 1.0
            props["affected_municipalities"] = ""
            n_digitized += 1
        else:
            anchor_code = name_to_code[row["worst_station"]]
            anchor_lonlat = fg.station_lonlat(anchor_code, locations)
            nr = fg.nearest_template(row["type"], anchor_lonlat, ror_centroids)
            fraction = fg.severity_fraction(row["worst_level_cm_nap"], reference_level[nr])
            geometry, affected = fg.synthesize_extent(
                reach_profile[nr], row["type"], anchor_lonlat, fraction, locations,
                synthesis_municipalities)
            props["matched_nr"] = str(nr)
            props["extent_source"] = "derived_municipalities"
            props["severity_fraction"] = fraction
            props["affected_municipalities"] = ";".join(a["naam"] for a in affected)
            n_derived += 1
            fractions.append(fraction)

        features.append({"type": "Feature", "properties": props, "geometry": mapping(geometry)})

    collection = {"type": "FeatureCollection", "features": features}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(collection, f, indent=2)

    print(f"\nWrote {len(features)} feature(s) to {args.out}")
    print(f"extent_source: digitized={n_digitized}, derived_municipalities={n_derived}")
    if fractions:
        fr = np.array(fractions)
        print(f"severity_fraction (derived rows only): min={fr.min():.3f}, "
              f"median={np.median(fr):.3f}, max={fr.max():.3f}")


if __name__ == "__main__":
    main()
