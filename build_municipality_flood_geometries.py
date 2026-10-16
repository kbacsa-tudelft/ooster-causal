"""
Builds a flood-extent GeoJSON directly from flood_events_municipality.py's municipality-level catalog
(flood_events_output/municipality_events.csv) - each row already names its own real municipality
(municipality_code), so this is a plain dict lookup against the national municipality boundaries, not
an approximation: no nearest-neighbor search, no severity-based scaling, no synthesis. Replaces
build_flood_geometries.py (deleted), which had to guess a municipality via a weighted-centroid-of-
contributing-stations approximation because the only detection available then ran at just 11 curated
national gauges.

Usage:
    python3 build_municipality_flood_geometries.py --out flood_events_output/municipality_flood_events.geojson
"""
import argparse
import json

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-csv", default="flood_events_output/municipality_events.csv")
    parser.add_argument("--municipalities", default="reference_data/municipalities_2024_national.geojson")
    parser.add_argument("--out", default="flood_events_output/municipality_flood_events.geojson")
    args = parser.parse_args()

    # municipality_code is PDOK's zero-padded string code (e.g. "0797") - without dtype=str, pandas
    # infers a numeric column and silently strips the leading zero, breaking every lookup below.
    events = pd.read_csv(args.events_csv, dtype={"municipality_code": str})
    with open(args.municipalities) as f:
        municipalities = json.load(f)["features"]
    geometry_by_code = {f["properties"]["code"]: f["geometry"] for f in municipalities}

    features, missing = [], []
    for _, row in events.iterrows():
        geometry = geometry_by_code.get(row["municipality_code"])
        if geometry is None:
            missing.append(row["municipality_code"])
            continue
        features.append({"type": "Feature", "properties": row.to_dict(), "geometry": geometry})

    collection = {"type": "FeatureCollection", "features": features}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(collection, f, indent=2)

    print(f"Wrote {len(features)}/{len(events)} feature(s) -> {args.out}")
    if missing:
        print(f"  {len(missing)} row(s) had a municipality_code not found in {args.municipalities} "
              f"(shouldn't happen - assign_municipalities() already filters by --max-join-km): "
              f"{sorted(set(missing))}")


if __name__ == "__main__":
    main()
