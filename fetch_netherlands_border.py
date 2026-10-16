"""
Fetches the Netherlands' national land-area boundary from PDOK/Kadaster's public OGC API (CC BY 4.0,
no account/login required, same collection family as fetch_municipalities.py's previous current-day
source) and writes a small, committed reference GeoJSON file - used by plot_flood_geometries.py to
draw the country outline for geographic context.

Unlike every other output in this project, this file is meant to be committed: it's static long-term
reference geography, not a per-run generated artifact.

Usage:
    python3 fetch_netherlands_border.py --out reference_data/netherlands_border.geojson
"""
import argparse
import json
import os
import urllib.request

from shapely.geometry import mapping, shape

LANDGEBIED_URL = (
    "https://api.pdok.nl/kadaster/bestuurlijkegebieden/ogc/v1/collections/landgebied/items?f=json&limit=10"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=LANDGEBIED_URL)
    parser.add_argument("--out", default="reference_data/netherlands_border.geojson")
    parser.add_argument("--simplify-deg", type=float, default=0.0005,
                         help="shapely simplify() tolerance in degrees (~50m) - the raw boundary has "
                              "~36,000 vertices, far more precise than an outline needs")
    args = parser.parse_args()

    print(f"Fetching Netherlands border from {args.url} ...")
    with urllib.request.urlopen(args.url, timeout=60) as resp:
        data = json.load(resp)
    feature = data["features"][0]
    geom = shape(feature["geometry"]).simplify(args.simplify_deg, preserve_topology=True)
    print(f"Fetched '{feature['properties']['naam']}', {len(geom.geoms)} part(s)")

    collection = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"naam": feature["properties"]["naam"]},
         "geometry": mapping(geom)}
    ]}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(collection, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
