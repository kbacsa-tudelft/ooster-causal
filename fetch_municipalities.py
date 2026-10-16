"""
One-time fetch of Dutch municipality (gemeente) boundaries from PDOK/Kadaster's public OGC API
(CC BY 4.0, no auth required), trimmed to the municipalities near the 10 ROR historical flood events
(extract_ror_events.py's output) and written as a small, committed reference GeoJSON file.

Unlike every other output in this project, this file is meant to be committed: it's static long-term
reference geography, not a per-run generated artifact, and committing it avoids a live network
dependency for every future flood-geometry run. Rerun this script only if PDOK's data updates or the
set of ROR source events changes.

Usage:
    python3 fetch_municipalities.py --ror-events-geojson flood_events_output/ror_historical_events.geojson \
        --out reference_data/municipalities.geojson
"""
import argparse
import json
import os
import urllib.request

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely.geometry import Point, mapping, shape
from shapely.ops import unary_union

from flood_events import DEFAULT_STATIONS

PDOK_ITEMS_URL = (
    "https://api.pdok.nl/kadaster/bestuurlijkegebieden/ogc/v1/collections/gemeentegebied/items"
    "?f=json&limit=1000"
)
KEEP_PROPERTIES = ["code", "identificatie", "naam", "ligt_in_provincie_code", "ligt_in_provincie_naam"]

# Only the 6 "Hoofdwatersysteem" ROR events (nr 1-6) are ever used downstream (flood_geometry.py's
# STORM_SURGE_CANDIDATES/RIVER_CANDIDATES) - the 4 "Regionaal watersysteem" ones (nr 7-10, canal/peat-
# dike/water-main failures) are a different flood mechanism and excluded from matching entirely (see
# the plan's Context). Trimming the search region to just nr 1-6 keeps this file to the municipalities
# actually relevant to the river/coastal corridors used, rather than also pulling in e.g. nr 8's 5967
# km^2 regional-flooding polygon, which would otherwise balloon the kept set to most of the country.
CANDIDATE_NRS = {1, 2, 3, 4, 5, 6}

TO_RD = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
TO_WGS84 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)


def _reproject(geom, transformer: Transformer):
    def _apply(coords):
        xs, ys = transformer.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([xs, ys])
    return shapely.transform(geom, _apply)


def fetch_all_municipalities(start_url: str) -> list:
    """Follows the OGC API's own 'next' pagination link until exhausted (confirmed in practice to
    return all ~342 municipalities in one page at limit=1000, but paginate defensively in case a future
    server-side limit is lower)."""
    features = []
    url = start_url
    for _ in range(50):  # safety cap, well above any plausible page count
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = json.load(resp)
        features.extend(data["features"])
        links = {link["rel"]: link["href"] for link in data.get("links", [])}
        if "next" not in links:
            break
        url = links["next"]
    return features


def build_search_region(ror_events_geojson: str, locations_csv: str, buffer_km: float):
    with open(ror_events_geojson) as f:
        collection = json.load(f)
    geoms = [shape(f["geometry"]) for f in collection["features"]
             if f["properties"]["nr"] in CANDIDATE_NRS]

    # Also buffer around each of the 11 gauge stations themselves - confirmed necessary by direct
    # testing: Delfzijl and Harlingen (both real stations flood_events.py uses) sit well outside even a
    # generous buffer of the 6 ROR polygons alone (1916 Zuiderzeevloed's real extent is the old
    # Zuiderzee/Flevoland shore, not the Wadden coast further northeast/west where these two sit).
    # Without this, any computed event anchored at either station would have zero municipalities to
    # build a corridor from. A lower-severity event realistically stays close to its own station
    # regardless of which distant historical event it's matched to for its reach *profile*, so this is
    # the right fallback minimum coverage, not just a workaround.
    locations = pd.read_csv(locations_csv, index_col="name")
    for stations in DEFAULT_STATIONS.values():
        for code, _ in stations:
            lon, lat = locations.loc[code, "lon"], locations.loc[code, "lat"]
            geoms.append(Point(lon, lat))

    union_wgs84 = unary_union(geoms)
    union_rd = _reproject(union_wgs84, TO_RD)
    buffered_rd = union_rd.buffer(buffer_km * 1000)
    return _reproject(buffered_rd, TO_WGS84)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ror-events-geojson", default="flood_events_output/ror_historical_events.geojson")
    parser.add_argument("--out", default="reference_data/municipalities.geojson")
    parser.add_argument("--buffer-km", type=float, default=20.0,
                         help="how far beyond the 6 candidate ROR events' real extents and the 11 "
                              "gauge stations' own locations to also keep neighboring municipalities")
    parser.add_argument("--locations-csv", default="rws_waterinfo/locations.csv")
    parser.add_argument("--pdok-url", default=PDOK_ITEMS_URL)
    parser.add_argument("--simplify-deg", type=float, default=0.0005,
                         help="shapely simplify() tolerance in degrees (~50m) - PDOK's boundaries are "
                              "survey-grade detailed (hundreds of vertices per municipality), far more "
                              "precise than this project's coarse corridor-intersection use needs; "
                              "simplifying keeps the committed file a reasonable size")
    args = parser.parse_args()

    print(f"Fetching municipality boundaries from {args.pdok_url} ...")
    raw_features = fetch_all_municipalities(args.pdok_url)
    print(f"Fetched {len(raw_features)} municipalities nationwide")

    search_region = build_search_region(args.ror_events_geojson, args.locations_csv, args.buffer_km)

    kept = []
    for feature in raw_features:
        geom = shape(feature["geometry"])
        if geom.intersects(search_region):
            simplified = geom.simplify(args.simplify_deg, preserve_topology=True)
            props = {k: feature["properties"].get(k) for k in KEEP_PROPERTIES}
            kept.append({"type": "Feature", "properties": props,
                         "geometry": mapping(simplified)})

    kept.sort(key=lambda f: f["properties"]["code"])
    collection = {"type": "FeatureCollection", "features": kept}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(collection, f, indent=2)

    print(f"Kept {len(kept)}/{len(raw_features)} municipalities within {args.buffer_km}km of the 6 "
          f"candidate ROR events' real extents or the 11 gauge stations -> {args.out}")


if __name__ == "__main__":
    main()
