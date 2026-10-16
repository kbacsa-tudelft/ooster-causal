"""
Fetches Dutch municipality boundaries for a given year from NLGis (Zijdeman & Mac Gillavry,
DataverseNL, DOI 10.34894/4G2NJM, CC-BY-SA-4.0) - yearly boundary files covering 1812-2020, each a
public GeoJSON download with no account/login required (confirmed directly). Trims to the
municipalities near the 6 candidate ROR historical flood events and the 11 flood_events.py gauge
stations, and writes a small, committed reference GeoJSON file.

Historical boundaries matter here because the Netherlands has steadily merged municipalities for
efficiency - 342 today vs. 1,249 in 1820 (confirmed the national maximum across the dataset, by
sampling feature counts year by year) and 1,014 in 1953 alone - so fetching whichever year is actually
relevant gives far finer granularity for the flood-geometry synthesis than today's boundaries would:
  - Each of the 6 candidate ROR events' own reach profile (flood_geometry.learn_reach_profile) uses
    that event's own year (1916/1926/1953/1993/1995), for historically accurate measurement.
  - Every synthesized (non-template) event uses year 1820, the single most granular map available,
    uniformly - per the user's explicit choice, visualization granularity matters more here than using
    each event's own (less detailed) contemporary boundaries.

Unlike every other output in this project, these files are meant to be committed: they're static
long-term reference geography, not a per-run generated artifact, and committing them avoids a live
network dependency for every future flood-geometry run. Rerun this script only if NLGis's data updates
or the set of ROR source events changes.

--source pdok fetches today's current-day boundaries from PDOK/Kadaster instead (CC BY 4.0, no auth) -
NLGis's own coverage stops at 2020. Combined with --national (skip the ROR/station trim entirely and
keep every municipality nationwide), this is what flood_events_municipality.py needs to join stations
scattered anywhere in the country, not just near the 6 historical flood events used for the map.

Usage:
    python3 fetch_municipalities.py --year 1820 --out reference_data/municipalities_1820.geojson
    python3 fetch_municipalities.py --year 1953 --out reference_data/municipalities_1953.geojson
    python3 fetch_municipalities.py --source pdok --national
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

NLGIS_DATASET_API = "https://dataverse.nl/api/datasets/:persistentId/?persistentId=doi:10.34894/4G2NJM"
NLGIS_FILE_ACCESS = "https://dataverse.nl/api/access/datafile/{file_id}"
# NLGis properties are naam/acode/cbscode/year - renamed to "code"/"naam" on output so every downstream
# consumer (flood_geometry.py) can read either a NLGis-year file or a PDOK file the same way, without
# caring which source or year it came from.
NLGIS_KEEP_PROPERTIES = {"cbscode": "code", "naam": "naam"}

# PDOK/Kadaster's current-day municipality API (CC BY 4.0, no auth) - used for --source pdok instead
# of a historical NLGis year, since NLGis's own coverage stops at 2020 and station-municipality
# assignment for flood_events_municipality.py wants today's boundaries, not a historical snapshot.
PDOK_ITEMS_URL = (
    "https://api.pdok.nl/kadaster/bestuurlijkegebieden/ogc/v1/collections/gemeentegebied/items"
    "?f=json&limit=1000"
)
PDOK_KEEP_PROPERTIES = {"code": "code", "naam": "naam"}

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


def resolve_file_id(year: int) -> int:
    """NLGis file ids aren't a documented/guaranteed sequence - look up the real id for nl_<year>.geojson
    from the dataset's own file list rather than assuming any numbering pattern."""
    with urllib.request.urlopen(NLGIS_DATASET_API, timeout=30) as resp:
        data = json.load(resp)
    target = f"nl_{year}.geojson"
    for f in data["data"]["latestVersion"]["files"]:
        if f["dataFile"]["filename"] == target:
            return f["dataFile"]["id"]
    raise ValueError(f"{target} not found in the NLGis dataset file list (valid range: 1812-2020)")


def fetch_year(year: int) -> list:
    file_id = resolve_file_id(year)
    url = NLGIS_FILE_ACCESS.format(file_id=file_id)
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = json.load(resp)
    return data["features"]


def fetch_pdok(start_url: str = PDOK_ITEMS_URL) -> list:
    """Follows the OGC API's own 'next' pagination link until exhausted (confirmed in practice to
    return all ~342 municipalities in one page at limit=1000, but paginate defensively in case a
    future server-side limit is lower)."""
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
    parser.add_argument("--source", choices=["nlgis", "pdok"], default="nlgis",
                         help="'nlgis': a historical year (1812-2020, see --year). 'pdok': today's "
                              "current-day municipality boundaries (no --year).")
    parser.add_argument("--year", type=int, default=None,
                         help="which NLGis yearly boundary file to fetch (1812-2020) - required for "
                              "--source nlgis, ignored for --source pdok")
    parser.add_argument("--national", action="store_true",
                         help="keep every municipality nationwide instead of trimming to the "
                              "6 candidate ROR events + 11 gauge stations (see build_search_region) - "
                              "needed when stations can be anywhere in the country, e.g. for "
                              "flood_events_municipality.py's station-to-municipality join")
    parser.add_argument("--ror-events-geojson", default="flood_events_output/ror_historical_events.geojson")
    parser.add_argument("--out", default=None,
                         help="default: reference_data/municipalities_<year>.geojson, or "
                              "reference_data/municipalities_2024_national.geojson for "
                              "--source pdok --national")
    parser.add_argument("--buffer-km", type=float, default=20.0,
                         help="how far beyond the 6 candidate ROR events' real extents and the 11 "
                              "gauge stations' own locations to also keep neighboring municipalities "
                              "(ignored with --national)")
    parser.add_argument("--locations-csv", default="rws_waterinfo/locations.csv")
    parser.add_argument("--simplify-deg", type=float, default=0.0005,
                         help="shapely simplify() tolerance in degrees (~50m), to keep the committed "
                              "file a reasonable size")
    args = parser.parse_args()
    if args.source == "nlgis" and args.year is None:
        parser.error("--year is required for --source nlgis")

    if args.source == "nlgis":
        out = args.out or f"reference_data/municipalities_{args.year}.geojson"
        print(f"Fetching NLGis {args.year} municipality boundaries ...")
        raw_features = fetch_year(args.year)
        keep_properties = NLGIS_KEEP_PROPERTIES
        print(f"Fetched {len(raw_features)} municipalities nationwide for {args.year}")
    else:
        out = args.out or "reference_data/municipalities_2024_national.geojson"
        print("Fetching PDOK current-day municipality boundaries ...")
        raw_features = fetch_pdok()
        keep_properties = PDOK_KEEP_PROPERTIES
        print(f"Fetched {len(raw_features)} municipalities nationwide")

    search_region = None if args.national else build_search_region(
        args.ror_events_geojson, args.locations_csv, args.buffer_km)

    kept = []
    for feature in raw_features:
        geom = shape(feature["geometry"])
        if search_region is None or geom.intersects(search_region):
            simplified = geom.simplify(args.simplify_deg, preserve_topology=True)
            props = {out_key: feature["properties"].get(in_key)
                     for in_key, out_key in keep_properties.items()}
            kept.append({"type": "Feature", "properties": props,
                         "geometry": mapping(simplified)})

    kept.sort(key=lambda f: f["properties"]["code"])
    collection = {"type": "FeatureCollection", "features": kept}

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(collection, f, indent=2)

    if args.national:
        print(f"Kept all {len(kept)}/{len(raw_features)} municipalities nationwide -> {out}")
    else:
        print(f"Kept {len(kept)}/{len(raw_features)} municipalities within {args.buffer_km}km of the "
              f"6 candidate ROR events' real extents or the 11 gauge stations -> {out}")


if __name__ == "__main__":
    main()
