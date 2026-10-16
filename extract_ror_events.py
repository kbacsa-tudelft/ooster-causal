"""
Distills the 10 digitized historical-flood polygons embedded in ror_historische_overstromingen_2018.html
(a GeoLibre web map built from the official "Overstromingsrisico's in Nederland" 2018 ROR report) into a
small, clean GeoJSON FeatureCollection - so nothing downstream needs to re-parse the large raw HTML.

The HTML embeds its whole map definition as one JSON document inside
<script type="application/json" id="geolibre-project">...</script>; data["layers"] has 10 entries, each
holding a nested one-feature GeoJSON FeatureCollection under layer["geojson"].

Usage:
    python3 extract_ror_events.py --html ror_historische_overstromingen_2018.html \
        --out flood_events_output/ror_historical_events.geojson
"""
import argparse
import json
import os
import re

SCRIPT_RE = re.compile(
    r'<script type="application/json" id="geolibre-project">(.*?)</script>', re.S)

# The only properties that carry real information (others, if any, are map-styling cruft).
KEEP_PROPERTIES = [
    "nr", "jaar", "naam", "bron", "systeem", "stroomgebied", "omvang_rapport",
    "gezondheid", "economie", "opp_polygoon_km2", "kleur", "local_id",
]


def extract(html: str) -> dict:
    m = SCRIPT_RE.search(html)
    if not m:
        raise ValueError('Could not find <script type="application/json" id="geolibre-project"> in HTML')
    data = json.loads(m.group(1))

    features = []
    for layer in data["layers"]:
        feature = layer["geojson"]["features"][0]
        props = {k: feature["properties"].get(k) for k in KEEP_PROPERTIES}
        features.append({"type": "Feature", "properties": props, "geometry": feature["geometry"]})

    features.sort(key=lambda f: f["properties"]["nr"])
    return {"type": "FeatureCollection", "features": features}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", default="ror_historische_overstromingen_2018.html")
    parser.add_argument("--out", default="flood_events_output/ror_historical_events.geojson")
    args = parser.parse_args()

    with open(args.html, encoding="utf-8") as f:
        html = f.read()
    collection = extract(html)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(collection, f, indent=2)

    geom_types = [f["geometry"]["type"] for f in collection["features"]]
    print(f"Wrote {len(collection['features'])} feature(s) to {args.out}")
    print(f"nr values: {sorted(f['properties']['nr'] for f in collection['features'])}")
    print(f"geometry types: Polygon={geom_types.count('Polygon')}, "
          f"MultiPolygon={geom_types.count('MultiPolygon')}")


if __name__ == "__main__":
    main()
