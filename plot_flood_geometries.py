"""
Renders build_flood_geometries.py's output (flood_events_output/flood_events.geojson) as an interactive
map - the "viz first" deliverable for inspecting whether the synthesized flood extents look plausible.

Fill color is keyed by alert (low/medium/high); stroke style is keyed by extent_source (solid =
digitized, a real ROR polygon; dashed = derived_municipalities, a synthesized union of real
municipality polygons) so approximations are visually distinguishable from the real thing.

Usage:
    python3 plot_flood_geometries.py --geojson flood_events_output/flood_events.geojson \
        --output flood_events_output/flood_map.html
"""
import argparse
import json

import folium
import pandas as pd
from shapely.geometry import mapping, shape

from flood_events import DEFAULT_STATIONS

ALERT_COLOR = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"}
TILES_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
TILES_ATTR = "Esri, HERE, Garmin, FAO, NOAA, USGS, © OpenStreetMap contributors, and the GIS User Community"

TOOLTIP_FIELDS = ["peak_date", "type", "alert", "worst_station", "extent_source", "matched_nr",
                   "severity_fraction", "affected_municipalities"]


def style_function(feature: dict) -> dict:
    props = feature["properties"]
    return {
        "fillColor": ALERT_COLOR.get(props.get("alert"), "#95a5a6"),
        "color": "#2c3e50",
        "weight": 2,
        "dashArray": None if props.get("extent_source") == "digitized" else "6,4",
        "fillOpacity": 0.35,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geojson", default="flood_events_output/flood_events.geojson")
    parser.add_argument("--locations-csv", default="rws_waterinfo/locations.csv")
    parser.add_argument("--output", default="flood_events_output/flood_map.html")
    parser.add_argument("--only-alert", choices=["low", "medium", "high"],
                         help="only plot events at this alert level")
    parser.add_argument("--only-source", choices=["digitized", "derived_municipalities"],
                         help="only plot events with this extent_source")
    parser.add_argument("--simplify-deg", type=float, default=0.002,
                         help="shapely simplify() tolerance in degrees (~200m), applied to this map's "
                              "rendering only - a derived_municipalities shape is a union of several "
                              "already-detailed municipality polygons, so plotting all 267 events "
                              "unsimplified produces an unusably large/slow HTML file. The underlying "
                              "flood_events.geojson this reads from is never modified. Plotting all "
                              "267 events at once is still a large (~20MB), slow-loading file even "
                              "simplified - use --only-alert/--only-source to cut it down for "
                              "practical browsing (e.g. --only-alert high is ~200 vertices total).")
    args = parser.parse_args()

    with open(args.geojson) as f:
        collection = json.load(f)
    features = collection["features"]
    if args.only_alert:
        features = [f for f in features if f["properties"]["alert"] == args.only_alert]
    if args.only_source:
        features = [f for f in features if f["properties"]["extent_source"] == args.only_source]
    if not features:
        raise ValueError("No features left to plot after applying --only-alert/--only-source")

    locations = pd.read_csv(args.locations_csv, index_col="name")
    center_lat, center_lon = float(locations["lat"].mean()), float(locations["lon"].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles=TILES_URL, attr=TILES_ATTR)

    for feature in features:
        simplified = dict(feature)
        simplified["geometry"] = mapping(shape(feature["geometry"]).simplify(
            args.simplify_deg, preserve_topology=True))
        folium.GeoJson(
            simplified,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(fields=[f for f in TOOLTIP_FIELDS
                                                   if f in feature["properties"]]),
        ).add_to(m)

    # Overlay the 11 gauge stations for grounding.
    for stations in DEFAULT_STATIONS.values():
        for code, name in stations:
            if code in locations.index:
                row = locations.loc[code]
                folium.CircleMarker(location=[row["lat"], row["lon"]], radius=4, color="#2c3e50",
                                    fill=True, fill_opacity=1.0, tooltip=name).add_to(m)

    m.save(args.output)
    n_digitized = sum(1 for f in features if f["properties"]["extent_source"] == "digitized")
    print(f"Wrote {args.output}: {len(features)} feature(s) "
          f"({n_digitized} digitized, {len(features) - n_digitized} derived)")


if __name__ == "__main__":
    main()
