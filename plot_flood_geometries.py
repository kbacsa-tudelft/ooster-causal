"""
Renders build_municipality_flood_geometries.py's output
(flood_events_output/municipality_flood_events.geojson) as an interactive map - one real municipality
polygon per flood_events_municipality.py-detected event, no approximation: each feature's geometry is
already that event's own real municipality (a direct municipality_code lookup), not a guess.

Fill/marker color is keyed by alert (low/medium/high) - there is no "type" (storm surge/river high
water) in this schema, since flood_events_municipality.py's municipality-scoped merge can mix coastal
and river stations within one municipality (see flood_events.merge_national()'s group_col="kind" vs.
group_col="municipality_code" paths).

Events are grouped into one toggleable layer per decade (via folium's native LayerControl, top right
of the map) - all decades are shown by default, un-check any to filter them out interactively without
regenerating the map.

Default --geometry is "extent": the real municipality polygon. Many events genuinely do overlap
(multiple historical events hit the same municipality over the decades) - that's real data, not an
approximation artifact, so low fillOpacity/thin strokes keep overlapping shapes legible by stacking
darker rather than one occluding the rest. --geometry point plots a marker at the municipality's own
centroid instead (sized by alert), for a quick decluttered view.

Usage:
    python3 plot_flood_geometries.py --geojson flood_events_output/municipality_flood_events.geojson \
        --output flood_events_output/flood_map.html
"""
import argparse
import json

import folium
import pandas as pd
from shapely.geometry import mapping, shape

from flood_events import DEFAULT_STATIONS

ALERT_COLOR = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"}
ALERT_RADIUS = {"low": 5, "medium": 9, "high": 14}
TILES_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
TILES_ATTR = "Esri, HERE, Garmin, FAO, NOAA, USGS, © OpenStreetMap contributors, and the GIS User Community"

TOOLTIP_FIELDS = ["peak_date", "alert", "municipality_naam", "worst_station", "worst_detail",
                   "worst_level_cm_nap", "stations"]
# Long fields (stations especially, which lists every contributing station) otherwise render as one
# unbroken line that runs off the visible tooltip - force wrapping and cap the width so it reads as a
# readable block instead.
TOOLTIP_STYLE = "white-space: normal; word-wrap: break-word; max-width: 420px; font-size: 13px;"


def decade_label(peak_date: str) -> str:
    year = int(peak_date[:4])
    return f"{(year // 10) * 10}s"


def style_function(feature: dict) -> dict:
    alert = feature["properties"].get("alert")
    return {
        "fillColor": ALERT_COLOR.get(alert, "#95a5a6"),
        "color": "#2c3e50",
        "weight": 1.5,
        # Low opacity so heavily-overlapping shapes (the same municipality can have several events
        # across the decades) build up darker where they stack, rather than one fully occluding the
        # others underneath it.
        "fillOpacity": 0.2,
    }


def point_marker(feature: dict) -> folium.CircleMarker:
    props = feature["properties"]
    centroid = shape(feature["geometry"]).centroid
    tooltip_rows = "".join(f"<b>{field}</b>: {props[field]}<br>" for field in TOOLTIP_FIELDS
                            if field in props)
    return folium.CircleMarker(
        location=[centroid.y, centroid.x], radius=ALERT_RADIUS.get(props.get("alert"), 5),
        color="#2c3e50", weight=1, fill=True,
        fill_color=ALERT_COLOR.get(props.get("alert"), "#95a5a6"), fill_opacity=0.6,
        tooltip=folium.Tooltip(tooltip_rows, style=TOOLTIP_STYLE),
    )


def legend_html() -> str:
    rows = "".join(
        f'<div><span style="background:{color};width:12px;height:12px;display:inline-block;'
        f'margin-right:6px;"></span>{label}</div>'
        for label, color in ALERT_COLOR.items()
    )
    return (
        '<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;'
        'padding:8px 12px;border:1px solid #2c3e50;border-radius:4px;font-size:13px;">'
        f'<b>alert</b>{rows}</div>'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geojson", default="flood_events_output/municipality_flood_events.geojson")
    parser.add_argument("--locations-csv", default="rws_waterinfo/locations.csv")
    parser.add_argument("--output", default="flood_events_output/flood_map.html")
    parser.add_argument("--geometry", choices=["extent", "point"], default="extent",
                         help="'extent': the real municipality polygon. 'point': a marker at the "
                              "municipality's own centroid, sized by alert - a quick decluttered view.")
    parser.add_argument("--netherlands-border", default="reference_data/netherlands_border.geojson",
                         help="country outline drawn for geographic context - pass '' to skip it "
                              "(see fetch_netherlands_border.py)")
    parser.add_argument("--only-alert", choices=["low", "medium", "high"],
                         help="only plot events at this alert level")
    parser.add_argument("--simplify-deg", type=float, default=0.002,
                         help="shapely simplify() tolerance in degrees (~200m), applied to this map's "
                              "rendering only - plotting all 5531 events unsimplified produces a very "
                              "large/slow HTML file. The underlying municipality_flood_events.geojson "
                              "this reads from is never modified - use --only-alert to cut the map "
                              "down for practical browsing (e.g. --only-alert high is ~100 events).")
    args = parser.parse_args()

    with open(args.geojson) as f:
        collection = json.load(f)
    features = collection["features"]
    if args.only_alert:
        features = [f for f in features if f["properties"]["alert"] == args.only_alert]
    if not features:
        raise ValueError("No features left to plot after applying --only-alert")

    locations = pd.read_csv(args.locations_csv, index_col="name")
    center_lat, center_lon = float(locations["lat"].mean()), float(locations["lon"].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles=TILES_URL, attr=TILES_ATTR)

    if args.netherlands_border:
        with open(args.netherlands_border) as f:
            border = json.load(f)
        folium.GeoJson(
            border,
            style_function=lambda _: {"fillOpacity": 0, "color": "#2c3e50", "weight": 2},
        ).add_to(m)

    decade_groups = {}
    for feature in features:
        decade = decade_label(feature["properties"]["peak_date"])
        group = decade_groups.setdefault(decade, folium.FeatureGroup(name=decade, show=True))

        if args.geometry == "point":
            point_marker(feature).add_to(group)
        else:
            simplified = dict(feature)
            simplified["geometry"] = mapping(shape(feature["geometry"]).simplify(
                args.simplify_deg, preserve_topology=True))
            folium.GeoJson(
                simplified,
                style_function=style_function,
                tooltip=folium.GeoJsonTooltip(
                    fields=[f for f in TOOLTIP_FIELDS if f in simplified["properties"]],
                    style=TOOLTIP_STYLE),
            ).add_to(group)
    for decade in sorted(decade_groups):
        decade_groups[decade].add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(legend_html()))

    # Overlay the 11 original flood_events.py gauge stations for grounding.
    for stations in DEFAULT_STATIONS.values():
        for code, name in stations:
            if code in locations.index:
                row = locations.loc[code]
                folium.CircleMarker(location=[row["lat"], row["lon"]], radius=4, color="#2c3e50",
                                    fill=True, fill_opacity=1.0, tooltip=name).add_to(m)

    m.save(args.output)
    print(f"Wrote {args.output}: {len(features)} feature(s)")


if __name__ == "__main__":
    main()
