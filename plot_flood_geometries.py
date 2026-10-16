"""
Renders build_flood_geometries.py's output (flood_events_output/flood_events.geojson) as an interactive
map - the "viz first" deliverable for inspecting whether the synthesized flood extents look plausible.

Fill color is keyed by --color-by (alert low/medium/high, or type storm surge/river high water);
stroke style is keyed by extent_source (solid = digitized, a real ROR polygon; dashed =
derived_municipalities, a synthesized union of real municipality polygons) so approximations are
visually distinguishable from the real thing.

Events are grouped into one toggleable layer per decade (via folium's native LayerControl, top right
of the map) - all decades are shown by default, un-check any to filter them out interactively without
regenerating the map.

Default --geometry is "first_municipality": just the single municipality (from --municipalities,
default the 1820 map) containing or nearest each event's own anchor gauge station - a real shape, but
a small, mostly non-overlapping one, since every one of the 267 events is anchored at one of only 11
gauge stations (flood_events.py's events.csv only ever tags a national event with its single
worst_station) and "first affected" is the same question regardless of severity. Events sharing an
anchor do still draw the exact same municipality shape on top of each other - that's inherent to only
having 11 possible anchors, not fixable by styling. --geometry extent plots the full synthesized
municipality-union instead (much more overlap, since it's severity-scaled and can span dozens of
municipalities); --geometry point plots a marker sized by severity instead of any shape at all.

Usage:
    python3 plot_flood_geometries.py --geojson flood_events_output/flood_events.geojson \
        --output flood_events_output/flood_map.html
"""
import argparse
import json

import folium
import pandas as pd
from shapely.geometry import mapping, shape

import flood_geometry as fg
from flood_events import DEFAULT_STATIONS

COLOR_PALETTES = {
    "alert": {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"},
    "type": {"storm surge": "#3498db", "river high water": "#8e44ad"},
}
TILES_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
TILES_ATTR = "Esri, HERE, Garmin, FAO, NOAA, USGS, © OpenStreetMap contributors, and the GIS User Community"

TOOLTIP_FIELDS = ["peak_date", "type", "alert", "worst_station", "extent_source", "matched_nr",
                   "severity_fraction", "affected_municipalities"]


def decade_label(peak_date: str) -> str:
    year = int(peak_date[:4])
    return f"{(year // 10) * 10}s"


def make_style_function(color_by: str):
    palette = COLOR_PALETTES[color_by]

    def style_function(feature: dict) -> dict:
        props = feature["properties"]
        return {
            "fillColor": palette.get(props.get(color_by), "#95a5a6"),
            "color": "#2c3e50",
            "weight": 1,
            "dashArray": None if props.get("extent_source") == "digitized" else "6,4",
            # Low opacity so heavily-overlapping shapes (many events share the same handful of anchor
            # stations - see module docstring) build up darker where they stack, rather than one
            # fully occluding the others underneath it.
            "fillOpacity": 0.12,
        }
    return style_function


MIN_RADIUS, MAX_RADIUS = 5, 20


def anchor_marker(feature: dict, locations, name_to_code: dict, color_by: str) -> folium.CircleMarker:
    props = feature["properties"]
    code = name_to_code[props["worst_station"]]
    lon, lat = fg.station_lonlat(code, locations)
    radius = MIN_RADIUS + (MAX_RADIUS - MIN_RADIUS) * props["severity_fraction"]
    palette = COLOR_PALETTES[color_by]
    tooltip_rows = "".join(f"<b>{field}</b>: {props[field]}<br>" for field in TOOLTIP_FIELDS
                            if field in props)
    return folium.CircleMarker(
        location=[lat, lon], radius=radius,
        color="#2c3e50", weight=1 if props.get("extent_source") == "digitized" else 0.5,
        dash_array=None if props.get("extent_source") == "digitized" else "4,3",
        fill=True, fill_color=palette.get(props.get(color_by), "#95a5a6"), fill_opacity=0.6,
        tooltip=folium.Tooltip(tooltip_rows),
    )


def legend_html(color_by: str) -> str:
    rows = "".join(
        f'<div><span style="background:{color};width:12px;height:12px;display:inline-block;'
        f'margin-right:6px;"></span>{label}</div>'
        for label, color in COLOR_PALETTES[color_by].items()
    )
    return (
        '<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;'
        'padding:8px 12px;border:1px solid #2c3e50;border-radius:4px;font-size:13px;">'
        f'<b>{color_by}</b>{rows}</div>'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geojson", default="flood_events_output/flood_events.geojson")
    parser.add_argument("--locations-csv", default="rws_waterinfo/locations.csv")
    parser.add_argument("--output", default="flood_events_output/flood_map.html")
    parser.add_argument("--color-by", choices=["alert", "type"], default="alert",
                         help="which property to color polygon fills by")
    parser.add_argument("--geometry", choices=["first_municipality", "extent", "point"],
                         default="first_municipality",
                         help="'first_municipality': just the single municipality at each event's "
                              "anchor station - a real shape, minimal overlap. 'extent': the full "
                              "severity-scaled synthesized union (much more overlap). 'point': a "
                              "marker sized by severity, no shape at all.")
    parser.add_argument("--municipalities", default="reference_data/municipalities_1820.geojson",
                         help="municipality set --geometry first_municipality picks from")
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

    style_function = make_style_function(args.color_by)
    name_to_code = fg.station_name_to_code()
    if args.geometry == "first_municipality":
        with open(args.municipalities) as f:
            municipalities = json.load(f)["features"]
        # Repeated anchors (many events share a station) always pick the same municipality - cache it
        # rather than re-scanning the whole municipality list for every one of the 267 events.
        municipality_cache = {}

    decade_groups = {}
    for feature in features:
        decade = decade_label(feature["properties"]["peak_date"])
        group = decade_groups.setdefault(decade, folium.FeatureGroup(name=decade, show=True))

        if args.geometry == "point":
            anchor_marker(feature, locations, name_to_code, args.color_by).add_to(group)
        else:
            if args.geometry == "first_municipality":
                anchor_code = name_to_code[feature["properties"]["worst_station"]]
                if anchor_code not in municipality_cache:
                    anchor_lonlat = fg.station_lonlat(anchor_code, locations)
                    municipality_cache[anchor_code] = fg.nearest_municipality(anchor_lonlat, municipalities)
                municipality = municipality_cache[anchor_code]
                render_feature = {"type": "Feature", "properties": dict(feature["properties"]),
                                   "geometry": municipality["geometry"]}
                render_feature["properties"]["first_municipality"] = municipality["properties"]["naam"]
            else:
                render_feature = feature

            simplified = dict(render_feature)
            simplified["geometry"] = mapping(shape(render_feature["geometry"]).simplify(
                args.simplify_deg, preserve_topology=True))
            folium.GeoJson(
                simplified,
                style_function=style_function,
                tooltip=folium.GeoJsonTooltip(fields=[f for f in TOOLTIP_FIELDS + ["first_municipality"]
                                                       if f in simplified["properties"]]),
            ).add_to(group)
    for decade in sorted(decade_groups):
        decade_groups[decade].add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(legend_html(args.color_by)))

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
