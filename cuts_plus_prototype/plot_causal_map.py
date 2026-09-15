"""
Plots the discovered causal graph on a folium map using real station coordinates from
locations.csv (EPSG:28992, Dutch RD). Nodes = channels with known coordinates; arrows = causal
edges (effect <- cause), thicker/darker for stronger edges, colored by source type (blue for
rainfall, red for water level). Rainfall (RH_*) stations are shifted northward so they visually sit
"above" the water-level network, since their real location isn't otherwise distinguishable on the
map from a water-level sensor.

--mode hover plots only nodes up front; hovering one animates its top --top-n strongest outgoing
edges in instead of drawing every edge at once, which is far more legible once the node count gets
large.

Channels missing from locations.csv (and any edge touching one) are skipped and reported - not
silently dropped.

Usage:
    python3 cuts_plus_prototype/plot_causal_map.py \
        --graph cuts_plus_prototype/scratch/full_run_v3/models/cuts_plus_graph.npy \
        --data-dir two_week_chunks_full \
        --output causal_map.html
"""
import argparse
import glob
import json
import os

import folium
import numpy as np
import pandas as pd
from folium.plugins import PolyLineTextPath
from pyproj import Transformer


def load_channel_names(data_dir: str, graph_path: str) -> list:
    """Prefers the channel_names.json cuts_plus_rca.py saves next to the graph - the training
    pipeline can drop zero-observation channels, so the graph's row/column order doesn't always
    match the dataset's raw column list. Falls back to the dataset's columns for older runs that
    predate this file."""
    names_path = os.path.join(os.path.dirname(graph_path), 'channel_names.json')
    if os.path.exists(names_path):
        with open(names_path) as fh:
            return json.load(fh)
    f = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))[0]
    return list(pd.read_parquet(f).columns)


def load_locations(locations_csv: str, channel_names: list, rain_shift_m: float) -> dict:
    """Maps channel name -> (lat, lon), converting from EPSG:28992 and shifting RH_* stations
    `rain_shift_m` meters north (in the original projected CRS, so the shift is a true distance)
    before converting to lat/lon. locations.csv drops the "WL_" prefix that water-level channels
    have in the data; RH_* names match directly."""
    locs = pd.read_csv(locations_csv).set_index('name')
    transformer = Transformer.from_crs('EPSG:28992', 'EPSG:4326', always_xy=True)

    coords = {}
    for c in channel_names:
        stripped = c[len('WL_'):] if c.startswith('WL_') else c
        if stripped not in locs.index:
            continue
        x, y = float(locs.loc[stripped, 'x']), float(locs.loc[stripped, 'y'])
        if c.startswith('RH_'):
            y = y + rain_shift_m
        lon, lat = transformer.transform(x, y)
        coords[c] = (float(lat), float(lon))  # plain floats: folium/branca JSON-serializes these
    return coords


def strongest_edges(graph: np.ndarray, channel_names: list, coords: dict, top_k: int) -> list:
    """Returns [(weight, effect_name, cause_name), ...], strongest first, restricted to edges
    where both endpoints have a known location."""
    edge_strength = graph.copy()
    np.fill_diagonal(edge_strength, 0.0)
    n = len(channel_names)
    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            effect, cause = channel_names[i], channel_names[j]
            if effect in coords and cause in coords:
                pairs.append((edge_strength[i, j], effect, cause))
    pairs.sort(key=lambda p: p[0], reverse=True)
    return pairs[:top_k]


def strongest_outgoing_per_node(graph: np.ndarray, channel_names: list, coords: dict) -> list:
    """For each node acting as a cause (source), keeps only its single strongest outgoing edge (to
    whichever effect it influences most). For an RH_* (rainfall) source, only WL_* (water-level)
    effects are considered - a rain gauge's strongest link to another rain gauge isn't the
    physically interesting relationship here. Returns [(weight, effect_name, cause_name), ...],
    strongest first - at most one entry per located source node."""
    edge_strength = graph.copy()
    np.fill_diagonal(edge_strength, 0.0)
    n = len(channel_names)
    pairs = []
    for j in range(n):
        cause = channel_names[j]
        if cause not in coords:
            continue
        is_rain = cause.startswith('RH_')
        best_i, best_w = None, -np.inf
        for i in range(n):
            if i == j:
                continue
            effect = channel_names[i]
            if effect not in coords:
                continue
            if is_rain and effect.startswith('RH_'):
                continue  # rain source: only consider water-level effects
            if edge_strength[i, j] > best_w:
                best_i, best_w = i, edge_strength[i, j]
        if best_i is not None:
            pairs.append((best_w, channel_names[best_i], cause))
    pairs.sort(key=lambda p: p[0], reverse=True)
    return pairs


def top_n_outgoing_per_node(graph: np.ndarray, channel_names: list, coords: dict, top_n: int) -> dict:
    """For each node acting as a cause (source), its top_n strongest outgoing edges (to other
    located nodes). For an RH_* (rainfall) source, only WL_* (water-level) effects are considered -
    a rain gauge's strongest link to another rain gauge isn't the physically interesting
    relationship here (same restriction as strongest_outgoing_per_node, generalized to top_n).
    Returns {cause_name: [(effect_name, weight), ...]}, strongest first."""
    edge_strength = graph.copy()
    np.fill_diagonal(edge_strength, 0.0)
    n = len(channel_names)
    result = {}
    for j in range(n):
        cause = channel_names[j]
        if cause not in coords:
            continue
        is_rain = cause.startswith('RH_')
        candidates = []
        for i in range(n):
            if i == j:
                continue
            effect = channel_names[i]
            if effect not in coords:
                continue
            if is_rain and effect.startswith('RH_'):
                continue
            candidates.append((edge_strength[i, j], effect))
        candidates.sort(reverse=True)
        result[cause] = [(effect, float(w)) for w, effect in candidates[:top_n]]
    return result


def build_hover_map(m: folium.Map, graph: np.ndarray, channel_names: list, coords: dict, top_n: int):
    """Plots every located node; hovering one draws its top_n strongest outgoing edges as animated
    dashed lines (color matches the source: blue for rainfall, red for water level), removed on
    mouseout. All interactivity runs client-side (embedded JS), since the graph the user hovers is
    fixed at render time - no server round-trip needed."""
    top_edges = top_n_outgoing_per_node(graph, channel_names, coords, top_n)

    marker_vars = []
    for name, (lat, lon) in coords.items():
        is_rain = name.startswith('RH_')
        color = 'steelblue' if is_rain else 'darkred'
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color=color,
            fill=True,
            fill_opacity=0.9,
        ).add_to(m)
        # Invisible padding ring, larger than the visible dot - the actual hover target, so the
        # cursor landing near the small dot's edge doesn't flicker in/out of the hit area. fill_opacity
        # just above 0 (rather than exactly 0) keeps it reliably hit-testable across browsers.
        hit_marker = folium.CircleMarker(
            location=[lat, lon],
            radius=16,
            color=color,
            weight=0,
            opacity=0,
            fill=True,
            fill_opacity=0.02,
            popup=name,
            tooltip=name,
        )
        hit_marker.add_to(m)
        marker_vars.append((hit_marker.get_name(), name))

    node_coords_json = json.dumps({n: [lat, lon] for n, (lat, lon) in coords.items()})
    top_edges_json = json.dumps(top_edges)

    script_lines = [f"""
<style>
.hover-edge {{ stroke-dasharray: 8, 8; animation: causal_dash 0.6s linear infinite; }}
@keyframes causal_dash {{ to {{ stroke-dashoffset: -16; }} }}
</style>
<script>
window.addEventListener('load', function() {{
    // Deferred to 'load' because folium/branca emits its own map/marker init scripts into the
    // document's script section at render time, in an order this element's position can't control -
    // referencing {m.get_name()} synchronously here can run before that code has executed. 'load'
    // fires only after everything else on the page has already run, so this is always safe.
    var map = {m.get_name()};
    var nodeCoords = {node_coords_json};
    var topEdges = {top_edges_json};
    var activeLines = [];

    function clearLines() {{
        activeLines.forEach(function(l) {{ map.removeLayer(l); }});
        activeLines = [];
    }}

    function showLinks(source, color) {{
        clearLines();
        var edges = topEdges[source] || [];
        if (!edges.length) return;
        var weights = edges.map(function(e) {{ return e[1]; }});
        var maxW = Math.max.apply(null, weights), minW = Math.min.apply(null, weights);
        var span = (maxW - minW) || 1;
        edges.forEach(function(e) {{
            var target = e[0], w = e[1];
            if (!nodeCoords[target]) return;
            var strength = (w - minW) / span;
            var line = L.polyline([nodeCoords[source], nodeCoords[target]], {{
                color: color, weight: 2 + 4 * strength, opacity: 0.55 + 0.45 * strength,
            }}).addTo(map);
            line.bindTooltip(source + ' -> ' + target + ': ' + w.toFixed(4));
            var el = line.getElement();
            if (el) {{ el.classList.add('hover-edge'); }}
            activeLines.push(line);
        }});
    }}
"""]
    for marker_var, name in marker_vars:
        color = 'steelblue' if name.startswith('RH_') else 'crimson'
        script_lines.append(
            f"    {marker_var}.on('mouseover', function() {{ "
            f"showLinks({json.dumps(name)}, {json.dumps(color)}); }});\n"
            f"    {marker_var}.on('mouseout', clearLines);\n"
        )
    script_lines.append('});\n</script>\n')

    m.get_root().html.add_child(folium.Element(''.join(script_lines)))
    print(f'Wrote hover map: {len(coords)} nodes, up to {top_n} outgoing edges shown per hover')


def build_map(graph_path: str, data_dir: str, locations_csv: str, output: str, top_k: int,
              rain_shift_m: float, mode: str = 'top-k', top_n: int = 5):
    graph = np.load(graph_path)
    channel_names = load_channel_names(data_dir, graph_path)
    if graph.shape != (len(channel_names), len(channel_names)):
        raise ValueError(f'graph shape {graph.shape} does not match {len(channel_names)} channels '
                          f'found in {data_dir} - is --data-dir the dataset this graph was trained on?')

    coords = load_locations(locations_csv, channel_names, rain_shift_m)
    missing = [c for c in channel_names if c not in coords]
    if missing:
        print(f'{len(missing)} channel(s) skipped (no location in {locations_csv}): {missing}')

    center_lat = float(np.mean([lat for lat, lon in coords.values()]))
    center_lon = float(np.mean([lon for lat, lon in coords.values()]))
    # folium's built-in 'OpenStreetMap' tiles reject this kind of local/embedded use under OSM's
    # tile usage policy (403), and 'CartoDB positron' now requires an API key too - Esri's public
    # basemap service remains free and keyless for this.
    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=10,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Esri, HERE, Garmin, FAO, NOAA, USGS, © OpenStreetMap contributors, and the GIS User Community',
    )

    if mode == 'hover':
        build_hover_map(m, graph, channel_names, coords, top_n)
        m.save(output)
        print(f'Wrote {output}')
        return

    if mode == 'per-node-outgoing':
        edges = strongest_outgoing_per_node(graph, channel_names, coords)
    else:
        edges = strongest_edges(graph, channel_names, coords, top_k)
    if not edges:
        raise ValueError('No edges to plot - no two located channels have a scored edge.')

    for name, (lat, lon) in coords.items():
        is_rain = name.startswith('RH_')
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color='steelblue' if is_rain else 'darkred',
            fill=True,
            fill_opacity=0.9,
            popup=name,
            tooltip=name,
        ).add_to(m)

    max_w = max(w for w, _, _ in edges)
    min_w = min(w for w, _, _ in edges)
    span = max_w - min_w or 1.0
    for w, effect, cause in edges:
        strength = float((w - min_w) / span)  # 0..1, relative to the plotted edges only - plain
        # float: folium/branca JSON-serializes these and chokes on numpy scalar types (e.g. float32).
        color = 'steelblue' if cause.startswith('RH_') else 'crimson'
        line = folium.PolyLine(
            locations=[coords[cause], coords[effect]],
            color=color,
            weight=1 + 4 * strength,
            opacity=0.4 + 0.5 * strength,
            tooltip=f'{cause} -> {effect}: {w:.4f}',
        )
        line.add_to(m)
        PolyLineTextPath(line, '   ➤   ', repeat=True, offset=6,
                          attributes={'fill': color, 'font-size': '14'}).add_to(m)

    m.save(output)
    print(f'Wrote {output}: {len(coords)} nodes, {len(edges)} edges plotted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', required=True, help='path to a cuts_plus_graph.npy')
    parser.add_argument('--data-dir', required=True,
                         help='dataset directory the graph was trained on (to recover channel order)')
    parser.add_argument('--locations-csv', default='two_week_chunks/locations.csv')
    parser.add_argument('--output', default='causal_map.html')
    parser.add_argument('--top-k', type=int, default=15,
                         help='number of strongest edges to draw (--mode top-k only)')
    parser.add_argument('--mode', choices=['top-k', 'per-node-outgoing', 'hover'], default='top-k',
                         help='top-k: globally strongest edges. per-node-outgoing: for each source '
                              'node, only its single strongest outgoing edge. hover: plot all nodes '
                              'and show a node\'s top --top-n outgoing edges (animated) on hover.')
    parser.add_argument('--rain-shift-m', type=float, default=10000,
                         help='meters to shift RH_* (rainfall) stations north before plotting')
    parser.add_argument('--top-n', type=int, default=5,
                         help='number of outgoing edges to show per node on hover (--mode hover only)')
    args = parser.parse_args()
    build_map(args.graph, args.data_dir, args.locations_csv, args.output, args.top_k,
              args.rain_shift_m, args.mode, args.top_n)


if __name__ == '__main__':
    main()
