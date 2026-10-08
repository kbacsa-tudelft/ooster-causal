"""
Plots the discovered causal graph on a folium map using real station coordinates from
locations.csv (EPSG:28992, Dutch RD). Nodes = channels with known coordinates; arrows = causal
edges (effect <- cause), thicker/darker for stronger edges, colored by source type (blue for
rainfall, green for discharge, red for water level), all at their real coordinates. Discharge
(Q_*) stations are shifted eastward by a small amount - a discharge gauge is very often at
essentially the same point as the water-level gauge of the same code, so without a shift the two
markers would land exactly on top of each other. Rainfall needs no such shift (--rain-shift-m
defaults to 0): a KNMI weather station isn't normally co-located with a water-level gauge.

--mode hover plots only nodes up front; hovering one animates its top --top-n strongest outgoing
edges in instead of drawing every edge at once, which is far more legible once the node count gets
large. --mode hover-incoming is the reverse: hovering a water-level node shows its top --top-n
strongest causes instead - which stations most drive that one.

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


def node_color(name: str, kind: str = 'marker') -> str:
    """Color convention used everywhere on this map: blue for rainfall, green for discharge, red
    (the water-level default, since that's everything else) otherwise. kind='marker' uses a darker
    red than kind='line'/'js', matching this script's existing marker-vs-line distinction."""
    if name.startswith('RH_'):
        return 'steelblue'
    if name.startswith('Q_'):
        return 'seagreen'
    return 'darkred' if kind == 'marker' else 'crimson'


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


def load_locations(locations_csv: str, channel_names: list, rain_shift_m: float,
                   discharge_shift_m: float = 3000.0) -> dict:
    """Maps channel name -> (lat, lon), shifting RH_* stations `rain_shift_m` meters north (0 by
    default - see below) and Q_* stations `discharge_shift_m` meters east before converting to
    lat/lon. locations.csv drops the
    "WL_"/"Q_" prefix that water-level and discharge channels have in the data (both are written by
    download_rws_waterlevel.py, which stores the bare station code); RH_* names match directly,
    since knmi_rain/locations.csv already stores the full prefixed channel name.

    The discharge shift exists because a discharge gauge is very often at (or essentially at) the
    same point as the water-level gauge of the same code - confirmed on real data, e.g. 'almen' gives
    identical coordinates for WL_almen and Q_almen. Without a shift the two markers would sit exactly
    on top of each other, and the default distance is just enough to separate two markers at the
    zoom level this map renders at, not a real location. rain_shift_m exists for the same mechanism
    but defaults to 0, since rain stations don't have this collision problem - pass it explicitly
    if you want rain pulled apart from the network for some other reason.

    Supports two locations.csv schemas, detected by column names:
      - name/x/y (EPSG:28992, e.g. two_week_chunks_*) - projected to lat/lon via pyproj, and the
        shift is a true distance since it's applied in the projected CRS before conversion.
      - name/lat/lon (already lat/lon, e.g. rws_data_adapted) - used directly; the shift is applied
        as an approximate degrees offset (111,320 m/degree), since there's no projected CRS to shift
        a true distance in here. For the eastward (longitude) shift this ignores the cos(latitude)
        compression, same simplification the existing northward shift already makes for latitude.
    """
    locs = pd.read_csv(locations_csv).set_index('name')
    is_projected = 'x' in locs.columns and 'y' in locs.columns
    if is_projected:
        transformer = Transformer.from_crs('EPSG:28992', 'EPSG:4326', always_xy=True)

    coords = {}
    for c in channel_names:
        if c.startswith('WL_') or c.startswith('Q_'):
            stripped = c.split('_', 1)[1]
        else:
            stripped = c
        if stripped not in locs.index:
            continue
        if is_projected:
            x, y = float(locs.loc[stripped, 'x']), float(locs.loc[stripped, 'y'])
            if c.startswith('RH_'):
                y = y + rain_shift_m
            elif c.startswith('Q_'):
                x = x + discharge_shift_m
            lon, lat = transformer.transform(x, y)
        else:
            lat, lon = float(locs.loc[stripped, 'lat']), float(locs.loc[stripped, 'lon'])
            if c.startswith('RH_'):
                lat = lat + rain_shift_m / 111320
            elif c.startswith('Q_'):
                lon = lon + discharge_shift_m / 111320
        coords[c] = (float(lat), float(lon))  # plain floats: folium/branca JSON-serializes these
    return coords


def strongest_edges(graph: np.ndarray, channel_names: list, coords: dict, top_k: int,
                    min_weight: float = 0.0) -> list:
    """Returns [(weight, effect_name, cause_name), ...], strongest first, restricted to edges
    where both endpoints have a known location and the effect is a WL_* (water-level) channel -
    a WL_* or RH_* cause's effect on rainfall isn't the physically interesting relationship here,
    so RH_* is never shown as an effect."""
    edge_strength = graph.copy()
    np.fill_diagonal(edge_strength, 0.0)
    n = len(channel_names)
    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            effect, cause = channel_names[i], channel_names[j]
            if not effect.startswith('WL_'):
                continue
            if effect in coords and cause in coords and edge_strength[i, j] >= min_weight:
                pairs.append((edge_strength[i, j], effect, cause))
    pairs.sort(key=lambda p: p[0], reverse=True)
    return pairs[:top_k]


def strongest_outgoing_per_node(graph: np.ndarray, channel_names: list, coords: dict,
                                min_weight: float = 0.0) -> list:
    """For each node acting as a cause (source), keeps only its single strongest outgoing edge (to
    whichever effect it influences most). Only WL_* (water-level) effects are considered, regardless
    of the source's type - a cause's effect on rainfall isn't the physically interesting
    relationship here. Returns [(weight, effect_name, cause_name), ...], strongest first - at most
    one entry per located source node."""
    edge_strength = graph.copy()
    np.fill_diagonal(edge_strength, 0.0)
    n = len(channel_names)
    pairs = []
    for j in range(n):
        cause = channel_names[j]
        if cause not in coords:
            continue
        best_i, best_w = None, -np.inf
        for i in range(n):
            if i == j:
                continue
            effect = channel_names[i]
            if effect not in coords:
                continue
            if not effect.startswith('WL_'):
                continue  # only consider water-level effects
            if edge_strength[i, j] > best_w:
                best_i, best_w = i, edge_strength[i, j]
        if best_i is not None and best_w >= min_weight:
            pairs.append((best_w, channel_names[best_i], cause))
    pairs.sort(key=lambda p: p[0], reverse=True)
    return pairs


def top_n_outgoing_per_node(graph: np.ndarray, channel_names: list, coords: dict, top_n: int,
                            min_weight: float = 0.0) -> dict:
    """For each node acting as a cause (source), its top_n strongest outgoing edges (to other
    located nodes). Only WL_* (water-level) effects are considered, regardless of the source's type
    (same restriction as strongest_outgoing_per_node, generalized to top_n). Returns
    {cause_name: [(effect_name, weight), ...]}, strongest first."""
    edge_strength = graph.copy()
    np.fill_diagonal(edge_strength, 0.0)
    n = len(channel_names)
    result = {}
    for j in range(n):
        cause = channel_names[j]
        if cause not in coords:
            continue
        candidates = []
        for i in range(n):
            if i == j:
                continue
            effect = channel_names[i]
            if effect not in coords:
                continue
            if not effect.startswith('WL_') or edge_strength[i, j] < min_weight:
                continue
            candidates.append((edge_strength[i, j], effect))
        candidates.sort(reverse=True)
        result[cause] = [(effect, float(w)) for w, effect in candidates[:top_n]]
    return result


def top_n_incoming_per_node(graph: np.ndarray, channel_names: list, coords: dict, top_n: int,
                            min_weight: float = 0.0) -> dict:
    """For each node acting as an effect (only WL_* nodes have effects evaluated at all, matching
    this script's rule that rainfall is never an effect), its top_n strongest incoming edges from
    other located nodes of either type. Returns {effect_name: [(cause_name, weight), ...]},
    strongest first - the reverse of top_n_outgoing_per_node."""
    edge_strength = graph.copy()
    np.fill_diagonal(edge_strength, 0.0)
    n = len(channel_names)
    result = {}
    for i in range(n):
        effect = channel_names[i]
        if effect not in coords or not effect.startswith('WL_'):
            continue
        candidates = []
        for j in range(n):
            if i == j:
                continue
            cause = channel_names[j]
            if cause not in coords or edge_strength[i, j] < min_weight:
                continue
            candidates.append((edge_strength[i, j], cause))
        candidates.sort(reverse=True)
        result[effect] = [(cause, float(w)) for w, cause in candidates[:top_n]]
    return result


def build_hover_map(m: folium.Map, graph: np.ndarray, channel_names: list, coords: dict, top_n: int,
                    min_weight: float = 0.0, direction: str = 'outgoing'):
    """Plots every located node; hovering one draws its top_n strongest edges as animated dashed
    lines, removed on mouseout. direction='outgoing' (default) shows what that node causes, colored
    by the hovered node's type; direction='incoming' shows what causes that node, colored per edge
    by the cause's type (the hovered node is always WL_*, but its causes can be either type). All
    interactivity runs client-side (embedded JS), since the graph the user hovers is fixed at render
    time - no server round-trip needed."""
    if direction == 'incoming':
        top_edges = top_n_incoming_per_node(graph, channel_names, coords, top_n, min_weight)
    else:
        top_edges = top_n_outgoing_per_node(graph, channel_names, coords, top_n, min_weight)

    marker_vars = []
    for name, (lat, lon) in coords.items():
        color = node_color(name)
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
    direction_json = json.dumps(direction)

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
    var direction = {direction_json};  // 'outgoing': node causes the edges; 'incoming': node is caused
    var activeLines = [];

    function clearLines() {{
        activeLines.forEach(function(l) {{ map.removeLayer(l); }});
        activeLines = [];
    }}

    function showLinks(node) {{
        clearLines();
        var edges = topEdges[node] || [];
        if (!edges.length) return;
        var weights = edges.map(function(e) {{ return e[1]; }});
        var maxW = Math.max.apply(null, weights), minW = Math.min.apply(null, weights);
        var span = (maxW - minW) || 1;
        edges.forEach(function(e) {{
            var other = e[0], w = e[1];
            if (!nodeCoords[other]) return;
            var strength = (w - minW) / span;
            // outgoing: node is the cause, line runs node -> other. incoming: node is the effect,
            // line runs other -> node. Either way the line is colored by whichever endpoint is the
            // cause, since that's what blue-vs-red means on this map.
            var from = direction === 'outgoing' ? node : other;
            var to = direction === 'outgoing' ? other : node;
            var causeNode = direction === 'outgoing' ? node : other;
            var color = causeNode.indexOf('RH_') === 0 ? 'steelblue'
                      : causeNode.indexOf('Q_') === 0 ? 'seagreen' : 'crimson';
            var line = L.polyline([nodeCoords[from], nodeCoords[to]], {{
                color: color, weight: 2 + 4 * strength, opacity: 0.55 + 0.45 * strength,
            }}).addTo(map);
            line.bindTooltip(from + ' -> ' + to + ': ' + w.toFixed(4));
            var el = line.getElement();
            if (el) {{ el.classList.add('hover-edge'); }}
            activeLines.push(line);
        }});
    }}
"""]
    for marker_var, name in marker_vars:
        script_lines.append(
            f"    {marker_var}.on('mouseover', function() {{ showLinks({json.dumps(name)}); }});\n"
            f"    {marker_var}.on('mouseout', clearLines);\n"
        )
    script_lines.append('});\n</script>\n')

    m.get_root().html.add_child(folium.Element(''.join(script_lines)))
    print(f'Wrote hover map: {len(coords)} nodes, up to {top_n} {direction} edges shown per hover')


def build_map(graph_path: str, data_dir: str, locations_csv: str, output: str, top_k: int,
              rain_shift_m: float, mode: str = 'top-k', top_n: int = 5, min_weight: float = 0.0,
              discharge_shift_m: float = 3000.0):
    graph = np.load(graph_path)
    channel_names = load_channel_names(data_dir, graph_path)
    if graph.shape != (len(channel_names), len(channel_names)):
        raise ValueError(f'graph shape {graph.shape} does not match {len(channel_names)} channels '
                          f'found in {data_dir} - is --data-dir the dataset this graph was trained on?')

    coords = load_locations(locations_csv, channel_names, rain_shift_m, discharge_shift_m)
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

    if mode in ('hover', 'hover-incoming'):
        direction = 'incoming' if mode == 'hover-incoming' else 'outgoing'
        build_hover_map(m, graph, channel_names, coords, top_n, min_weight, direction)
        m.save(output)
        print(f'Wrote {output}')
        return

    if mode == 'per-node-outgoing':
        edges = strongest_outgoing_per_node(graph, channel_names, coords, min_weight)
    else:
        edges = strongest_edges(graph, channel_names, coords, top_k, min_weight)
    if not edges:
        # a threshold above every edge weight leaves nothing to draw: keep the map with its nodes
        print(f'WARNING: no edges with weight >= {min_weight}; the map shows nodes only')
        for name, (lat, lon) in coords.items():
            folium.CircleMarker(location=[lat, lon], radius=4, color=node_color(name), tooltip=name).add_to(m)
        m.save(output)
        print(f'Wrote {output}: {len(coords)} nodes, 0 edges plotted')
        return

    for name, (lat, lon) in coords.items():
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color=node_color(name),
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
        color = node_color(cause, kind='line')
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
    parser.add_argument('--mode', choices=['top-k', 'per-node-outgoing', 'hover', 'hover-incoming'],
                         default='top-k',
                         help='top-k: globally strongest edges. per-node-outgoing: for each source '
                              'node, only its single strongest outgoing edge. hover: plot all nodes '
                              'and show a node\'s top --top-n outgoing edges (animated) on hover. '
                              'hover-incoming: same, but shows a node\'s top --top-n causes instead.')
    parser.add_argument('--rain-shift-m', type=float, default=0,
                         help='meters to shift RH_* (rainfall) stations north before plotting; '
                              '0 (default) plots them at their real coordinates')
    parser.add_argument('--discharge-shift-m', type=float, default=3000,
                         help='meters to shift Q_* (discharge) stations east before plotting - '
                              'keeps them from landing exactly on a same-code water-level station')
    parser.add_argument('--top-n', type=int, default=5,
                         help='number of edges to show per node on hover (--mode hover/hover-incoming only)')
    parser.add_argument('--min-weight', type=float, default=0.0,
                         help='drop edges weaker than this (all modes); 0 keeps every edge')
    args = parser.parse_args()
    build_map(args.graph, args.data_dir, args.locations_csv, args.output, args.top_k,
              args.rain_shift_m, args.mode, args.top_n, args.min_weight, args.discharge_shift_m)


if __name__ == '__main__':
    main()
