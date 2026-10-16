"""
Synthesizes a flood-extent polygon for every flood_events.py-computed national event (events.csv),
by borrowing a *reach profile* - how far one of the 10 ROR historical events actually reaches upstream/
downstream along its river (or along the coast) and inland from its anchor gauge station, measured
against real Dutch municipality boundaries for the relevant year (reference_data/municipalities_<year>.
geojson, see fetch_municipalities.py) - and scaling that profile by the event's severity relative to
its matched ROR counterpart. The output is a *fresh* shape: the union of whichever real municipality polygons fall
inside the severity-scaled search region, not a copy of the ROR polygon itself.

Only the 6 "Hoofdwatersysteem" ROR events are valid templates (see build_flood_geometries.py's
docstring for why the 4 "Regionaal watersysteem" ones are excluded). Of those 6, nr 3/4/5/6 have a
directly-matching row already in events.csv (the same real historical flood, already detected from the
time series) - that row's own worst_level_cm_nap is used as nr's severity "100%" reference, and that
row is attached the real ROR polygon verbatim rather than a synthesized one. nr 1/2 predate events.csv's
1950-04-24 coverage, so they fall back to the nearest candidate station's own worst-ever recorded level
in events.csv as the reference instead.
"""
import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import LineString, Point, shape
from shapely.ops import substring, unary_union

from flood_events import DEFAULT_STATIONS

RIVER_BRANCHES = {
    # Each branch is ordered upstream -> downstream. Only the first/last entry is one of
    # flood_events.py's own DEFAULT_STATIONS (the actual flood-detection gauge); the rest are extra
    # waypoints from this project's own rws_waterinfo/locations.csv, added because the Maas/Waal both
    # curve substantially between their 2 gauge endpoints - a straight line between just the 2 gauges
    # badly approximates the real river path, inflating the perpendicular/"inland" distance measured
    # in learn_reach_profile() for any real polygon that (correctly) follows the actual winding river.
    "maas": ["maastricht.borgharen.maas.beneden", "elsloo.maas", "maaseik", "maasbracht", "venlo",
             "shertogenbosch.empel.maas", "andel.maas"],
    "waal": ["lobith.bovenrijn.tolkamer", "millingenaanderijn.pannerdensekop", "nijmegen.waal",
             "sintandries.waal", "tiel.sluis.waal", "loevestein.waal.km951.ro"],
    "ijssel": ["deventer"],  # single-station branch
}
# Approximate NE -> SW geographic sequence along the coast/estuaries.
COASTLINE_ORDER = [
    "delfzijl", "harlingen.waddenzee", "denhelder.marsdiep",
    "hoekvanholland", "dordrecht.oudemaas.benedenmerwede", "vlissingen",
]

# nr: 1916 Zuiderzeevloed, 1953 Watersnoodramp.
STORM_SURGE_CANDIDATES = [1, 3]
# nr: 1926 Rivieroverstroming Rijn en Maas, 1993 Hoogwater Maas, 1995 Hoogwater Maas, 1995 Hoogwater Rijn.
RIVER_CANDIDATES = [2, 4, 5, 6]
DIRECT_MATCH_YEAR_TYPE = {
    3: (1953, "storm surge"),
    4: (1993, "river high water"),
    5: (1995, "river high water"),
    6: (1995, "river high water"),
}

MIN_SEARCH_RADIUS_M = 500.0  # degenerate (zero-reach) events still get a small buffered-point region

TO_RD = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
TO_WGS84 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)


def _reproject(geom, transformer: Transformer):
    def _apply(coords):
        xs, ys = transformer.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([xs, ys])
    return shapely.transform(geom, _apply)


def station_name_to_code() -> dict:
    """events.csv's worst_station column holds display names ('Vlissingen'); DEFAULT_STATIONS pairs
    those with the dataset's own codes ('vlissingen')."""
    return {name: code for stations in DEFAULT_STATIONS.values() for code, name in stations}


def station_lonlat(code: str, locations) -> tuple:
    row = locations.loc[code]
    return float(row["lon"]), float(row["lat"])


def _branch_axis_rd(branch_codes: list, locations) -> LineString:
    points_rd = [_reproject(Point(station_lonlat(c, locations)), TO_RD) for c in branch_codes]
    if len(points_rd) == 1:
        # A single-station branch (e.g. "ijssel": just Deventer) has no real direction to measure
        # upstream/downstream along - shapely also requires >=2 points for a LineString. Make a
        # negligibly-short 2-point line at the station itself, so axis_rd.project()/.length still work
        # (they'll just always resolve to ~this one point - upstream/downstream naturally collapse to
        # 0, leaving only the perpendicular/inland distance meaningful, which is correct here).
        p = points_rd[0]
        return LineString([p, Point(p.x + 1.0, p.y)])
    return LineString(points_rd)


def _coastline_axis_rd(locations) -> LineString:
    return _branch_axis_rd(COASTLINE_ORDER, locations)


def nearest_station(centroid_lonlat: tuple, station_codes: list, locations) -> str:
    centroid_rd = _reproject(Point(centroid_lonlat), TO_RD)
    return min(station_codes, key=lambda c: centroid_rd.distance(
        _reproject(Point(station_lonlat(c, locations)), TO_RD)))


def _nearest_branch(point_rd: Point, locations) -> str:
    return min(RIVER_BRANCHES, key=lambda branch: point_rd.distance(_branch_axis_rd(
        RIVER_BRANCHES[branch], locations)))


def axis_for(event_type: str, anchor_rd: Point, locations) -> LineString:
    """The along-river or along-coast reference line (RD/metric coordinates) relevant to an anchor
    point - the coastline for a storm-surge event, or whichever river branch is nearest for a river
    event (a river event's anchor station always belongs to exactly one branch, but this also lets a
    synthesized event be measured against the branch physically nearest its own anchor)."""
    if event_type == "storm surge":
        return _coastline_axis_rd(locations)
    branch = _nearest_branch(anchor_rd, locations)
    return _branch_axis_rd(RIVER_BRANCHES[branch], locations)


def nearest_template(event_type: str, anchor_lonlat: tuple, ror_centroids: dict) -> int:
    """ror_centroids: {nr: shapely Point, WGS84} for the 6 candidate ROR events."""
    candidates = STORM_SURGE_CANDIDATES if event_type == "storm surge" else RIVER_CANDIDATES
    anchor_rd = _reproject(Point(anchor_lonlat), TO_RD)
    return min(candidates, key=lambda nr: anchor_rd.distance(_reproject(ror_centroids[nr], TO_RD)))


def severity_fraction(event_level: float, reference_level: float) -> float:
    """Ratio of an event's severity to its matched template's, clipped to [0, 1] - nothing should end
    up "more severe" than the real historical flood it's borrowing a reach profile from."""
    if not reference_level or reference_level <= 0:
        return 1.0
    return float(max(0.0, min(1.0, event_level / reference_level)))


def learn_reach_profile(ror_polygon_wgs84, municipalities: list, event_type: str,
                         anchor_lonlat: tuple, locations) -> dict:
    """Measures how far `ror_polygon_wgs84` (one of the 10 real ROR polygons) actually reaches,
    relative to `anchor_lonlat` (its matched candidate's own gauge station), by checking which real
    municipalities it intersects and how far their centroids sit upstream/downstream/inland along
    `axis_for(event_type, ...)`."""
    anchor_rd = _reproject(Point(anchor_lonlat), TO_RD)
    axis_rd = axis_for(event_type, anchor_rd, locations)
    anchor_along = axis_rd.project(anchor_rd)
    ror_polygon_rd = _reproject(ror_polygon_wgs84, TO_RD)

    offsets, perp_dists = [], []
    for feature in municipalities:
        muni_rd = _reproject(shape(feature["geometry"]), TO_RD)
        if muni_rd.intersects(ror_polygon_rd):
            centroid_rd = muni_rd.centroid
            along = axis_rd.project(centroid_rd) - anchor_along
            nearest_on_axis = axis_rd.interpolate(axis_rd.project(centroid_rd))
            offsets.append(along)
            perp_dists.append(centroid_rd.distance(nearest_on_axis))

    if not offsets:
        return {"upstream_km": 0.0, "downstream_km": 0.0, "inland_km": 0.0, "n_municipalities": 0}
    return {
        "upstream_km": max(0.0, -min(offsets)) / 1000,
        "downstream_km": max(0.0, max(offsets)) / 1000,
        "inland_km": max(perp_dists) / 1000,
        "n_municipalities": len(offsets),
    }


def synthesize_extent(profile: dict, event_type: str, anchor_lonlat: tuple, fraction: float,
                       locations, municipalities: list):
    """Scales `profile`'s reach linearly by `fraction` (a lower-severity event reaches less far in
    every direction - a simple, clearly-approximate heuristic, not hydraulically derived), builds a
    corridor search region from `anchor_lonlat`, and returns the union of every real municipality
    polygon that intersects it. Returns (geometry_wgs84, [{"code", "naam"}, ...])."""
    anchor_rd = _reproject(Point(anchor_lonlat), TO_RD)
    axis_rd = axis_for(event_type, anchor_rd, locations)
    anchor_along = axis_rd.project(anchor_rd)

    up_m = profile["upstream_km"] * 1000 * fraction
    down_m = profile["downstream_km"] * 1000 * fraction
    inland_m = max(profile["inland_km"] * 1000 * fraction, MIN_SEARCH_RADIUS_M)

    start_along = max(0.0, anchor_along - up_m)
    end_along = min(axis_rd.length, anchor_along + down_m)
    if end_along - start_along < 1.0:  # degenerate (near-zero) reach - just buffer the anchor point
        search_region_rd = anchor_rd.buffer(inland_m)
    else:
        segment_rd = substring(axis_rd, start_along, end_along)
        search_region_rd = segment_rd.buffer(inland_m)

    selected = [f for f in municipalities
                if _reproject(shape(f["geometry"]), TO_RD).intersects(search_region_rd)]
    if not selected:  # fallback: whichever municipality is nearest the anchor
        selected = [min(municipalities,
                        key=lambda f: _reproject(shape(f["geometry"]), TO_RD).distance(anchor_rd))]

    union_rd = unary_union([_reproject(shape(f["geometry"]), TO_RD) for f in selected])
    union_wgs84 = _reproject(union_rd, TO_WGS84)
    affected = [{"code": f["properties"]["code"], "naam": f["properties"]["naam"]} for f in selected]
    return union_wgs84, affected


def nearest_municipality(anchor_lonlat: tuple, municipalities: list) -> dict:
    """The single municipality containing, or otherwise nearest, anchor_lonlat - "the first
    municipality affected" by an event anchored there. Returns the raw GeoJSON feature (not just its
    geometry), so callers can also read its code/naam."""
    anchor_rd = _reproject(Point(anchor_lonlat), TO_RD)
    return min(municipalities, key=lambda f: _reproject(shape(f["geometry"]), TO_RD).distance(anchor_rd))


def combined_first_municipalities(station_weights: dict, locations, municipalities: list) -> list:
    """The combination of each contributing station's own nearest/containing municipality - not a
    single blended point. Tried averaging the stations' raw lon/lat (and separately, their position
    along the coastline/river axis) first and then finding the nearest municipality to *that* - both
    can land somewhere with no real connection to any contributing station whenever they're not
    roughly collinear (confirmed by direct testing: Den Helder + Vlissingen's flat centroid lands near
    Wassenaar, nowhere near the coast at all). Each station's own nearest_municipality() is
    individually grounded and correct, so the real combination of "first affected" places for an
    event that crossed threshold at several gauges simultaneously is just their union - de-duplicated,
    since two stations can share a municipality. Returns the raw municipality features (not merged
    into one geometry), in no particular order."""
    picked = {}
    for code in station_weights:
        muni = nearest_municipality(station_lonlat(code, locations), municipalities)
        picked[muni["properties"]["code"]] = muni
    return list(picked.values())
