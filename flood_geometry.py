"""
Small shared geometry helpers (reprojection, station lookups) used by fetch_municipalities.py and
flood_events_municipality.py.

This module used to also hold a flood-extent *synthesis* pipeline (reach profiles borrowed from the 10
ROR historical polygons, severity-scaled corridor search, nearest-municipality approximation) for
build_flood_geometries.py, back when the only detection available ran at 11 curated national gauges and
a municipality had to be guessed after the fact. flood_events_municipality.py now detects anomalies at
408 real stations, each already joined to its own real municipality - build_municipality_flood_geometries.py
reads that directly (an exact municipality_code lookup, no approximation needed), so the whole
synthesis pipeline (and build_flood_geometries.py itself) was retired.
"""
import numpy as np
import shapely
from pyproj import Transformer

from flood_events import DEFAULT_STATIONS

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
