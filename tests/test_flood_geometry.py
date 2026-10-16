import json
import os
import sys

import pandas as pd
from shapely.geometry import Point, box, mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import extract_ror_events  # noqa: E402
import flood_geometry as fg  # noqa: E402


def _fake_locations():
    """3 stations on a short straight line (close enough together that EPSG:28992 reprojection and
    buffering stay well-behaved)."""
    return pd.DataFrame(
        {"lat": [51.0, 51.0, 51.0], "lon": [5.0, 5.1, 5.2]},
        index=["up", "mid", "down"],
    )


def _fake_municipalities():
    return [
        {"properties": {"code": "A", "naam": "A"}, "geometry": mapping(box(4.95, 50.97, 5.02, 51.03))},
        {"properties": {"code": "B", "naam": "B"}, "geometry": mapping(box(5.07, 50.97, 5.13, 51.03))},
        {"properties": {"code": "C", "naam": "C"}, "geometry": mapping(box(5.17, 50.97, 5.23, 51.03))},
    ]


def test_nearest_template_picks_geographically_correct_nr():
    # Roughly nr1 (1916 Zuiderzeevloed) and nr3 (1953 Watersnoodramp)'s real centroids.
    coastal_centroids = {1: Point(5.38, 52.55), 3: Point(4.11, 51.85)}
    assert fg.nearest_template("storm surge", fg.station_lonlat("delfzijl", pd.DataFrame(
        {"lat": [53.328], "lon": [6.931]}, index=["delfzijl"])), coastal_centroids) == 1
    assert fg.nearest_template("storm surge", (3.6, 51.442), coastal_centroids) == 3  # Vlissingen

    # nearest_template() always filters by the module's RIVER_CANDIDATES ([2, 4, 5, 6]), so every one
    # of those nrs needs an entry here. nr2/nr5 use their real centroids (not under test); nr4/nr6 use
    # their real centroids too, and the 2 query points below are placed right on top of those same
    # centroids - testing the geometric "nearest" logic directly, rather than via a real gauge station,
    # since real stations can be surprisingly ambiguous here (checked by hand: Lobith/Nijmegen sit
    # about as close to nr2's 1926 "Rijn en Maas" centroid as to nr6's - a real, correct outcome of
    # "nearest wins" given the real geography, not a case this test should assert a specific answer for).
    river_centroids = {2: Point(5.74, 51.98), 4: Point(5.73, 51.38),
                       5: Point(5.73, 51.39), 6: Point(5.63, 52.04)}
    assert fg.nearest_template("river high water", (5.73, 51.38), river_centroids) == 4
    assert fg.nearest_template("river high water", (5.63, 52.04), river_centroids) == 6


def test_severity_fraction():
    assert fg.severity_fraction(100.0, 100.0) == 1.0
    assert 0.0 < fg.severity_fraction(50.0, 100.0) < 1.0
    assert fg.severity_fraction(150.0, 100.0) == 1.0  # clipped - never "more severe" than the template
    assert fg.severity_fraction(100.0, 0.0) == 1.0  # guard against a zero/missing reference


def test_learn_reach_profile_measures_known_offsets(monkeypatch):
    monkeypatch.setattr(fg, "RIVER_BRANCHES", {"fake": ["up", "mid", "down"]})
    locations = _fake_locations()
    ror_polygon = box(4.95, 50.95, 5.25, 51.05)  # covers all 3 fake municipalities

    profile = fg.learn_reach_profile(ror_polygon, _fake_municipalities(), "river high water",
                                      anchor_lonlat=(5.1, 51.0), locations=locations)  # anchor = "mid"
    assert profile["n_municipalities"] == 3
    assert profile["upstream_km"] > 0    # municipality A sits upstream of "mid"
    assert profile["downstream_km"] > 0  # municipality C sits downstream of "mid"


def test_synthesize_extent_scales_with_severity(monkeypatch):
    monkeypatch.setattr(fg, "RIVER_BRANCHES", {"fake": ["up", "mid", "down"]})
    locations = _fake_locations()
    municipalities = _fake_municipalities()
    profile = {"upstream_km": 10.0, "downstream_km": 10.0, "inland_km": 5.0, "n_municipalities": 3}
    anchor = (5.1, 51.0)  # "mid"

    _, affected_full = fg.synthesize_extent(profile, "river high water", anchor, 1.0,
                                             locations, municipalities)
    _, affected_low = fg.synthesize_extent(profile, "river high water", anchor, 0.1,
                                            locations, municipalities)
    codes_full = {a["code"] for a in affected_full}
    codes_low = {a["code"] for a in affected_low}
    assert codes_low <= codes_full
    assert len(codes_low) < len(codes_full)  # strictly fewer municipalities at lower severity


def test_extract_round_trips_on_inline_fixture():
    fixture = {
        "layers": [{
            "geojson": {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "properties": {
                        "nr": 1, "jaar": 1916, "naam": "Test", "bron": "", "systeem": "Hoofdwatersysteem",
                        "stroomgebied": "Rijn", "omvang_rapport": "", "gezondheid": "", "economie": "",
                        "opp_polygoon_km2": 1, "kleur": "#fff", "local_id": "NL_TEST1",
                        "extra_map_styling_field": "should be dropped",
                    },
                    "geometry": {"type": "Polygon",
                                 "coordinates": [[[5.0, 52.0], [5.1, 52.0], [5.1, 52.1], [5.0, 52.0]]]},
                }],
            },
        }],
    }
    html = f'<html><script type="application/json" id="geolibre-project">{json.dumps(fixture)}</script></html>'

    result = extract_ror_events.extract(html)

    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) == 1
    props = result["features"][0]["properties"]
    assert props["nr"] == 1
    assert props["naam"] == "Test"
    assert "extra_map_styling_field" not in props
