import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import extract_ror_events  # noqa: E402

# flood_geometry.py used to also hold a flood-extent synthesis pipeline (nearest_template,
# learn_reach_profile, synthesize_extent, weighted_centroid_lonlat, etc.) for the now-deleted
# build_flood_geometries.py - retired along with it once flood_events_municipality.py's real
# municipality joins made the approximation unnecessary. Its remaining functions
# (_reproject/station_name_to_code/station_lonlat) are simple wrappers already exercised indirectly by
# tests/test_flood_events_municipality.py's assign_municipalities() tests.


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
