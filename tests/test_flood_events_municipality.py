import json
import os
import sys

import pandas as pd
from shapely.geometry import box, mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flood_events_municipality as fem  # noqa: E402
from flood_events import merge_national  # noqa: E402


def test_default_stations_bypass_the_exclusion_heuristic_entirely():
    """dordrecht.oudemaas.benedenmerwede and maastricht.borgharen.maas.beneden would otherwise be
    caught by the boven/beneden pattern - confirmed by direct testing against the real 595 station
    names. They must never be excluded, since they're 2 of the 11 trusted DEFAULT_STATIONS."""
    locations = pd.DataFrame({"name": ["dordrecht.oudemaas.benedenmerwede",
                                        "maastricht.borgharen.maas.beneden"]})
    channels = pd.DataFrame({"Code": locations["name"], "Naam": ["Dordrecht", "Borgharen"]})

    stations, audit = fem.select_open_water_candidates(locations, channels)

    assert len(stations) == 2
    assert audit["included"].all()
    assert (audit["exclusion_reason"] == "already_default").all()


def test_select_open_water_candidates_classification():
    locations = pd.DataFrame({"name": [
        "ijmuiden.noordersluis.oost",   # substring:sluis
        "somewhere.kanaaltje",          # substring:kanaal
        "arcen.boven",                  # boven_beneden (token boundary)
        "arcenboven",                   # NOT boven_beneden - no token boundary, should survive
        "scheveningen",                 # plain name, no match - included
    ]})
    channels = pd.DataFrame({"Code": locations["name"], "Naam": locations["name"]})

    stations, audit = fem.select_open_water_candidates(locations, channels)

    reasons = audit.set_index("code")["exclusion_reason"]
    assert reasons["ijmuiden.noordersluis.oost"] == "substring:sluis"
    assert reasons["somewhere.kanaaltje"] == "substring:kanaal"
    assert reasons["arcen.boven"] == "boven_beneden"
    assert pd.isna(reasons["arcenboven"])
    assert pd.isna(reasons["scheveningen"])
    included_codes = {st.code for st in stations}
    assert included_codes == {"arcenboven", "scheveningen"}
    assert all(st.kind == "river" for st in stations)  # no official SURGE_CLASSES beyond the 6 coastal


def test_assign_municipalities_drops_stations_beyond_max_join_km():
    locations = pd.DataFrame({"name": ["near", "far"], "lat": [52.0, 55.5], "lon": [5.0, 3.0]})
    municipalities = [{"properties": {"code": "X", "naam": "X"},
                        "geometry": mapping(box(4.95, 51.95, 5.05, 52.05))}]

    result = fem.assign_municipalities(["near", "far"], locations, municipalities, max_join_km=15.0)

    assert not result.loc["near", "offshore_or_outside_nl"]
    assert result.loc["near", "municipality_code"] == "X"
    assert result.loc["far", "offshore_or_outside_nl"]
    assert pd.isna(result.loc["far", "municipality_code"])


def test_merge_national_defaults_are_unchanged_for_the_original_kind_call_site():
    """The group_col="kind"/gap_days=None defaults must reproduce flood_events.py's own original,
    single-argument call site (merge_national(per_station)) exactly - verified in this session by
    diffing a full real run's events.csv/station_events.csv byte-for-byte before and after this
    signature change landed."""
    ev = pd.DataFrame([
        {"station": "A", "code": "a", "kind": "coast", "start": pd.Timestamp("2020-01-01"),
         "end": pd.Timestamp("2020-01-02"), "peak_time": pd.Timestamp("2020-01-01 12:00"),
         "peak_level": 400, "alert": "high", "detail": ""},
    ])
    out = merge_national(ev)
    assert list(out.columns) == ["start", "end", "peak_date", "type", "alert", "worst_station",
                                  "worst_detail", "worst_level_cm_nap", "stations"]
    assert out.iloc[0]["type"] == "storm surge"


def test_merge_national_groups_by_an_arbitrary_column_with_a_flat_gap():
    """flood_events_municipality.py's own call site: group_col="municipality_code", a flat gap_days
    (not the per-kind MERGE_DAYS lookup, since a municipality may mix coastal+river stations)."""
    ev = pd.DataFrame([
        {"station": "A", "code": "a", "kind": "coast", "municipality_code": "M1",
         "start": pd.Timestamp("2020-01-01"), "end": pd.Timestamp("2020-01-02"),
         "peak_time": pd.Timestamp("2020-01-01 12:00"), "peak_level": 400, "alert": "high", "detail": ""},
        {"station": "B", "code": "b", "kind": "river", "municipality_code": "M1",
         "start": pd.Timestamp("2020-01-03"), "end": pd.Timestamp("2020-01-04"),
         "peak_time": pd.Timestamp("2020-01-03 12:00"), "peak_level": 300, "alert": "medium", "detail": ""},
        {"station": "C", "code": "c", "kind": "river", "municipality_code": "M2",
         "start": pd.Timestamp("2020-01-01"), "end": pd.Timestamp("2020-01-02"),
         "peak_time": pd.Timestamp("2020-01-01 12:00"), "peak_level": 200, "alert": "low", "detail": ""},
    ])
    out = merge_national(ev, group_col="municipality_code", gap_days=5)

    assert "type" not in out.columns
    assert "municipality_code" in out.columns
    m1 = out[out["municipality_code"] == "M1"]
    assert len(m1) == 1  # A and B merged - same municipality, 2 days apart, within the 5-day gap
    assert m1.iloc[0]["alert"] == "high"  # the worse of the 2 contributing stations
    m2 = out[out["municipality_code"] == "M2"]
    assert len(m2) == 1
