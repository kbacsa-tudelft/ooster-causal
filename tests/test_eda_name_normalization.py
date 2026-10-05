import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'cuts_plus_prototype'))

from eda_compare_datasets import normalize_station  # noqa: E402


def test_water_level_prefix_and_suffix_are_stripped():
    assert normalize_station('WL_BG2_observed_waterlevel') == 'bg2'
    assert normalize_station('WL_Haringvliet_10_observed_waterlevel') == 'haringvliet.10'


def test_rain_channel_loses_its_gauge_id_but_not_station_number():
    assert normalize_station('RH_HOEK-VAN-HOLLAND_330') == 'hoek.van.holland'
    # a trailing number on a water-level station is part of its identity, not a gauge id
    assert normalize_station('WL_Haringvliet_10_observed_waterlevel') == 'haringvliet.10'


def test_rws_style_dotted_name_matches_normalized_full_name():
    assert normalize_station('haringvliet.10') == 'haringvliet.10'
