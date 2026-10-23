"""Tests for AIS helpers (no network)."""

import pytest

pd = pytest.importorskip("pandas")

from marine_vhf_asr import ais  # noqa: E402


@pytest.mark.parametrize("raw,spoken", [
    ("SCORPIO 2", "SCORPIO TWO"),
    ("NYPD353", "N Y P D THREE FIVE THREE"),
    ("BUCHANAN #12", "BUCHANAN TWELVE"),
    ("MSC ISABELLA", "M S C ISABELLA"),
    ("JOHN J HARVEY", "JOHN J HARVEY"),
    ("ARMY CORPS NYSB-4", "ARMY CORPS N Y S B FOUR"),
    ("SEA & SKY", "SEA AND S K Y"),
])
def test_spoken_name(raw, spoken):
    assert ais.spoken_name(raw) == spoken


@pytest.mark.parametrize("raw", ["", "123456789", "UNKNOWN", "@@"])
def test_unspeakable_names(raw):
    assert ais.spoken_name(raw) is None


def test_type_group():
    assert ais.type_group(52) == "tug" and ais.type_group(71.0) == "cargo"
    assert ais.type_group(None) == "unknown" and ais.type_group(99) == "other"


def test_snapshot_window_radius_and_lists():
    rows = [
        # MMSI, time, lat, lon, SOG, name, type, class
        (1, "2024-07-15 13:55", 40.61, -74.06, 8.0, "EVER GIVEN", 70, "A"),     # in range, moving, class A
        (2, "2024-07-15 13:58", 40.62, -74.05, 0.0, "SEA LION", 37, "B"),       # in range, moored pleasure
        (3, "2024-07-15 13:59", 41.50, -74.06, 9.0, "FAR AWAY", 70, "A"),       # ~53 nm away
        (4, "2024-07-15 13:30", 40.61, -74.06, 9.0, "TOO EARLY", 70, "A"),      # outside 10-minute window
    ]
    df = pd.DataFrame(rows, columns=["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "VesselName",
                                     "VesselType", "TransceiverClass"])
    df["t"] = pd.to_datetime(df.BaseDateTime)
    snap = ais.snapshot(df, "2024-07-15 14:00", 40.61, -74.06, radius_nm=20, window_min=10)
    assert sorted(v.name for v in snap) == ["EVER GIVEN", "SEA LION"]
    lists = ais.candidate_lists(snap)
    assert lists == {"moving_a": ["EVER GIVEN"], "all": ["EVER GIVEN", "SEA LION"]}
