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
    ("SEA & SKY", "SEA AND SKY"),
    ("HEIDI LYNN", "HEIDI LYNN"),
    ("FDNY BRAVEST", "F D N Y BRAVEST"),
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
    df = df.sort_values("t").reset_index(drop=True)
    snap = ais.snapshot(df, "2024-07-15 14:00", 40.61, -74.06, radius_nm=20, window_min=10)
    assert sorted(v.name for v in snap) == ["EVER GIVEN", "SEA LION"]
    lists = ais.candidate_lists(snap)
    assert lists == {"moving_a": ["EVER GIVEN"], "all": ["EVER GIVEN", "SEA LION"]}


def _fake_region():
    rows = []
    names = [("EVER GIVEN", 70, "A", 9.0), ("MSC ISABELLA", 70, "A", 8.0), ("JOHN J HARVEY", 52, "A", 5.0),
             ("SEA LION", 37, "B", 0.0), ("BLUE MOON", 36, "B", 0.0), ("SECRET SHIP", 70, "A", 7.0)]
    for i, (name, typ, cls, sog) in enumerate(names):
        for minute in range(0, 24 * 60, 5):
            ts = pd.Timestamp("2024-07-15") + pd.Timedelta(minutes=minute)
            rows.append((i + 1, str(ts), 40.61 + 0.01 * i, -74.06, sog, name, typ, cls))
    df = pd.DataFrame(rows, columns=["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "VesselName",
                                     "VesselType", "TransceiverClass"])
    df["t"] = pd.to_datetime(df.BaseDateTime)
    return df.sort_values("t").reset_index(drop=True)


def test_harbor_dialogues_respect_exclusions_and_speaker_kinds():
    from marine_vhf_asr import harbor
    recs = list(harbor.build_dialogues({"new_york": _fake_region()}, 12, seed=3, shares=(0.4, 0.3, 0.3),
                                       unknown_names=["FLYING DUTCHMAN", "SEA LION"],
                                       exclude=frozenset({"SECRET SHIP"}), min_listed=2))
    assert recs and all("SECRET SHIP" not in r["text"] and "SECRET SHIP" not in r["contexts"]["all"] for r in recs)
    kinds = {r["speaker_kind"] for r in recs}
    assert kinds <= {"commercial", "small_craft", "unknown"} and len(kinds) >= 2
    for r in recs:
        if r["speaker_kind"] == "unknown" and r["slots"]:
            # Unknown speakers are on no list ("SEA LION" is heard, so only FLYING DUTCHMAN qualifies)
            assert "FLYING DUTCHMAN" in r["text"] and "FLYING DUTCHMAN" not in r["contexts"]["all"]
