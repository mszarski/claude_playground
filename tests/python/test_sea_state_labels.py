"""A site or table column named by a sea state states that sea state's wind (the owner, 2026-11-26: the labels are
corrected, no number moves). The table is `acoustics.SEA_STATE_WIND_SPEED` (via secondary: UnderwaterAcoustics.jl)."""

import pathlib
import re

import yaml

from gpusonar import acoustics

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAIR = re.compile(r"sea state (\d)\D{0,6}?\(?([0-9.]+) m/s|([0-9.]+) m/s wind \(sea state (\d)")


def pairs(text):
    for a, b, c, d in PAIR.findall(text):
        yield (int(a), float(b)) if a else (int(d), float(c))


def test_a_site_named_by_sea_state_states_the_tables_wind():
    sites = yaml.safe_load((ROOT / "configs" / "builds" / "environments" / "illustrative.yaml").read_text())["environments"]
    named = [s for s in sites if s.get("sea_state") is not None]
    assert named, "no site states a sea state: the test checks nothing"
    for s in named:
        wind = acoustics.sea_state_wind_speed(s["sea_state"])
        assert s.get("wind_speed_m_s") in (None, wind)
        assert f"sea state {s['sea_state']} ({wind:g} m/s wind)" in s["title"], s["title"]


def test_every_sea_state_and_wind_printed_together_agree_with_the_table():
    found = 0
    for rel in ("configs/builds/environments/illustrative.yaml", "tools/popoto_depth_planner.py", "docs/DESIGN.md"):
        for state, wind in pairs((ROOT / rel).read_text()):
            found += 1
            assert wind == acoustics.sea_state_wind_speed(state), f"{rel}: sea state {state} printed with {wind} m/s"
    assert found >= 3
