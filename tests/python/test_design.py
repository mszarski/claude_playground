"""The design is traceable: every owner requirement is met by the config files, the design
report is generated from them and is up to date, and the configuration knobs it relies on
(field of view, array size, boards) do what they say."""

import importlib.util
import pathlib

import numpy as np
import pytest

from gpusonar.sim import make_pose
from gpusonar.sonar import SonarModel

pytest.importorskip("matplotlib", reason="the design report draws its diagrams with matplotlib")
ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("design_report", ROOT / "tools" / "design_report.py")
design_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(design_report)


@pytest.fixture(scope="module")
def built():
    return design_report.build(write=False)


def test_every_requirement_is_met(built):
    _, reqs, _ = built
    assert len(reqs) >= 30
    assert {"POP-3", "POP-4", "FLS-X", "SAS-4"} <= {r["id"] for r in reqs}    # 12 km, environments, clutter, rocks
    assert [r["id"] for r in reqs if not r["met"]] == []
    for r in reqs:
        assert (ROOT / r["set_in"].split(":")[0]).exists(), r["set_in"]       # the file each one points to exists


def test_design_document_is_up_to_date(built):
    _, _, text = built
    on_disk = (ROOT / "docs" / "DESIGN.md").read_text().split("<!-- measured -->")[0]
    assert on_disk.strip() == text.strip(), "run tools/design_report.py"


def test_derived_numbers_follow_from_the_configs(built):
    D, _, _ = built
    for name, f in (("fls_120k", 120e3), ("fls_170k", 170e3), ("fls_330k", 330e3)):
        d = D[name]
        assert d["wavelength_mm"] == pytest.approx(1500.0 / f * 1e3, rel=1e-6)
        assert d["array_length_m"] == pytest.approx(d["elements"] * d["pitch_mm"] / 1e3)
        assert d["grating_lobe_free_steering_deg"] >= 60.0                    # the whole 120 degree sector
        assert 0.7 < d["beamwidth_deg"] < 0.9
        assert d["min_ping_interval_s"] == pytest.approx(2 * d["max_range_m"] / 1500.0 + d["pulse_ms"] / 1e3)
        assert d["raw_data_rate_mbit_s"] == pytest.approx(d["elements"] * d["adc_rate_khz"] * 1e3 * d["adc_bits"] / 1e6)
    sas = D["sas_minsas120"]
    assert sas["along_track_resolution_cm"] == pytest.approx(2.1)
    assert sas["range_resolution_cm"] == pytest.approx(1.875)
    assert sas["max_advance_per_ping_m"] == pytest.approx(0.544, abs=1e-4)
    assert D["popoto_array"]["array_gain_db"] == pytest.approx(10 * np.log10(32), abs=0.02)        # against isotropic noise: 10 log N within 0.02 dB at this pitch
    far = D["popoto_long_range"]
    assert far["centre_frequency_khz"] == pytest.approx(5.0) and far["elements"] == 32
    assert far["pitch_in_wavelengths_at_top_of_band"] <= 0.61               # 0.15 m at 6 kHz
    # 12 km: one row's SNR against what the chain's own detector needs of it (no typed threshold); the 27.5 kHz array is nowhere near
    from gpusonar import buildspec as bs
    need = {n: bs.detection(bs.derive(bs.load(f"popoto_face_{n}x32")))["threshold_db"] for n in (1, 3)}
    assert need[3] < far["_eq"].snr(12000.0) < need[1] and D["popoto_array"]["_eq"].snr(12000.0) < 0.0
    assert far["array_gain_db"] == pytest.approx(bs.directivity_db(32, 0.15, 5000.0), abs=1e-9)        # from_sonar: the array's gain against isotropic noise


def test_twelve_km_study_matches_the_sonar_equation():
    """tools/popoto_range_study.py and the link budget in DESIGN.md are separate code: they must agree."""
    spec = importlib.util.spec_from_file_location("study", ROOT / "tools" / "popoto_range_study.py")
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    eq = design_report.build(write=False)[0]["popoto_long_range"]["_eq"]
    need = float(study.required_source_level(eq.center_frequency, 12000.0, 32, 25.0, eq.pulse_duration, 13.0))
    assert need == pytest.approx(eq.source_level_db - (eq.snr(12000.0) - 13.0), abs=0.6)
    # Lower frequency is better down to about 4 kHz, then noise and array length turn it round.
    sl = study.required_source_level(np.array([2e3, 4e3, 8e3, 16e3, 27.5e3]), 12000.0, 32, 25.0, 1.0, 13.0)
    assert sl[1] < sl[0] and sl[1] < sl[2] < sl[3] < sl[4] and sl[4] > 300.0
    assert study.acoustic_power_w(170.8) == pytest.approx(1.0) and study.acoustic_power_w(200.8) == pytest.approx(1000.0)


def test_fov_block_sets_beams_and_tilt():
    s = SonarModel.load("fls_330k")
    assert np.degrees(s.beamwidth_az) == pytest.approx(120.0) and np.degrees(s.beamwidth_el) == pytest.approx(5.0)
    assert np.degrees(np.arcsin(-s.boresight[2])) == pytest.approx(3.5)        # up is -z
    rec = s.make_record(make_pose())
    assert np.degrees(rec.geometry.tx.beamwidth_el) == pytest.approx(5.0)
    assert np.allclose(rec.geometry.tx.boresight, s.boresight)
    # The vertical field of view and the tilt are adjustable from an override.
    s2 = SonarModel.load("fls_330k", {"fov": {"vertical_deg": 12.0, "tilt_up_deg": -8.0}})
    assert np.degrees(s2.beamwidth_el) == pytest.approx(12.0)
    assert np.degrees(np.arcsin(-s2.boresight[2])) == pytest.approx(-8.0)
    # An explicit beamwidth wins over the block.
    s3 = SonarModel.load("fls_330k", {"array": {"beamwidth_el": 0.2}})
    assert s3.beamwidth_el == pytest.approx(0.2)
    with pytest.raises(ValueError, match="horizontal"):
        SonarModel.load("fls_330k", {"array": {"boresight": [1.0, 0.0, -0.2]}})


@pytest.mark.parametrize("name", ["fls_120k", "fls_170k", "fls_330k", "sas_minsas120"])
def test_element_count_and_pitch_are_configurable(name):
    base = SonarModel.load(name)
    wide = SonarModel.load(name, {"array": {"count": 2 * base.n_elements}})
    assert wide.n_elements == 2 * base.n_elements
    assert wide.aperture == pytest.approx(base.aperture * (2 * base.n_elements - 1) / (base.n_elements - 1))
    fine = SonarModel.load(name, {"array": {"pitch": 0.5 * float(base.config["array"]["pitch"])}})
    assert fine.aperture == pytest.approx(0.5 * base.aperture)
    rec = wide.make_record(make_pose())
    assert rec.samples.n_channels == 2 * base.n_elements


def test_popoto_array_scales_by_one_number():
    for n, boards in ((8, 1), (16, 2), (32, 4), (64, 8), (44, 6)):
        s = SonarModel.load("popoto_array", {"array": {"count": n}})
        assert s.n_elements == n and len(s.boards) == boards
        assert s.channel_board.tolist() == [i // 8 for i in range(n)]
        rec = s.make_record(make_pose())
        assert rec.samples.n_channels == n and len(rec.metadata.boards) == boards
    # Another band: the pitch follows by hand, the boards do not care.
    low = SonarModel.load("popoto_array", {"array": {"count": 32, "pitch": 0.094},
                                           "transmit": {"waveform": {"f_start": 6e3, "f_stop": 10e3}}})
    assert low.aperture == pytest.approx(31 * 0.094)
    # With the defaults it is the stock eight-hydrophone board.
    assert np.allclose(SonarModel.load("popoto_array").positions, SonarModel.load("popoto_8").positions)


# --- POP-3: whom the 12 km requirement is for (the owner, 2026-11-26; D-235) -----------------------------------------
# The owner: "Large submerged mostly but also Large Ship is what we care about". POP-3 is checked for `large_submerged`
# first and `ship` second, at the free-field design point and at every illustrative site of the depth plan; the
# requirement row may not read as a bare pass while a listed site fails.

import json as _json
import re as _re

_PLAN = ROOT / "docs" / "design" / "popoto_depth_planner.json"
_TRADE = ROOT / "docs" / "design" / "popoto_trade_study.json"


def _pop3(reqs):
    return next(r for r in reqs if r["id"] == "POP-3")


def _sites():
    """The trade study's illustrative sites (docs/design/popoto_trade_study.json) and the shipped face's plans for them."""
    sites = [s["name"] for s in _json.loads(_TRADE.read_text())["sites"]]
    plans = _json.loads(_PLAN.read_text())["builds"]["popoto_face_3x32"]["plans"]
    return sites, plans


def _section(text, title):
    return text.split(f"### {title}\n", 1)[1].split("\n### ", 1)[0]


def test_pop3_names_both_classes_large_submerged_first(built):
    _, reqs, _ = built
    r = _pop3(reqs)
    assert r["classes"] == ["large_submerged", "ship"]
    words = r["requirement"]
    assert "Large submerged mostly but also Large Ship is what we care about" in words and "2026-11-26" in words
    assert words.index("`large_submerged`") < words.index("`ship`")
    assert r["value"].index("large_submerged") < r["value"].index("ship")
    for open_question in ("sea state", "detection probability", "false-alarm", "operating area"):    # still the owner's to say
        assert open_question in r["value"], open_question


def test_pop3_cell_is_not_a_bare_pass_while_a_site_fails(built):
    _, reqs, text = built
    r = _pop3(reqs)
    sites, plans = _sites()
    failing = sorted({s for s in sites for c in ("large_submerged", "ship") if plans[s]["classes"][c]["recommended"]["margin_db"] < 0.0})
    assert failing, "the committed plan has failing sites: this test is written for that case"
    row = next(line for line in text.splitlines() if line.startswith("| POP-3 |"))
    cell = row.rstrip(" |").rsplit(" | ", 1)[1]
    assert cell.strip() not in ("yes", "**yes**") and "not" in cell
    for s in failing:
        assert s in cell, s
    assert r["status"] == "partly met"
    # The design-point check itself (free field, both classes, three rows) is what `met` and --check hold.
    assert r["met"] is True


def test_design_point_table_per_site_equals_the_planner(built):
    _, _, text = built
    sec = _section(text, "The 12 km design point")
    sites, plans = _sites()
    trade = _json.loads(_TRADE.read_text())["answers"]["per_class_site"]
    # The section leads with the two named classes, the primary first, and keeps xl_auv as information only.
    head = sec.strip().split("\n\n", 1)[0]
    assert head.index("`large_submerged`") < head.index("`ship`")
    assert any("xl_auv" in line and "not the requirement" in line for line in sec.splitlines())
    rows = {}
    for line in sec.splitlines():
        m = _re.match(r"\| `(\w+)`[^|]*\| `(large_submerged|ship)` \| ([+-]\d+\.\d) dB \| (yes|no) \|.*\| (\d+\.\d) km[^|]*\| (\d+) m \|$", line)
        if m:
            rows[(m[1], m[2])] = (float(m[3]), m[4], float(m[5]), float(m[6]), len(rows))
    for s in sites:
        for c in ("large_submerged", "ship"):
            assert (s, c) in rows, (s, c)
            margin, held, reach_km, depth, _ = rows[(s, c)]
            rec = plans[s]["classes"][c]["recommended"]
            assert margin == pytest.approx(rec["margin_db"], abs=0.051)                         # printed to 0.1 dB
            assert margin == pytest.approx(trade[c][s]["shipped_3x32"]["A"]["margin_db"], abs=0.051)
            assert held == ("yes" if rec["margin_db"] >= 0.0 else "no")
            assert reach_km == pytest.approx(rec["reach_m"] / 1e3, abs=0.051)
            assert depth == pytest.approx(rec["own_depth_m"], abs=0.5)
    # Worst first, by the primary class.
    order = sorted(sites, key=lambda s: rows[(s, "large_submerged")][4])
    lsm = [plans[s]["classes"]["large_submerged"]["recommended"]["margin_db"] for s in order]
    assert lsm == sorted(lsm)


def test_pop3_free_field_figures_equal_the_planners(built):
    """The design point's per-class free-field margin (buildspec.performance's ship-sized entry moved by each class's low
    target strength) equals the depth plan's own free-field margin at every site of the design point's 5 m/s wind."""
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import reference_plan as rp
    X = rp.twelve_km_by_site()
    sites, plans = _sites()
    calm = [s for s in sites if plans[s]["wind_speed_m_s"] == 5.0]
    assert len(calm) >= 4
    for c in rp.REQUIREMENT_CLASSES + ("xl_auv",):
        for s in calm:
            assert X["free"][c]["margin_db"] == pytest.approx(plans[s]["classes"][c]["free_field_margin_db"], abs=0.01), (c, s)
