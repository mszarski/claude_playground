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
    assert len(reqs) >= 20
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
    assert sas["along_track_resolution_cm"] == pytest.approx(3.3)
    assert sas["range_resolution_cm"] == pytest.approx(1.875)
    assert sas["max_advance_per_ping_m"] == pytest.approx(0.544)
    assert D["popoto_array"]["array_gain_db"] == pytest.approx(10 * np.log10(32))


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
