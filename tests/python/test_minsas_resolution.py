"""The MINSAS-class sonar resolves what its design says: 2.1 cm along track, 1.9 cm in
range, measured on a point target in the survey geometry (docs/DESIGN.md, SAS-1)."""

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import recon
from gpusonar.config import load_scenario_config
from gpusonar.sim import run_scenario
from gpusonar.sonar import SonarModel

HOST = gs.core.Backend.Host


def half_power_width(values, step):
    v = np.abs(values) ** 2
    i = int(v.argmax())
    half = v[i] / 2
    a = i
    while a > 0 and v[a] > half:
        a -= 1
    b = i
    while b < len(v) - 1 and v[b] > half:
        b += 1
    lo = a + (half - v[a]) / (v[a + 1] - v[a])
    hi = b - (half - v[b]) / (v[b - 1] - v[b])
    return (hi - lo) * step


def point_response(speed, ground_range=45.0):
    cfg = load_scenario_config("sas_minsas_boulder_field", {"platform": {"speed": speed}})
    cfg["scene"] = [{"type": "points", "positions": [[16.0, ground_range, 0.0]], "target_strength_db": 0.0}]
    records = run_scenario(cfg, backend=HOST).records
    d = 0.002
    x, y = 16.0 + np.arange(-40, 41) * d, ground_range + np.arange(-40, 41) * d
    X, Y = np.meshgrid(x, y, indexing="ij")
    img = recon.sas_image(records, np.stack([X, Y, np.zeros_like(X)], axis=-1), interp="linear", backend=HOST)
    i, j = np.unravel_index(np.abs(img).argmax(), img.shape)
    xw = 16.0 + np.arange(-400, 401) * 0.01
    cut = np.abs(recon.sas_image(records, np.stack([xw, np.full_like(xw, y[j]), np.zeros_like(xw)], axis=-1)[None],
                                 interp="linear", backend=HOST))[0]
    side = 20 * np.log10(np.where(np.abs(xw - 16.0) > 0.2, cut, 0.0).max() / cut.max())
    return (x[i], y[j]), half_power_width(img[:, j], d), half_power_width(img[i, :], d), side


def test_survey_scenarios_advance_half_the_array_per_ping():
    sonar = SonarModel.load("sas_minsas120")
    half_array = 0.5 * sonar.config["array"]["count"] * sonar.config["array"]["pitch"]
    for name in ("sas_minsas_survey", "sas_minsas_boulder_field", "sas_minsas_rocky_ripples"):
        cfg = load_scenario_config(name)
        assert cfg["platform"]["speed"] * cfg["pings"]["interval"] == pytest.approx(half_array, abs=1e-3), name


def test_point_target_is_resolved_to_the_design_figures():
    peak, along, across, sidelobe = point_response(2.176)
    assert peak == pytest.approx((16.0, 45.0), abs=0.005)
    assert along == pytest.approx(0.021, abs=0.002)              # half the 4.2 cm projector
    assert across == pytest.approx(0.019, abs=0.0015)            # c / 2B with 40 kHz, on the ground at 15 degrees
    assert sidelobe < -30.0                                      # no along-track ambiguities


def test_uneven_phase_centres_raise_along_track_ambiguities():
    # 0.4 m per ping against the 0.544 m half-array: phase centres of successive pings overlap,
    # the sampling along track is uneven, and ghosts appear a few decimetres from every target.
    _, along, _, sidelobe = point_response(1.6)
    assert along == pytest.approx(0.021, abs=0.002)
    assert -20.0 < sidelobe < -10.0


def test_weighting_for_display_trades_range_width_for_sidelobes():
    # The survey figures use a Hamming-weighted replica and a beam-weighted aperture 1.6 beamwidths long.
    cfg = load_scenario_config("sas_minsas_boulder_field")
    cfg["scene"] = [{"type": "points", "positions": [[16.0, 50.0, 0.0]], "target_strength_db": 0.0}]
    out = {}
    for window in ("none", "hamming"):
        records = run_scenario(cfg, backend=HOST).records
        yw = 50.0 + np.arange(-250, 251) * 0.002
        xw = 16.0 + np.arange(-250, 251) * 0.002
        kw = dict(interp="linear", fov_factor=1.6, directivity_weight=True, replica_window=window, backend=HOST)
        across = np.abs(recon.sas_image(records, np.stack([np.full_like(yw, 16.0), yw, np.zeros_like(yw)], axis=-1)[None], **kw))[0]
        along = np.abs(recon.sas_image(records, np.stack([xw, np.full_like(xw, 50.0), np.zeros_like(xw)], axis=-1)[None], **kw))[0]
        far = 20 * np.log10(np.where(np.abs(yw - 50.0) > 0.1, across, 0.0).max() / across.max())
        near_along = 20 * np.log10(np.where(np.abs(xw - 16.0) > 0.05, along, 0.0).max() / along.max())
        out[window] = (half_power_width(across, 0.002), far, half_power_width(along, 0.002), near_along)
    assert out["none"][0] == pytest.approx(0.019, abs=0.0015) and out["hamming"][0] == pytest.approx(0.026, abs=0.002)
    assert out["none"][1] > -30.0 and out["hamming"][1] < -38.0             # range sidelobes beyond 10 cm
    for window in out:
        assert out[window][2] == pytest.approx(0.020, abs=0.002)            # along track unchanged
        assert out[window][3] < -33.0                                       # against -23 dB with the default aperture


@pytest.mark.parametrize("count, width, limit", [(48, 0.022, (-200.0, -50.0)), (32, 0.030, (-46.0, -38.0))])
def test_along_track_ambiguity_falls_with_finer_element_pitch(count, width, limit):
    # Receive phase centres half a pitch apart put a grating lobe at asin(wavelength / pitch): 7.6 degrees
    # for 32 elements over 1.088 m, 11 degrees for the shipped 48. Each scatterer has a defocused copy
    # that far up and down the track; the beam patterns set how strong it is.
    cfg = load_scenario_config("sas_minsas_boulder_field",
                               {"sonar_overrides": {"array": {"count": count, "pitch": 1.088 / count, "element_width": width}}})
    cfg["scene"] = [{"type": "points", "positions": [[16.0, 50.0, 0.0]], "target_strength_db": 0.0}]
    cfg.pop("synthesis", None)                                   # keep paths far down the beams
    records = run_scenario(cfg, backend=HOST).records
    kw = dict(interp="linear", fov_factor=1.6, directivity_weight=True, replica_window="hamming", backend=HOST)
    peak = np.abs(recon.sas_image(records, np.array([[[16.0, 50.0, 0.0]]]), **kw)).max()
    x, y = np.arange(4.0, 30.0, 0.04), np.arange(49.5, 51.5, 0.04)
    X, Y = np.meshgrid(x, y, indexing="ij")
    img = np.abs(recon.sas_image(records, np.stack([X, Y, np.zeros_like(X)], axis=-1), **kw))
    img[np.abs(X - 16.0) < 1.0] = 0.0
    ghost = 20 * np.log10(img.max() / peak)
    assert limit[0] < ghost < limit[1]
