"""configs/scenarios/asiaex_dahl_2004.yaml against what Dahl (2004) prints and its geometry gives
(docs/references/CONFORMANCE.md A7).

Source: P. H. Dahl, JASA 115, 589-599 (2004), Sec. IV.A, read in
docs/references/surface_forward_scatter_dahl_1996_2004.md ("[NF]") Sec. 2.6 (printed: geometry, sea,
pulses) and Secs. 2.6-2.7 (read from figures: Fig. 3c arrival times, Fig. 5 coherence, +-0.03). The
numbers our geometry gives are computed here, not printed. Not a reproduction of the paper's
coherence curves: its model (Dahl 1999's small-slope approximation with a measured spectrum) is not
in hand; the vertical coherence is recorded as an expected failure.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import sst_validation as sv  # noqa: E402

from gpusonar import acoustics  # noqa: E402
from gpusonar.config import load_scenario_config  # noqa: E402
from gpusonar.sim import ImageSourceProvider, run_scenario  # noqa: E402
from gpusonar.sim.timespread import boundary_parameters, wrap  # noqa: E402

NAME = "asiaex_dahl_2004"


@pytest.fixture(scope="module")
def cfg():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_scenario_config(NAME)


def _geometry(cfg):
    src = np.asarray(cfg["sources"][0]["origin"], dtype=float)
    rx = np.asarray(cfg["platform"]["position"], dtype=float)
    return src, rx, float(cfg["environment"]["sound_speed"]), float(cfg["propagation"]["bottom_z"])


def test_the_scenario_holds_what_dahl_2004_prints(cfg):
    src, rx, c, depth = _geometry(cfg)
    assert src[2] == 25.0 and rx[2] == 52.0 and depth == 105.0                      # source, lower array, water
    assert float(np.hypot(*(src[:2] - rx[:2]))) == 467.0                             # Fig. 4 caption
    wf = cfg["sonar_overrides"]["transmit"]["waveform"]
    assert wf["type"] == "cw" and wf["f_start"] == wf["f_stop"] == 8000.0 and wf["duration"] == pytest.approx(3e-3)
    assert cfg["time_spread"]["surface"]["wind_speed"] == 7.0 and cfg["time_spread"]["surface"]["rms_height"] == 0.3
    assert cfg["pings"]["count"] == 20 and cfg["pings"]["transmit"] is False
    # What the paper does not print is named.
    for block in ("environment.sound_speed", "environment.bottom", "sonar_overrides", "pings.interval"):
        assert block in cfg["unconfirmed"]
    # The surface's statistics: the measured height, the long-wave slope (CONFORMANCE.md A4).
    par = boundary_parameters(cfg["time_spread"]["surface"], surface=True)
    assert par["rms_height"] == 0.3 and acoustics.rms_wave_height(7.0) == pytest.approx(0.262, abs=1e-3)
    assert par["mean_square_slope"] == pytest.approx(0.0046 * np.log(2.1 * 49.0)) == pytest.approx(0.0213, abs=1e-4)
    assert np.sqrt(par["mean_square_slope"]) == pytest.approx(0.146, abs=5e-4)                # Dahl 2004 p. 599: 0.146 (and 0.152)


def test_the_geometry_gives_the_printed_grazing_angle_and_the_arrival_times(cfg):
    """[NF] Sec. 2.6: iso-speed specular angle at 52 m 9.4 degrees (with refraction 10.9). Computed
    here: surface arrival 3.62 ms and bottom arrival 11.67 ms after the direct one at 1525 m/s
    (3.62-3.68 / 11.6-11.9 over 1521-1528 m/s and the 467-500 m of the text, CONFORMANCE.md A7)."""
    src, rx, c, depth = _geometry(cfg)
    assert np.degrees(np.arctan2(src[2] + rx[2], 467.0)) == pytest.approx(9.4, abs=0.05)
    p = ImageSourceProvider(c, surface_z=0.0, bottom_z=depth, max_order=2, bottom_coefficient=0.5)
    e = p.eigenrays(src, rx[None, :])
    ok = e.valid[0]
    d, ns, nb = e.delay[0][ok], e.n_surface[0][ok], e.n_bottom[0][ok]
    first = d.min()
    surface = float((d[(ns == 1) & (nb == 0)][0] - first) * 1e3)
    bottom = float((d[(ns == 0) & (nb == 1)][0] - first) * 1e3)
    assert 3.60 < surface < 3.68 and surface > 3.0                                    # clear of the 3 ms direct pulse
    assert 11.6 < bottom < 11.9
    # Dahl 2004 Fig. 3c shows the bottom arrival at about 13.7 ms (read from figure) in water that refracts:
    # 2 ms later than straight rays in constant 1525 m/s give. Recorded, not fitted.
    assert 13.7 - bottom == pytest.approx(2.0, abs=0.3)


def test_the_record_has_the_arrivals_where_the_geometry_puts_them(cfg):
    """One ping without time spreading: the half-amplitude onsets of direct, surface and bottom
    arrivals in the record against the image delays (the window opens 5 ms before the direct one)."""
    src, rx, c, depth = _geometry(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_scenario(NAME, {"time_spread": None, "pings": {"count": 1}})
    rec = res.records[0]
    fs, t0 = rec.samples.sample_rate, rec.samples.start_time
    x = sv.samples_of(res, 0)[0]
    env = np.abs(sv.envelope(x, fs, 8000.0, 4000.0))
    t = t0 + np.arange(len(x)) / fs
    p = ImageSourceProvider(c, surface_z=0.0, bottom_z=depth, max_order=2, bottom=dict(cfg["propagation"]["bottom"]))
    e = p.eigenrays(src, rx[None, :])
    ok = e.valid[0]
    d, ns, nb = e.delay[0][ok], e.n_surface[0][ok], e.n_bottom[0][ok]
    first = float(d.min())
    assert first - 5e-3 == pytest.approx(t0, abs=0.2e-3)                               # the window as stated (0.1 ms of rounding)
    wf = cfg["sonar_overrides"]["transmit"]["waveform"]
    rise = 0.25 * wf["window_alpha"] * wf["duration"]                                  # the taper is at half amplitude here (75 us)
    for sel in ((ns == 0) & (nb == 0), (ns == 1) & (nb == 0), (ns == 0) & (nb == 1)):
        when = float(d[sel][0])
        win = (t > when - 0.5e-3) & (t < when + 1.5e-3)
        plateau = float(np.median(env[(t > when + 0.9e-3) & (t < when + 2.1e-3)]))
        k = np.flatnonzero(win & (env > 0.5 * plateau))[0]
        assert t[k] - rise == pytest.approx(when, abs=25e-6)                        # measured 2-13 us late: one sample is 15.6 us


def test_at_7_m_s_and_0_3_m_the_surface_bounce_has_no_coherent_part(cfg):
    """[NF] Sec. 3.1: the coherent share exp(-Gamma^2) at 8 kHz, 9.4 degrees, H 0.3 m is 3e-5 (Gamma^2
    10.3); the papers work at chi >> 1. In the record: the surface arrival's peak varies from ping to
    ping as a fully rough one does."""
    src, rx, c, depth = _geometry(cfg)
    g = np.arctan2(src[2] + rx[2], 467.0)
    gamma2 = (2.0 * 2.0 * np.pi * 8000.0 / c * 0.3 * np.sin(g)) ** 2
    assert gamma2 == pytest.approx(10.35, abs=0.05) and np.exp(-gamma2) < 1e-4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_scenario(NAME, {"propagation": {"bottom_z": None}})
    rec = res.records[0]
    fs, t0 = rec.samples.sample_rate, rec.samples.start_time
    x = sv.samples_of(res, 0)
    env2 = np.abs(sv.envelope(x, fs, 8000.0, 4000.0)) ** 2
    t = t0 + np.arange(x.shape[-1]) / fs
    first = float(np.hypot(467.0, 27.0) / c)
    direct = (t > first + 0.9e-3) & (t < first + 2.1e-3)
    surf = (t > first + 3.62e-3 + 0.9e-3) & (t < first + 3.62e-3 + 2.1e-3)
    level = lambda sel: 10.0 * np.log10(env2[:, sel].mean(axis=1))        # noqa: E731
    assert level(direct).std() < 1e-3                                       # the direct path is steady
    assert level(surf).std() > 2.0                                          # faded (3.4 dB over 1.2 ms of the pulse; one Rayleigh sample: 5.6)


@pytest.mark.xfail(strict=True, reason="Vertical coherence at low grazing is about 2 to 6 times short in our facet model: at the "
                   "lower array (52 m), 20 kHz, 0.30 m Dahl 2004 Fig. 5 shows |Gamma_V| 0.69 (read from figure; 0.57 at 26 m); "
                   "ours gives about 0.1 (0.03-0.2 with the Cox-Munk slope, [NF] Sec. 3.2) and geometric optics 0.48. Dahl's "
                   "small-slope model (Dahl 1999, not in hand) fits. CONFORMANCE.md A7; [NF] WP6")
def test_vertical_coherence_at_the_lower_array_at_20_khz_reaches_the_measured_value(cfg):
    """The van Cittert-Zernike form Dahl 2004 uses: the coherence across a vertical separation d is
    the intensity-weighted mean of exp(i k d u_z) over the scattered arrivals' directions u, taken
    here from our taps (`angle_spread`) over 4000 independent realisations of the scenario's surface."""
    src, rx, c, depth = _geometry(cfg)
    f = 20e3
    inner = ImageSourceProvider(c, surface_z=0.0, bottom_z=None, max_order=1)
    prov = wrap(inner, {"surface": dict(cfg["time_spread"]["surface"]), "taps": 24, "angle_spread": True,
                        "coherence_length": 1e-3, "seed": 1}, frequency=f)
    n = 4000
    pts = np.repeat(rx[None, :], n, axis=0)
    pts[:, 1] += np.arange(n) * 2e-3                                        # a new realisation each, the same geometry
    e = prov.eigenrays(src, pts)
    weight = np.abs(e.amplitude) ** 2 * ((e.n_surface == 1) & e.valid)
    k = 2.0 * np.pi * f / c
    gamma = float(np.abs((weight * np.exp(1j * k * 0.30 * e.arrival[..., 2])).sum()) / weight.sum())
    assert gamma >= 0.5
