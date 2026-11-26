"""The link budget's reverberation term (issue sonar-w2w3, D-198): gpusonar/reverb_budget.py (the Python reference: the
simulator's own scattering-function integral), `SonarEquation(reverberation_db=...)`, and the coverage picture's term
in C++ (libs/live coverage.hpp, both chain engines).

Held here: the term is Jackson and Richardson 2007 (G.11)-(G.13) for one path each way; the C++ rays and the Python
reference's image sources give the same level; its limits (zero strength is the noise-only budget exactly; a
reverberation-limited SNR does not move with the source level; a CW pulse twice as long has 3 dB more); the level
against the simulated reverberation of two scenes with different boundaries (sst_sediment_reverb: a fluid sediment,
free field; flank_patrol: Lambert bottom and Chapman-Harris surface through Bellhop); `flank_patrol`'s three targets,
predicted margin against the margin measured in the simulated data; a noise-limited budget unchanged; both chain
engines printing the same member, and the term off printing what the picture printed before it existed."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import coverage, flank, live
from gpusonar import reverb_budget as rb
from gpusonar.config import load_scenario_config
from gpusonar.recon.fls import aperture_weights
from gpusonar.sim import find_bellhop, make_pose, run_scenario
from gpusonar.sim.propagation import FreeFieldProvider, ImageSourceProvider
from gpusonar.sonar_equation import SonarEquation, isotropic_noise_gain_db


core = gs.core
ROOT = Path(__file__).resolve().parents[2]
SAND = {"speed": 1600.0, "density": 1.8, "attenuation": 0.5}
FC, B, PSI, ALPHA = 8000.0, 4000.0, 0.1, 0.0005            # Hz, Hz, rad, dB/m
LAMBERT = {"mu_db": -27.0}
ISO = coverage.water([0.0, 100.0], [1500.0, 1500.0])


def g13(ranges, height, mu_db=-27.0, psi=PSI, band=B, alpha=ALPHA, c=1500.0):
    """J&R (G.13), one direct path each way, Lambert: RL - SL = mu sin^2 g Psi (c / 2B) / r^3, less 2 alpha r."""
    r = np.asarray(ranges, dtype=float)
    g = np.arcsin(height / r)
    return 10.0 * np.log10(10.0 ** (mu_db / 10.0) * np.sin(g) ** 2 * psi * c / (2.0 * band) / r**3) - 2.0 * alpha * r


def cpp_level(ranges, items, band=B, psi=PSI, surface=None, max_bounces=1, own=30.0, depth=100.0, bottom=SAND):
    o = coverage.cpp_options({"reverberation": {"items": items, "frequency_hz": FC, "max_bounces": max_bounces}, "bottom": bottom, "surface": surface or {},
                              "rays": {"fan": 1440}, "budget": {"absorption_db_per_km": ALPHA * 1000.0}})
    return np.array(core.coverage_reverberation_db(ISO, depth, own, own, list(2.0 * np.asarray(ranges) / 1500.0), o,
                                                   {"f_centre": FC, "bandwidth": band, "pulse": 1.0}, psi))


def test_one_path_each_way_is_jackson_and_richardson_g13():
    """Lambert bottom 70 m below the sonar, surface paths removed (coefficient 0), 1 to 4 km: both implementations
    within 0.2 dB of (G.13) with the compressed pulse (docs/references/jackson_richardson_2007/03, Sec. 2.9)."""
    ranges = np.array([1000.0, 1500.0, 2200.0, 3000.0, 4000.0])
    want = g13(ranges, 70.0)
    got = cpp_level(ranges, [LAMBERT], surface={"coefficient": 0.0})
    prov = ImageSourceProvider(1500.0, surface_z=0.0, bottom_z=100.0, bottom=SAND, max_order=1, surface_coefficient=0.0)
    ref = rb.level_db(prov, np.array([0.0, 0.0, 30.0]), [(100.0, rb.item_law(LAMBERT, FC, 1500.0)[1])], 2.0 * ranges / 1500.0, B, PSI, ALPHA)
    assert np.max(np.abs(got - want)) < 0.2 and np.max(np.abs(ref - want)) < 0.2, (got - want, ref - want)


@pytest.mark.parametrize("order", [1, 3])
def test_the_coverage_rays_give_the_python_references_level(order):
    """Bottom (Lambert) and surface (Chapman-Harris, 5 m/s) with every path of up to `order` boundary reflections each
    way, isovelocity 100 m over sand: the C++ term (rays, the coverage picture's) against the Python reference
    (image sources through sim/reverb.py's own integral), within 0.3 dB from 1 to 4 km (1 dB at 500 m, near the
    onset, where the two discretise the first returns differently)."""
    items = [LAMBERT, {"model": "chapman_harris", "wind_speed": 5.0}]
    ranges = np.array([500.0, 1000.0, 1500.0, 2200.0, 3000.0, 4000.0])
    got = cpp_level(ranges, items, max_bounces=order)
    prov = ImageSourceProvider(1500.0, surface_z=0.0, bottom_z=100.0, bottom=SAND, max_order=order)
    laws = [(100.0, rb.item_law(items[0], FC, 1500.0)[1]), (0.0, rb.item_law(items[1], FC, 1500.0)[1])]
    ref = rb.level_db(prov, np.array([0.0, 0.0, 30.0]), laws, 2.0 * ranges / 1500.0, B, PSI, ALPHA)
    assert np.max(np.abs(got - ref)[1:]) < 0.3 and abs(got[0] - ref[0]) < 1.0, got - ref


def test_limits_of_the_term():
    """Zero scattering strength is the noise-only budget exactly; with the noise far down the SNR does not move with the
    source level; a CW pulse twice as long has 3 dB more reverberation (both implementations); the scattering laws are
    the simulator's own (a Lambert item, its offset and a named surface law through sim/simulator.py)."""
    ranges = np.array([1000.0, 2000.0, 3000.0])
    eq = SonarEquation(200.0, FC, B, 1.0, 32, 45.0, target_strength_db=-3.0, depth=30.0)
    zero = rb.at_ranges(ImageSourceProvider(1500.0, surface_z=0.0, bottom_z=100.0, bottom=SAND, max_order=1), np.array([0, 0, 30.0]),
                        [(100.0, lambda a, b: 0.0 * a * b)], B, PSI, 1500.0, ranges=ranges)
    with_zero = SonarEquation(**{**eq.__dict__, "reverberation_db": zero})
    assert np.array_equal(with_zero.snr(ranges), eq.snr(ranges)) and eq.table(ranges) == SonarEquation(**eq.__dict__).table(ranges)
    level = rb.at_ranges(ImageSourceProvider(1500.0, surface_z=0.0, bottom_z=100.0, bottom=SAND, max_order=1), np.array([0, 0, 30.0]),
                         [(100.0, rb.item_law(LAMBERT, FC, 1500.0)[1])], B, PSI, 1500.0, ranges=ranges)
    quiet = dict(eq.__dict__, noise_spectrum_db=-60.0, reverberation_db=level)
    a, b = SonarEquation(**quiet).snr(ranges), SonarEquation(**dict(quiet, source_level_db=220.0)).snr(ranges)
    assert np.allclose(a, b, atol=1e-9) and set(SonarEquation(**quiet).limited_by(ranges)) == {"reverberation"}
    loud = SonarEquation(**dict(eq.__dict__, reverberation_db=level))
    assert np.all(loud.snr(ranges) < eq.snr(ranges)) and loud.table(ranges)[0]["limited_by"] == "reverberation"
    # CW: the band is 1 / T; twice the pulse, twice the ring
    prov = ImageSourceProvider(1500.0, surface_z=0.0, bottom_z=100.0, bottom=SAND, max_order=1)
    laws = [(100.0, rb.item_law(LAMBERT, FC, 1500.0)[1])]
    for band in ((100.0, 50.0), (20.0, 10.0)):
        py = [rb.level_db(prov, np.array([0, 0, 30.0]), laws, 2.0 * ranges[1:] / 1500.0, b_, PSI, ALPHA) for b_ in band]
        cpp = [cpp_level(ranges[1:], [LAMBERT], band=b_) for b_ in band]
        assert np.allclose(py[1] - py[0], 10.0 * np.log10(2.0), atol=0.15) and np.allclose(cpp[1] - cpp[0], 10.0 * np.log10(2.0), atol=0.15), (band, py, cpp)
    # the simulator's laws: a Lambert item's offset is in its strength; Chapman-Harris is the acoustics function
    _, law = rb.item_law({"mu_db": -27.0, "offset_db": 3.0}, FC, 1500.0)
    assert law(np.radians(30.0), np.radians(30.0)) == pytest.approx(10.0 ** (-2.4) * 0.25, rel=1e-12)
    from gpusonar import acoustics
    _, law = rb.item_law({"model": "chapman_harris", "wind_speed": 5.0}, FC, 1500.0)
    assert 10.0 * np.log10(law(0.2, 0.2)) == pytest.approx(acoustics.chapman_harris_db(0.2, 5.0, FC), abs=1e-9)
    assert rb.boundary_of({"model": "chapman_harris"}) == "surface" and rb.boundary_of({"mu_db": -27.0}) == "bottom"
    with pytest.raises(ValueError, match="volume"):
        rb.strength_tables([{"model": "volume", "layer": [10.0, 20.0], "sv_db": -70.0}], FC, 1500.0)


def test_beam_integral_of_a_line_and_of_the_record():
    """Psi of an unweighted line at half a wavelength, broadside, both implementations; a Hann window widens it by the
    window's loss against uncorrelated noise (1.76 dB for 32 elements, the same factor that lowers the noise gain);
    the record's (C++, what the chain uses) is the array's at broadside."""
    from gpusonar.sonar import SonarModel
    rec = SonarModel.load("flank_32").make_record(make_pose())
    pos = rec.rx_positions()
    psi_py = rb.beam_integral(pos, FC, 1500.0, np.pi / 2.0)
    psi_cpp = core.coverage_beam_integral([float(v) for v in pos[:, 0]], [float(v) for v in pos[:, 1]], FC, 1500.0, np.pi / 2.0)
    assert psi_cpp == pytest.approx(psi_py, rel=1e-6)
    assert core.coverage_beam_integral_of(rec) == pytest.approx(rb.beam_integral(pos, FC, rec.environment.sound_speed, np.pi / 2.0), rel=1e-6)
    w = aperture_weights(rec, "hann").astype(float)
    widen = 10.0 * np.log10(rb.beam_integral(pos, FC, 1500.0, np.pi / 2.0, weights=w) / psi_py)
    assert widen == pytest.approx(10.0 * np.log10(live._window_loss(32, "hann")), abs=0.3)


# --- against simulated reverberation ---------------------------------------------------------------------------------

def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_level_against_the_simulated_fluid_sediment_reverberation():
    """sst_sediment_reverb (omnidirectional, free field, `medium_sand` by the fluid-sediment model, no absorption) with
    an LFM pulse of 2 kHz: the target's compressed peak over the mean compressed reverberation at its delay, measured
    on 8 pings x 8 channels, against the prediction TS - 40 log r - RL (re SL), RL by the Python reference and by the
    C++ term, at 50, 80 and 120 m. Measured 27.83, 30.67, 34.24 dB; predicted 27.28, 30.85, 34.30 (Python) and 28.14,
    30.85, 34.30 (C++). Held: rms difference 0.6 dB, each within 1 dB (8 x 8 looks of exponential reverberation over a
    +-4 ms window: about 0.2 dB of scatter; the rest is the compressed pulse's shape, a Tukey-tapered sweep, against
    the sinc of its band)."""
    sst = _tool("sst_validation")
    parts = rb.scenario_parts("sst_sediment_reverb")
    c = float(parts["environment"]["sound_speed"])
    fc, band, duration, ts = 20000.0, 2000.0, 0.02, 20.0
    wave = {"type": "lfm", "f_start": fc - 0.5 * band, "f_stop": fc + 0.5 * band, "duration": duration, "window_alpha": 0.1}
    rows = []
    for slant in (50.0, 80.0, 120.0):
        sonar = {"transmit": {"waveform": wave}, "receive": {"window": {"range": [slant - 30.0, slant + 30.0]}}}
        target = [{"type": "points", "label": "target", "positions": [[0.0, slant, 0.0]], "target_strength_db": ts}]
        rev = sst._quiet_run(sst.SEDIMENT, {"sonar_overrides": sonar, "pings": {"count": 8}})
        echo = sst._quiet_run(sst.SEDIMENT, {"sonar_overrides": sonar, "pings": {"count": 1}, "reverberation": [], "scene": target})
        z_rev, t0, fs = sst._compressed(rev)
        z_echo, _, _ = sst._compressed(echo)
        t = t0 + np.arange(z_rev.shape[-1]) / fs
        t_target = 2.0 * slant / c
        near = np.abs(t - t_target) < 0.004
        reverberation = float((np.abs(z_rev[:, :, near]) ** 2).mean())
        peak = float(np.mean([sst._peak_near(z_echo[0, ch], t0, fs, t_target, 3.0 / fs)[0] for ch in range(z_echo.shape[1])]))
        measured = 10.0 * np.log10(peak**2 / reverberation)
        law = rb.item_law(parts["items"][0], fc, c)[1]
        py = float(rb.level_db(FreeFieldProvider(c), np.zeros(3), [(20.0, law)], [t_target], band, 2.0 * np.pi, 0.0, step=2.5e-4)[0])
        o = coverage.cpp_options({"reverberation": {"items": [dict(parts["items"][0], boundary="bottom")], "frequency_hz": fc, "max_bounces": 1},
                                  "surface": {"coefficient": 0.0}, "rays": {"fan": 2880}})
        cpp = float(core.coverage_reverberation_db(coverage.water([0.0, 20.0], [c, c]), 20.0, 1e-3, 1e-3, [t_target], o,
                                                   {"f_centre": fc, "bandwidth": band, "pulse": duration}, 2.0 * np.pi)[0])
        rows.append((measured, ts - 40.0 * np.log10(slant) - py, ts - 40.0 * np.log10(slant) - cpp))
    d = np.array([[m - p, m - q] for m, p, q in rows])
    print("sst_sediment_reverb measured, predicted (py, cpp) dB:", [tuple(round(v, 2) for v in r) for r in rows])
    assert np.sqrt(np.mean(d**2)) < 0.6 and np.max(np.abs(d)) < 1.0, d


def test_cw_reverberation_goes_through_the_triangle_not_the_sinc():
    """C(e) (sonar-w2w3's follow-up). sst_sediment_reverb as shipped is a CW pulse (20 kHz, 2 ms) over medium sand 20 m
    below the sonar: the mean compressed reverberation of 16 pings x 8 channels over a +20 dB target's peak at 50 m,
    against the prediction with the matched filter's own triangle (1 - |u| / T)^2 and with the sinc of the band 1 / T
    (what stood for it), from just before the first bottom return (20 m) to 50 m. Measured: 2.1, 5.1, 4.6, 2.8, -2.5, -7.3,
    -11.1, -16.2, -23.4 dB at 19.5, 20, 20.4, 20.75, 21.5, 23, 26, 35, 50 m; the triangle within 0.7 dB of each, the
    sinc 0.9 to 1.9 dB high from 20 m on (1.5 on average) (its integral is T, the triangle's 2 T / 3: 1.76 dB). The C++ term (CoverageSonar
    cw) gives the Python reference's triangle within 0.2 dB from 23 m on; nearer, the first return comes in steeper than
    its fan's 75 degrees and is missing (-12 dB at 20 m; nothing the coverage picture's kilometres need)."""
    sst = _tool("sst_validation")
    parts = rb.scenario_parts("sst_sediment_reverb")
    c, fc, T, slant, ts = 1500.0, 20000.0, 0.002, 50.0, 20.0
    win = {"receive": {"window": {"range": [5.0, 80.0]}}}
    rev = sst._quiet_run(sst.SEDIMENT, {"sonar_overrides": win, "pings": {"count": 16}})
    echo = sst._quiet_run(sst.SEDIMENT, {"sonar_overrides": win, "pings": {"count": 1}, "reverberation": [],
                                         "scene": [{"type": "points", "label": "t", "positions": [[0.0, slant, 0.0]], "target_strength_db": ts}]})
    z, t0, fs = sst._compressed(rev)
    ze, _, _ = sst._compressed(echo)
    t = t0 + np.arange(z.shape[-1]) / fs
    peak = float(np.mean([sst._peak_near(ze[0, ch], t0, fs, 2.0 * slant / c, 3.0 / fs)[0] for ch in range(ze.shape[1])]))
    r_eval = np.array([19.5, 20.0, 20.4, 20.75, 21.5, 23.0, 26.0, 35.0, 50.0])
    delays, echo_db = 2.0 * r_eval / c, ts - 40.0 * np.log10(slant)
    law = rb.item_law(parts["items"][0], fc, c)[1]
    tri = rb.level_db(FreeFieldProvider(c), np.zeros(3), [(20.0, law)], delays, 1.0 / T, 2.0 * np.pi, 0.0, step=2.5e-5, cw_length=T) - echo_db
    snc = rb.level_db(FreeFieldProvider(c), np.zeros(3), [(20.0, law)], delays, 1.0 / T, 2.0 * np.pi, 0.0, step=2.5e-5) - echo_db
    power = np.abs(z) ** 2
    measured = np.array([10.0 * np.log10(power[:, :, np.abs(t - d) <= 0.5 / fs + 1e-12].mean() / peak**2) for d in delays])
    o = coverage.cpp_options({"reverberation": {"items": [dict(parts["items"][0], boundary="bottom")], "frequency_hz": fc, "max_bounces": 1},
                              "surface": {"coefficient": 0.0}, "rays": {"fan": 2880}})
    o["reverb_rings"] = 4000
    cpp = np.array(core.coverage_reverberation_db(coverage.water([0.0, 20.0], [c, c]), 20.0, 1e-3, 1e-3, list(delays), o,
                                                  {"f_centre": fc, "bandwidth": 1.0 / T, "pulse": T, "cw": True}, 2.0 * np.pi)) - echo_db
    print("CW onset, measured / triangle / sinc / C++ (dB):", [tuple(round(float(v), 2) for v in row) for row in zip(measured, tri, snc, cpp)])
    assert np.max(np.abs(measured - tri)) < 0.7, measured - tri
    assert np.all(snc[1:] - measured[1:] > 0.8) and np.mean(snc[1:] - measured[1:]) > 1.4, snc - measured
    assert np.max(np.abs(cpp - tri)[5:]) < 0.2, cpp - tri
    assert core.compressed_power_average(0.0, 0.0, 1.0 / T, T) == pytest.approx(1.0) and core.compressed_power_average(1.01 * T, 0.0, 1.0 / T, T) == 0.0
    assert rb.triangle2_average(0.3 * T, 0.2 * T, T) == pytest.approx(core.compressed_power_average(0.3 * T, 0.2 * T, 1.0 / T, T), rel=1e-12)


def test_the_bounce_count_is_found_by_convergence():
    """C(c). The paths' boundary reflections are added until those above the count change no level by more than 0.1 dB
    (coverage.hpp reverb_max_bounces 0). Lambert -27 dB and Chapman-Harris over sand, 4-6 kHz, 0.8 to 12.7 km, the array at
    mid-depth: the converged levels are a 160-reflection run's within 0.1 dB in 30 and 100 m of water, where the fixed rule
    it replaces (0.4 x range / depth, at most 40) was 1.2 dB low in 30 m (0.07 dB in 100 m)."""
    r = np.linspace(800.0, 12700.0, 48)

    def level(depth, bounces=None):
        block = {"items": [LAMBERT, {"model": "chapman_harris", "wind_speed": 5.0}], "frequency_hz": 5000.0}
        if bounces:
            block["max_bounces"] = bounces
        o = coverage.cpp_options({"reverberation": block, "bottom": {"speed": 1650.0, "density": 1.9, "attenuation": 0.8}, "surface": {"rms_height_m": 0.134},
                                  "budget": {"absorption_db_per_km": 0.32}})
        return np.array(core.coverage_reverberation_db(coverage.water([0.0, depth], [1500.0, 1500.0]), depth, 0.5 * depth, 0.5 * depth, list(2.0 * r / 1500.0), o,
                                                       {"f_centre": 5000.0, "bandwidth": 2000.0, "pulse": 1.0}, 0.05))
    for depth, old in ((30.0, 40), (100.0, 40)):
        ref, got, rule = level(depth, 160), level(depth), level(depth, old)
        assert np.max(np.abs(got - ref)) < 0.1, (depth, got - ref)
        if depth == 30.0:
            assert np.max(ref - rule) > 1.0, ref - rule


def test_off_broadside_the_beam_integral_is_the_patterns_own():
    """C(d). Psi of a line steered off broadside, the C++ term's (`coverage_beam_integral`, and the record's with
    `steer_from_boresight`, what the chain uses with coverage.reverberation.steer_deg) against a direct numerical integral
    of the array's |AF|^2 over azimuth on a fine grid, at 0, 15, 30, 45 and 60 degrees: within 1e-4. It widens as about
    1 / cos (32 hydrophones at half a wavelength: +0.16, +0.63, +1.56, +3.09 dB; 1 / cos: +0.15, +0.62, +1.51, +3.01)."""
    from gpusonar.sonar import SonarModel
    rec = SonarModel.load("flank_32").make_record(make_pose())
    pos = rec.rx_positions()
    c = rec.environment.sound_speed
    w = rec.transmit.waveform
    f = 0.5 * (w.f_start + w.f_stop)
    k = 2.0 * np.pi * f / c
    phi = 2.0 * np.pi * (np.arange(200000) + 0.5) / 200000
    broadside = None
    for deg in (0.0, 15.0, 30.0, 45.0, 60.0):
        steer = np.pi / 2.0 - np.radians(deg)
        af = np.abs(np.exp(1j * k * (np.outer(np.cos(phi) - np.cos(steer), pos[:, 0]) + np.outer(np.sin(phi) - np.sin(steer), pos[:, 1]))).sum(axis=1)) ** 2 / len(pos) ** 2
        direct = float(af.mean() * 2.0 * np.pi)
        got = core.coverage_beam_integral([float(v) for v in pos[:, 0]], [float(v) for v in pos[:, 1]], f, c, steer)
        assert got == pytest.approx(direct, rel=1e-4), deg
        assert core.coverage_beam_integral_of(rec, np.radians(deg)) == pytest.approx(direct, rel=1e-4), deg
        assert rb.beam_integral(pos, f, c, steer) == pytest.approx(direct, rel=1e-4), deg
        broadside = direct if broadside is None else broadside
        assert 10.0 * np.log10(direct / broadside) == pytest.approx(-10.0 * np.log10(np.cos(np.radians(deg))), abs=0.12), deg
    o = coverage.cpp_options({"reverberation": {"items": [LAMBERT], "frequency_hz": FC, "steer_deg": 60.0}})
    assert o["reverb_steer"] == pytest.approx(np.radians(60.0))


# --- flank_patrol: the issue's case (D-198), and a noise-limited flank scene -----------------------------------------

FLANK_TARGETS = {"submarine": ((-600.0, 2900.0, 60.0), 16.0), "small_vehicle": ((900.0, 2000.0, 40.0), -3.0), "wreck": ((1500.0, 3270.0, 98.0), 12.0)}


def flank_prediction(name, point, ts, items=None):
    """The coverage picture's figures (C++, both engines' class) for a target of `ts` at `point` in a flank scenario as
    the gallery images it: one ping, Hann-weighted beam on the target's bearing (its gain against isotropic noise and its
    beam integral), the scene's source level, noise, absorption, water, bottom and reverberation items; the projector
    0.5 m below the array (flank_32's lever arm). Rays: the coverage defaults. {sinr_db, snr_noise_db, r_over_n_db}."""
    parts = rb.scenario_parts(name)
    env, sonar = parts["environment"], parts["sonar"]
    c, fc = float(env["sound_speed"]), parts["frequency"]
    own = float(parts["config"]["platform"]["position"][2])
    rec = sonar.make_record(make_pose((0.0, 0.0, own)))
    pos, w = rec.rx_positions(), aperture_weights(rec, "hann").astype(float)
    r_h, az = np.hypot(point[0], point[1]), np.pi / 2.0 - np.arctan2(point[0], point[1])
    u = np.array([np.cos(az), np.sin(az), 0.0])
    half = coverage.halfspace(env["bottom"]) if isinstance(env.get("bottom"), str) else dict(env["bottom"])
    ambient = parts["config"]["receiver"]["ambient_noise"]
    if "level_db" in ambient:
        noise_db = float(ambient["level_db"])
    else:                                                                       # the Wenz fit at the band centre, as the simulator draws it
        from gpusonar.sim import noise as noise_models
        noise_db = float(noise_models.wenz_spectrum_db(fc, float(ambient.get("wind_speed", 5.0)), float(ambient.get("shipping", 0.5))))
    cfg = {"budget": {"source_level_db": float(sonar.source_level_db_rms), "noise_spectrum_db": noise_db,
                      "array_gain_db": isotropic_noise_gain_db(pos, fc, c, u, weights=w), "absorption_db_per_km": parts["absorption_db_per_m"] * 1000.0},
           "bottom": {k: half[k] for k in ("speed", "density", "attenuation")},
           "reverberation": {"items": parts["items"] if items is None else items, "frequency_hz": fc}}
    depth = float(env["site_water_depth"])
    prof = env.get("profile") or {"depth": [0.0, depth], "speed": [c, c]}
    f = coverage.field(coverage.water(prof["depth"], prof["speed"]), depth, own, [point[2]], [r_h], cfg,
                       {"f_centre": fc, "bandwidth": sonar.bandwidth, "pulse": float(sonar.waveform_config["duration"]),
                        "beam_integral": rb.beam_integral(pos, fc, c, az, weights=w)},
                       projector_below_m=float(sonar.config["transmit"].get("lever_arm", [0.0, 0.0, 0.0])[2]))
    s, n = float(f["snr_0db_target_db"][0, 0]) + ts, float(f["snr_noise_only_0db_target_db"][0, 0]) + ts
    return {"sinr_db": s, "snr_noise_db": n, "r_over_n_db": float(10.0 * np.log10(max(10.0 ** ((n - s) / 10.0) - 1.0, 1e-30)))}


def flank_map(name, seed=17, scene=True, reverberation=True, noise=True, edit=None):
    """One ping of a flank scenario at full size, beamformed with Hann weights as tools/gallery.py does: fine-sample power.
    `edit`: a function that changes the loaded configuration first."""
    cfg = load_scenario_config(name)
    if edit is not None:
        cfg = edit(cfg)
    cfg["pings"]["count"], cfg["seed"] = 1, seed
    if not scene:
        cfg["scene"] = []
    if not reverberation:
        cfg["reverberation"] = []
    if not noise:
        cfg["receiver"].pop("ambient_noise", None)
    rec = run_scenario(cfg, backend=core.Backend.Host).records[0]
    bm = flank.farfield_beamform(rec, np.radians(np.arange(-40.0, 40.01, 0.5)), window="hann").valid()
    return np.abs(bm.data) ** 2, bm.ranges, bm.bearings


def _ring(ranges, centre):
    return (np.abs(ranges - centre) > 75.0) & (np.abs(ranges - centre) < 315.0)        # the gallery's CFAR training cells, m


def _stationary_points(cfg):
    """flank_patrol's targets as stationary points at their full strength: the submarine one +16 dB point at its centre, the
    small vehicle stopped (what the coverage picture models: a class has neither a speed nor an extent)."""
    for item in cfg["scene"]:
        if item["label"] == "submarine":
            item.update(positions=[[-600.0, 2900.0, 60.0]], target_strength_db=16.0, velocity=[0.0, 0.0, 0.0])
        elif item.get("velocity") is not None:
            item["velocity"] = [0.0, 0.0, 0.0]
    return cfg


@pytest.mark.skipif(find_bellhop() is None, reason="Bellhop not installed (GPUSONAR_BELLHOP)")
def test_flank_patrol_margins_predicted_against_the_simulated_data():
    """The issue's acceptance (sonar-w2w3). flank_patrol at full size (Bellhop, 32 arrivals), seed 17, one ping: each
    target's clean echo (no reverberation, no noise), the target-free background (reverberation and noise), and noise
    alone, beamformed as the gallery does. Measured on the target's beam: the clean echo's peak sample over the mean
    background in the gallery's training ring (75 to 315 m from the peak), and reverberation over noise there.
    Predicted: the coverage picture with the scene's own inputs, nothing tuned per target.

    Measured / predicted (dB): reverberation over noise 19.7 / 19.9 (submarine, 3.0 km), 24.5 / 24.3 (small vehicle,
    2.2 km), 16.7 / 17.4 (wreck, 3.6 km). Peak over background, the scene as shipped: 26.6 / 28.8, 6.9 / 10.3, 29.4 /
    28.6 (before C(a), sonar-w2w3's follow-up: predicted 26.5, 10.0, 24.7, worst 4.7 dB, rms 3.3; noise-only 46.5, 34.3,
    42.2). The same targets as stationary points (the submarine one +16 dB point, the vehicle stopped; what the coverage
    picture models): 28.3 / 28.8, 10.2 / 10.3, 29.4 / 28.6 (worst 0.9 dB, rms 0.6). Held (tightened by C(a), from 5 dB
    each and 3.5 dB rms): the reverberation within 1 dB of the data; the stationary points each within 1 dB, rms 0.8 dB;
    the scene as shipped each within 4 dB, rms 3.0 dB. What C(a) changed: the four images about the target (one surface and one bottom
    reflection at most) add as pressures (coverage.hpp `echo_coherent`): the wreck 2 m above the bed is 4.7 dB under the
    data when its bottom-reflected paths add as power. More arrivals do not matter: 6 to 1000 arrivals and 3 to 40 bounces move the
    three predictions by under 0.15 dB. What is left in the scene as shipped is the targets, not the budget: the vehicle's
    2 m/s opening costs its 1 s LFM 3.4 dB of peak (Doppler mismatch), the submarine's eight highlights 1.7 dB against
    one point; neither is in a class of the coverage picture (follow-up)."""
    clean, r, b = flank_map("flank_patrol", reverberation=False, noise=False)
    still = flank_map("flank_patrol", reverberation=False, noise=False, edit=_stationary_points)[0]
    bg, _, _ = flank_map("flank_patrol", scene=False)
    noise, _, _ = flank_map("flank_patrol", scene=False, reverberation=False)
    rows = {}
    for name, (p, ts) in FLANK_TARGETS.items():
        ib = int(np.argmin(np.abs(b - np.arctan2(p[0], p[1]))))
        gate = np.abs(r - np.hypot(p[0], p[1])) < 150.0
        k = np.flatnonzero(gate)[np.argmax(clean[ib, gate])]
        ring = _ring(r, r[k])
        n = noise[ib, ring].mean()
        measured = {"margin_db": 10.0 * np.log10(clean[ib, k] / bg[ib, ring].mean()), "margin_stationary_db": 10.0 * np.log10(still[ib, gate].max() / bg[ib, ring].mean()),
                    "r_over_n_db": 10.0 * np.log10((bg[ib, ring].mean() - n) / n)}
        rows[name] = (measured, flank_prediction("flank_patrol", p, ts))
    print("\nflank_patrol", {k: {"measured": {a: round(float(v), 2) for a, v in m.items()}, "predicted": {a: round(v, 2) for a, v in q.items()}} for k, (m, q) in rows.items()})
    for name, (m, q) in rows.items():
        assert abs(m["r_over_n_db"] - q["r_over_n_db"]) < 1.0, name
        assert abs(m["margin_stationary_db"] - q["sinr_db"]) < 1.0, name
        assert abs(m["margin_db"] - q["sinr_db"]) < 4.0, name
        assert q["snr_noise_db"] - q["sinr_db"] > 10.0, name                       # reverberation-limited, all three
    assert np.sqrt(np.mean([(m["margin_stationary_db"] - q["sinr_db"]) ** 2 for m, q in rows.values()])) < 0.8
    assert np.sqrt(np.mean([(m["margin_db"] - q["sinr_db"]) ** 2 for m, q in rows.values()])) < 3.0


def _measured_r_over_n(name, range_m, bearing=0.0):
    """Reverberation over noise (dB) on the beam at `bearing` (rad) in the ring about `range_m`: the scene's target-free
    reverberation alone against its noise alone, one ping each."""
    rev, r, b = flank_map(name, scene=False, noise=False)
    noise, _, _ = flank_map(name, scene=False, reverberation=False)
    ib, ring = int(np.argmin(np.abs(b - bearing))), _ring(r, range_m)
    return float(10.0 * np.log10(rev[ib, ring].mean() / noise[ib, ring].mean()))


@pytest.mark.skipif(find_bellhop() is None, reason="Bellhop not installed (GPUSONAR_BELLHOP)")
def test_a_noise_limited_scene_is_unchanged_and_the_reverberation_is_the_datas_at_long_range():
    """popoto_env_deep_silt_1000m (1000 m of refracting water over silt, Lambert -30 dB, Chapman-Harris at 5 m/s, the
    ship at 10 km): reverberation over noise measured -15.1 dB, predicted -15.4, so the budget with the term is the
    noise-only one within 0.2 dB (0.13 dB). flank_target_10km (100 m over sand, Lambert -27 dB): measured -6.5, predicted
    -5.2; the term costs that target 1.1 dB. Held: each within 1.5 dB of the data."""
    deep = flank_prediction("popoto_env_deep_silt_1000m", (800.0, 10000.0, 4.0), 18.0)
    assert abs(deep["sinr_db"] - deep["snr_noise_db"]) < 0.2, deep
    assert abs(_measured_r_over_n("popoto_env_deep_silt_1000m", 10000.0) - deep["r_over_n_db"]) < 1.5
    flat = flank_prediction("flank_target_10km", (0.0, 10000.0, 50.0), 10.0)
    measured = _measured_r_over_n("flank_target_10km", 10000.0)
    print("reverberation over noise at 10 km: deep silt", deep["r_over_n_db"], "flank", measured, flat["r_over_n_db"])
    assert abs(measured - flat["r_over_n_db"]) < 1.5 and flat["snr_noise_db"] - flat["sinr_db"] < 1.5
