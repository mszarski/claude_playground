"""Milestone 10 (part): flank-array beamforming and the sonar equation.

Acceptance covered here: the far-field beamformer and backprojection agree; beam width
and grating lobes match theory for `flank_32` and both `popoto_8` layouts; the detection
range in simulation matches the sonar-equation tool within 3 dB.
"""

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import flank, recon
from gpusonar.metrics import width_at_level
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel
from gpusonar.sonar_equation import SonarEquation, noise_rms_volts

C = 1500.0


def flank_ping(range_m, bearing_deg, name="flank_32", duration=0.05, noise_rms=0.0, seed=0,
               half_window=60.0, ts_db=0.0, overrides=None, **kw):
    over = {"receive": {"window": {"range": [range_m - half_window, range_m + half_window]}},
            "transmit": {"waveform": {"duration": duration}}}
    if overrides:
        over = gs.config.deep_merge(over, overrides)
    sonar = SonarModel.load(name, over)
    b = np.radians(bearing_deg)
    # Bearing from the boresight (+y) towards the array axis (+x).
    target = range_m * np.array([np.sin(b), np.cos(b), 0.0])
    rec = simulate_ping(sonar, make_pose(), Scene().add_points([target], target_strength_db=ts_db),
                        environment={"sound_speed": C, "temperature": 10.0, "salinity": 35.0},
                        noise_rms=noise_rms, adc=False, rng=np.random.default_rng(seed), **kw)
    return sonar, rec, target


# --- far field against backprojection ------------------------------------------------------------

@pytest.mark.parametrize("bearing", [0.0, 17.0, -35.0])
def test_farfield_beamformer_agrees_with_backprojection_at_long_range(bearing):
    sonar, rec, target = flank_ping(5000.0, bearing, absorption_model="none")
    bearings = np.radians(bearing + np.linspace(-6, 6, 49))
    ranges = 5000.0 + np.linspace(-1.0, 1.0, 21)
    bp = flank.backprojection_map(rec, ranges, bearings)
    ff = flank.farfield_beamform(rec, bearings)
    # Compare on the same pixels: the far-field output at the two-way delay to the array
    # centre, with the carrier phase backprojection restores.
    tx = np.array(rec.tx_position())
    centre = rec.rx_positions().mean(axis=0)
    worst = 0.0
    peak = np.abs(bp.image).max()
    for ib in range(len(bearings)):
        pts = bp.points[:, ib, :]
        tau = (np.linalg.norm(pts - tx, axis=1) + np.linalg.norm(pts - centre, axis=1)) / C
        far = flank.sample_beam_map(ff, ib, tau) * np.exp(2j * np.pi * rec.samples.center_frequency * tau)
        worst = max(worst, np.abs(far - bp.image[:, ib]).max() / peak)
    # 5 km is 50 Fraunhofer distances for this 3 m array at 8 kHz: the two must coincide.
    assert worst < 0.02, worst
    ir, ib = np.unravel_index(np.argmax(np.abs(bp.image)), bp.image.shape)
    assert np.degrees(bearings[ib]) == pytest.approx(bearing, abs=0.3)


def test_farfield_beamformer_is_wrong_in_the_near_field_and_backprojection_is_not():
    """Well inside the Fraunhofer distance (2 L^2 / lambda = 96 m) plane-wave steering
    defocuses, while backprojection still gives the diffraction-limited beam."""
    R = 8.0
    sonar, rec, target = flank_ping(R, 0.0, absorption_model="none", half_window=5.0)
    bearings = np.radians(np.linspace(-30, 30, 241))
    bp = flank.backprojection_map(rec, [R], bearings)
    ff = flank.farfield_beamform(rec, bearings)
    tx = np.array(rec.tx_position())
    centre = rec.rx_positions().mean(axis=0)
    tau = (np.linalg.norm(target - tx) + np.linalg.norm(target - centre)) / C
    far_row = np.array([abs(flank.sample_beam_map(ff, ib, [tau])[0]) for ib in range(len(bearings))])
    bp_row = np.abs(bp.image[0])
    theory = np.degrees(flank.beamwidth_theory(32 * 0.094, 8e3))
    assert np.degrees(width_at_level(bearings, bp_row, -3.0)) == pytest.approx(theory, rel=0.10)
    assert far_row.max() < 0.6 * bp_row.max()                    # half the coherent gain is gone
    assert width_at_level(bearings, far_row, -3.0) > 3.0 * width_at_level(bearings, bp_row, -3.0)


# --- beam width and grating lobes against theory -------------------------------------------------

def measured_pattern(name, overrides, bearing_span_deg, n=721, range_m=3000.0):
    sonar, rec, target = flank_ping(range_m, 0.0, name=name, absorption_model="none", overrides=overrides)
    bearings = np.radians(np.linspace(-bearing_span_deg, bearing_span_deg, n))
    ff = flank.farfield_beamform(rec, bearings)
    tx = np.array(rec.tx_position())
    tau = (np.linalg.norm(target - tx) + np.linalg.norm(target - rec.rx_positions().mean(axis=0))) / C
    row = np.array([abs(flank.sample_beam_map(ff, ib, [tau])[0]) for ib in range(len(bearings))])
    return sonar, bearings, row / row.max()


def test_flank_32_beam_width_and_no_grating_lobes():
    sonar, bearings, row = measured_pattern("flank_32", None, 80.0)
    theory = flank.beamwidth_theory(32 * 0.094, sonar.center_frequency)        # 3.17 degrees
    assert width_at_level(bearings, row, -3.0) == pytest.approx(theory, rel=0.10)
    # Pitch 9.4 cm against a 15 cm wavelength at the top of the band: no grating lobes
    # for a broadside beam. The highest lobe outside the main lobe is an ordinary sidelobe.
    away = np.abs(np.degrees(bearings)) > 6.0
    assert 20 * np.log10(row[away].max()) < -12.0
    rep = flank.ambiguity_report(sonar.positions[:, 0], 6e3, 10e3)
    assert rep["narrowband_worst_lobe_db"] == pytest.approx(-13.3, abs=0.5)
    assert rep["grating_free_frequency"] == pytest.approx(C / (2 * 0.094))


@pytest.mark.parametrize("layout", ["dense", "sparse"])
def test_popoto_8_layouts_match_their_predicted_patterns(layout):
    over = {"array": {"layout_choice": layout}}
    sonar, bearings, row = measured_pattern("popoto_8", over, 85.0, n=1701, range_m=1500.0)
    x = sonar.positions[:, 0]
    rep = flank.ambiguity_report(x, 20e3, 35e3)
    # Beam width: measured (wideband pulse) against the narrowband pattern at band centre.
    measured_bw = np.degrees(width_at_level(bearings, row, -3.0))
    assert measured_bw == pytest.approx(rep["beamwidth_deg_center"], rel=0.10)
    if layout == "dense":
        # A filled 8-element array at half-wavelength pitch (at 27.5 kHz).
        assert measured_bw == pytest.approx(np.degrees(flank.beamwidth_theory(8 * 0.027, 27.5e3)), rel=0.10)
        assert rep["narrowband_worst_lobe_db"] < -10.0          # sidelobes only, no ambiguity
    else:
        # 3 m of aperture from 8 elements: a far narrower beam, paid for with ambiguities.
        assert measured_bw < 1.0
        assert rep["narrowband_worst_lobe_db"] > -4.0           # single-frequency lobes nearly full height
        assert rep["wideband_lobe_db"] < rep["narrowband_worst_lobe_db"] - 3.0   # the band smears them
        # The measured wideband pattern shows the same ambiguity level the report predicts.
        away = np.abs(np.degrees(bearings)) > 3.0
        measured_lobe = 20 * np.log10(row[away].max())
        assert measured_lobe == pytest.approx(rep["wideband_lobe_db"], abs=3.0)


def test_array_pattern_of_a_uniform_array_has_grating_lobes_where_expected():
    x = (np.arange(8) - 3.5) * 0.3                 # 0.3 m pitch
    f = 10e3                                       # wavelength 0.15 m: pitch = 2 lambda
    u = np.linspace(-1, 1, 4001)
    p = flank.array_pattern(x, f, u)
    for m in (-1, 1):                              # grating lobes at sin(theta) = m lambda / d
        assert p[np.argmin(np.abs(u - m * 0.5))] == pytest.approx(1.0, abs=1e-3)


# --- transmit blanking ---------------------------------------------------------------------------

def test_transmit_blanking_zeroes_the_pulse_interval():
    sonar = SonarModel.load("popoto_8", {"receive": {"window": {"range": [0.0, 150.0]}},     # starts at transmit
                                          "transmit": {"waveform": {"duration": 0.05}}})
    rec = simulate_ping(sonar, make_pose(), Scene(), noise_rms=1e-3, adc=False, rng=np.random.default_rng(0))
    flank.blank_transmit(rec, guard=0.01, taper=0.005)
    x = rec.samples.array
    fs = rec.samples.sample_rate
    n_blank = int((0.05 + 0.01) * fs)
    assert np.all(x[:, :n_blank] == 0)
    assert x[:, n_blank + int(0.006 * fs):].std() == pytest.approx(1e-3, rel=0.05)
    assert 0 < np.abs(x[:, n_blank + int(0.002 * fs)]).mean() < 0.8e-3      # inside the taper


# --- the sonar equation against simulation -------------------------------------------------------

NOISE_SPECTRUM = 50.0          # dB re 1 uPa^2/Hz: a moderate sea at 8 kHz


def simulated_snr_db(range_m, seed):
    """Output SNR at the target, measured: beamform, then peak power over mean noise power."""
    duration = 1.0
    sens = -180.0
    sonar_cfg = SonarModel.load("flank_32")
    rms = noise_rms_volts(NOISE_SPECTRUM, sens, sonar_cfg.sample_rate)
    sonar, rec, target = flank_ping(range_m, 0.0, duration=duration, noise_rms=rms, seed=seed,
                                    half_window=300.0, ts_db=10.0)
    quiet = rec.copy()                      # same ping without the target: the noise reference
    bearings = np.radians(np.linspace(-1.0, 1.0, 5))
    ff = flank.farfield_beamform(rec, bearings)
    tx = np.array(rec.tx_position())
    tau = (np.linalg.norm(target - tx) + np.linalg.norm(target - rec.rx_positions().mean(axis=0))) / C
    row = np.abs(ff.data[2])
    i = int(round((tau - ff.times[0]) / (ff.times[1] - ff.times[0])))
    peak = row[i - 3:i + 4].max()
    far = np.r_[0:i - 2000, i + 2000:len(row)]
    noise_power = np.mean(row[far] ** 2)
    return 10 * np.log10(peak**2 / noise_power)


@pytest.mark.parametrize("range_m", [1000.0, 5000.0, 10000.0])
def test_simulated_snr_matches_the_sonar_equation(range_m):
    sonar = SonarModel.load("flank_32")
    eq = SonarEquation.from_sonar(sonar, NOISE_SPECTRUM, target_strength_db=10.0,
                                  temperature=10.0, salinity=35.0)
    predicted = eq.snr(range_m)
    measured = np.mean([simulated_snr_db(range_m, seed) for seed in (1, 2)])
    assert measured == pytest.approx(predicted, abs=3.0), (measured, predicted)


def test_detection_range_follows_from_the_snr_curve():
    sonar = SonarModel.load("flank_32")
    eq = SonarEquation.from_sonar(sonar, NOISE_SPECTRUM, target_strength_db=10.0)
    r = eq.detection_range(13.0)
    assert eq.snr(r) == pytest.approx(13.0, abs=0.01)
    assert eq.snr(0.5 * r) > 13.0 > eq.snr(2.0 * r)
    # 10 dB more source level buys range; at these ranges absorption (about 0.7 dB/km at
    # 8 kHz) limits the gain to well under the factor 1.78 spherical spreading would give.
    louder = SonarEquation.from_sonar(sonar, NOISE_SPECTRUM, target_strength_db=10.0)
    louder.source_level_db += 10.0
    assert 1.05 * r < louder.detection_range(13.0) < 1.78 * r
    # Bookkeeping identities.
    assert eq.processing_gain() == pytest.approx(10 * np.log10(4000.0 * 1.0))
    assert eq.array_gain() == pytest.approx(10 * np.log10(32))
    assert eq.table([1000.0])[0]["tl_db"] == pytest.approx(60.0 + eq.absorption_db_per_m() * 1000.0)
    with pytest.raises(ValueError, match="uncalibrated"):
        SonarEquation.from_sonar(SonarModel.load("l12_5d"), NOISE_SPECTRUM)


# --- the plan's link budgets ----------------------------------------------------------------------

@pytest.mark.parametrize("range_m, source_level, target_db, plan_snr", [
    (100.0, 170.0, 0.0, 60.0), (1000.0, 180.0, 0.0, 29.0), (10000.0, 200.0, 0.0, -3.0),
    (10000.0, 210.0, 0.0, 7.0), (10000.0, 200.0, 20.0, 17.0), (1000.0, 180.0, 10.0, 39.0),
])
def test_sonar_equation_reproduces_the_plans_phase_6_link_budget(range_m, source_level, target_db, plan_snr):
    """docs/plan, Phase 6: spherical spreading, 12 C seawater, noise 45 dB re 1 uPa^2/Hz at
    8 kHz, 1 s chirp, 15 dB of array gain. (The plan used the Ainslie-McColm formula.)"""
    eq = SonarEquation(source_level_db=source_level, center_frequency=8e3, bandwidth=4e3,
                       pulse_duration=1.0, n_elements=32, noise_spectrum_db=45.0,
                       target_strength_db=target_db, temperature=12.0, salinity=35.0,
                       absorption_model="ainslie_mccolm", array_gain_db=15.0)
    assert eq.snr(range_m) == pytest.approx(plan_snr, abs=1.0)
    assert eq.absorption_db_per_m() * 1000 == pytest.approx(0.64, abs=0.06)       # plan: ~0.64 dB/km


def test_sonar_equation_against_the_plans_popoto_estimate():
    """docs/plan, Phase 0: +10 dB target, 0.5 s chirp, eight elements, 190 dB source:
    about 37 dB at 1 km, 13 dB at 2 km, none by 3 km. The plan does not state the noise
    level it used; 38 dB re 1 uPa^2/Hz reproduces both figures."""
    eq = SonarEquation(source_level_db=190.0, center_frequency=27e3, bandwidth=10e3, pulse_duration=0.5,
                       n_elements=8, noise_spectrum_db=38.0, target_strength_db=10.0,
                       temperature=12.0, salinity=35.0, absorption_model="ainslie_mccolm")
    assert eq.absorption_db_per_m() * 1000 == pytest.approx(5.6, abs=0.4)         # plan: ~5.6 dB/km
    assert eq.snr(1000.0) == pytest.approx(37.0, abs=1.5)
    assert eq.snr(2000.0) == pytest.approx(13.0, abs=1.5)
    assert eq.snr(3000.0) < 0.0


def test_configs_carry_the_plans_front_end_numbers():
    p = SonarModel.load("popoto_8").config
    assert p["receive"]["gain_db"] == 43.0                      # 43 dB preamp
    assert p["receive"]["adc_full_scale"] == pytest.approx(0.0185 * 10 ** (43 / 20), rel=0.01)   # 37 mVpp at the input
    assert p["transmit"]["waveform"]["duration"] == 0.5
    assert SonarModel.load("flank_32").config["receive"]["sensitivity_db"] == -208.0
