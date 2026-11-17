"""Validation against SST's own document (APL-UW TR 0702), end to end: scenario -> records ->
processing, with tools/sst_validation.py doing the measuring.

Two worked examples of the report are run ("Pursuit", Sec. 3; ASIAEX forward scatter, Sec. 12)
with parameters of our own where the report gives none: the scenario files mark them
`unconfirmed`, and nothing here is a numerical reproduction of the report's figures. Each
feature the text describes is asserted as a number on our output against a closed form
evaluated for our geometry (TR 0702 eqs. 36-41, 63, 66, 68, 71 and the rules of Sec. 12.4).
Then closed forms through the whole simulator that the feature tests check at module level
only.

Two tests pin a defect this package found in the reverberation generator (one Doppler shift
per arrival direction); they were strict xfails until the reverberation package fixed it.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import sst_validation as sv  # noqa: E402

from gpusonar.config import load_scenario_config  # noqa: E402


@pytest.fixture(scope="module")
def pursuit():
    return sv.pursuit()


@pytest.fixture(scope="module")
def asiaex():
    return sv.asiaex()


# =================================================================================================
# The "Pursuit" example (TR 0702 Sec. 3, Fig. 2)
# =================================================================================================

def test_pursuit_scenario_holds_what_tr0702_states(pursuit):
    """What Sec. 3 gives: monostatic, forward-looking, five receive channels, a narrow-band
    tone near 10 kHz, a three-highlight target running away, 0.5 to 5 s. Everything else is
    marked as our choice."""
    cfg = load_scenario_config("sst_pursuit")
    so = cfg["sonar_overrides"]
    assert so["array"]["count"] == 5 and pursuit["parameters"]["channels"] == 5
    assert so["array"]["boresight"] == [1, 0, 0] and cfg["platform"]["velocity"][1:] == [0, 0]        # looks where it goes
    assert so["transmit"]["lever_arm"] == [0, 0, 0]                                                    # monostatic
    wf = so["transmit"]["waveform"]
    assert wf["type"] == "cw" and abs(wf["f_start"] - 10e3) < 200.0
    target = cfg["scene"][0]
    assert len(target["highlights"]) == 3 and target["velocity"][0] > 0.0                              # fleeing
    c = cfg["environment"]["sound_speed"]
    r0, r1 = so["receive"]["window"]["range"]
    assert 2 * r0 / c == pytest.approx(0.5) and 2 * r1 / c == pytest.approx(5.0)
    for block in ("platform", "propagation", "scene", "reverberation", "sonar_overrides"):
        assert block in cfg["unconfirmed"]
    assert cfg["reverberation"][0]["geometry"] == "bistatic"       # the mode with a Doppler per arrival direction


def test_pursuit_reverberation_hook_follows_the_closed_form(pursuit):
    """ "The early part of the signal comes from almost directly overhead ... very little
    Doppler shift. The later part ... from the main lobe ... a high upward Doppler shift."
    The ridge of the spectrogram of the surface return is f (1 + 2 v cos(angle) / c) at each
    time, cos(angle) from the vehicle's depth and the two-way time."""
    hook = pursuit["hook"]
    cell = hook["resolution_hz"]
    assert hook["fraction_inside_slew_bracket"] > 0.9
    assert abs(hook["settled_ridge_minus_first_order_hz"]["median"]) < 0.35 * cell
    assert hook["settled_ridge_minus_first_order_hz"]["max_abs"] < cell
    assert hook["ridge_minus_first_order_hz"]["p90_abs"] < cell
    # The hook: low early, high late, rising.
    assert hook["early_shift_hz"] < 0.6 * hook["late_shift_hz"]
    assert hook["rank_correlation_with_time"] > 0.9
    assert hook["late_shift_hz"] == pytest.approx(hook["late_closed_form_hz"], abs=0.35 * cell)
    # The closed form is eq. (71) itself, as the generator now has it; the first-order form is 3.6 Hz lower at 20 m/s.
    assert -5.0 < hook["first_order_minus_eq71_hz"] < 0.0


def test_pursuit_bottom_bounce_streak_starts_when_geometry_says(pursuit):
    """ "Less obvious is the streak that starts at about 1.7 seconds and merges with the main
    ridge; that feature results from a path that reflects from the bottom." With our water
    depth the first such path (one leg direct, one by the bottom) arrives at 2 D / c."""
    b = pursuit["bottom_bounce"]
    lo, hi = b["onset_window_s"]
    assert b["closed_form_onset_s"] == pytest.approx(1.7)
    assert lo <= b["measured_onset_s"] <= hi
    assert abs(b["excess_before_db"]) < 3.0 and b["excess_after_db"] > 6.0
    # Its power lies on the frequency track eq. (71) gives that pair.
    assert b["on_true_track_db_re_no_bottom"] > 8.0
    assert b["on_true_track_db_re_no_bottom"] > b["control_40hz_above_track_db_re_no_bottom"] + 6.0


def test_pursuit_target_echo_is_the_simple_target_echo_model(pursuit):
    """ "The target echo is the blip at about 2.8 seconds and 10.13 kHz": at the time and the
    frequency the geometry gives (eqs. 63 / 68), and equal, sample by sample, to eq. (66) for
    the three highlights with the sonar equation's level."""
    t = pursuit["target"]
    assert t["measured_arrival_s"] == pytest.approx(t["closed_form_arrival_s"], abs=2e-3)
    assert t["closed_form_arrival_s"] == pytest.approx(2.8, abs=0.02)          # where we put it
    assert t["eq68_frequency_hz"] == pytest.approx(t["exact_kinematics_frequency_hz"], abs=1e-6)
    assert t["measured_frequency_hz"] == pytest.approx(t["eq68_frequency_hz"], abs=0.2)
    assert t["eq68_frequency_hz"] == pytest.approx(10.13e3, abs=10.0)          # where we put it
    assert t["echo_vs_eq66_residual"] < 0.01
    assert abs(t["level_error_db"]) < 0.05
    # Time spreading of the surface does not touch the direct echo.
    assert pursuit["time_spreading"]["direct_echo_change"] < 1e-6
    # It only makes the weaker, later echo with a surface-reflected leg a random one.
    assert pursuit["time_spreading"]["surface_bounce_echo_db_re_direct"] < -3.0
    assert abs(pursuit["time_spreading"]["surface_bounce_echo_change_db"]) < 3.0


def test_pursuit_target_is_below_the_reverberation_but_separated_in_doppler(pursuit):
    """ "It has a lower Doppler shift than the main reverberation ridge because the target is
    running away. This is an 'easy' high-Doppler detection despite the fact that the level of
    the target return is well below the level of the reverberation." """
    t = pursuit["target"]
    assert t["separation_from_ridge_hz"] > 1.5 * pursuit["hook"]["resolution_hz"]
    assert t["cell_db_above_reverberation"] > 8.0 and t["cell_db_above_noise"] > 8.0
    assert t["band_level_echo_minus_reverberation_db"] < -3.0
    assert t["band_level_rise_from_the_echo_db"] < 1.5                         # "doesn't show at all on the level plot"


def test_pursuit_surface_reverberation_dominates(pursuit):
    """ "The dominant feature in the figure is the reverberation, which comes primarily from
    the surface." Echoes, reverberation and noise add, so each can be measured alone."""
    levels, noise = pursuit["levels"], pursuit["noise"]
    assert levels["ridge_psd_over_noise_psd_db"]["median"] > 15.0
    assert levels["band_reverberation_over_noise_db"]["median"] > 6.0
    assert levels["band_reverberation_over_noise_db"]["fraction_above"] > 0.85
    assert levels["bottom_paths_share_db"] < -3.0                              # the paths by the bottom are the lesser part
    # What is left of the full record without echoes and reverberation is the noise asked for.
    assert noise["psd_db_re_upa2_per_hz"] == pytest.approx(noise["scenario_spectrum_db"], abs=0.5)


# The defect these two tests pinned (fixed in sim/reverb_bistatic.py: components are keyed by Doppler bin as well): a
# reverberation component was (azimuth sector, elevation of the RECEIVE ray) and carried ONE Doppler shift, the
# power-weighted mean over all transmit paths and all times that arrive from there. Eigenray pairs with different
# transmit legs (direct and bottom-reflected) that share a receive ray therefore shared a shift: the weaker pair was
# given the stronger pair's Doppler. TR 0702 eq. (71) gives each pair pq its own.


def test_reverberation_gives_each_eigenray_pair_its_own_doppler():
    """Deterministic form: the mean Doppler shift of the surface return from dead ahead
    against two-way time, from the generator's components and from a direct integration
    over the surface with eq. (71) for each pair of paths (direct, one bottom bounce)."""
    found = sv.pursuit_doppler_of_cross_pairs()
    assert found["max_error_direct_only_hz"] < 3.0
    assert found["max_error_with_bottom_paths_hz"] < 3.0


def test_doppler_comparison_is_sound_without_a_bottom():
    """The same comparison with the direct path only: the method of the test above agrees
    with the generator where no two pairs share a receive ray."""
    found = sv.pursuit_doppler_of_cross_pairs(bottom=False)
    assert found["max_error_direct_only_hz"] < 3.0
    assert found["max_error_with_bottom_paths_hz"] < 3.0


def test_bottom_paths_change_nothing_before_they_can_arrive(pursuit):
    """Causality, end to end: before 2 D / c the record with a reflecting bottom and the
    record without one differ only by their random draws (three pings averaged, 70 Hz bands,
    three segments: about 2 dB)."""
    assert abs(pursuit["bottom_bounce"]["largest_change_before_onset_db"]) < 6.0
    # And once they arrive, nothing lies above the largest Doppler eq. (71) allows that pair
    # (40 Hz above its track from dead ahead).
    assert pursuit["bottom_bounce"]["control_40hz_above_track_db_re_no_bottom"] < 3.0


# =================================================================================================
# ASIAEX forward scatter (TR 0702 Sec. 12.2-12.6, Figs. 3-5)
# =================================================================================================

def test_asiaex_scenario_holds_what_tr0702_states(asiaex):
    cfg = load_scenario_config("sst_asiaex")
    p = asiaex["parameters"]
    assert p["frequency_hz"] == 8000.0 and p["pulse_ms"] == pytest.approx(3.0) and p["separation_m"] == pytest.approx(500.0)
    assert p["pings"] == 20
    assert cfg["time_spread"]["bottom"] == {"rms_height": 0.012, "mean_square_slope": 0.045}           # as printed, p. 104
    assert cfg["time_spread"]["surface"]["wind_speed"] in (3.0, 4.0)                                   # the two winds of Sec. 12.6
    for block in ("platform", "sonar_overrides", "propagation", "time_spread.surface"):
        assert block in cfg["unconfirmed"]
    # The five arrivals of Sec. 12.2 lie inside the 35 ms of the figures, in that order.
    times = [asiaex["arrivals"][k]["closed_form_ms"] for k in ("direct", "surface", "bottom", "surface-bottom", "bottom-surface")]
    assert times == sorted(times) and times[0] == 0.0 and times[-1] + p["pulse_ms"] < 35.0
    assert np.diff(times).min() > p["pulse_ms"]                                                        # separate pulses


def test_asiaex_arrival_times_and_levels_are_the_image_closed_forms(asiaex):
    """Arrival times against eqs. (36)-(37) and levels against eq. (38), read off the record."""
    assert asiaex["arrival_time_max_error_us"] < 2.0
    assert asiaex["arrival_level_max_error_db"] < 0.05
    for row in asiaex["arrivals"].values():
        assert row["eq36_vertical_m"] == pytest.approx(row["unfolded_vertical_m"], abs=1e-9)


def test_asiaex_without_time_spreading_peaks_are_crisp_and_identical(asiaex):
    """ "there is no ping-to-ping variation. The leading and trailing edges of the peaks are
    all equally crisp." """
    flat = asiaex["no_time_spread"]
    assert flat["ping_to_ping_relative_std"] < 1e-9
    assert max(flat["db_after_pulse_re_peak"].values()) < -35.0
    rise = list(flat["rise_time_10_90_ms"].values())
    assert max(rise) - min(rise) < 0.02 and max(rise) < 0.3                    # equally crisp: the pulse's own edge


def test_asiaex_with_time_spreading_bounce_paths_get_tails_and_vary(asiaex):
    """ "the DirectSignal result shows time spread and ping-to-ping variation": the bounce
    paths do, the direct path does not."""
    spread, flat = asiaex["time_spread"], asiaex["no_time_spread"]
    assert spread["ping_to_ping_std_db"]["direct"] < 1e-3
    assert spread["peak_mean_db_re_flat"]["direct"] == pytest.approx(0.0, abs=1e-3)
    assert spread["ping_to_ping_std_db"]["surface"] > 0.5
    assert spread["tail_ping_to_ping_std_db"]["surface"] > 2.0
    # Tails: tens of dB above what a flat boundary leaves after the pulse.
    for name in ("surface", "bottom"):
        assert spread["db_after_pulse_re_peak"][name] > flat["db_after_pulse_re_peak"][name] + 20.0
    # "more pronounced for the surface than for the bottom"
    assert spread["db_after_pulse_re_peak"]["surface"] > spread["db_after_pulse_re_peak"]["bottom"]


def test_asiaex_tails_decay_at_the_rule_of_thumb(asiaex):
    """Sec. 12.4: "the characteristic spreading time is roughly equal to the mean square slope
    times half of the total one-way propagation time if the reflection is near the middle of
    the path", less for an asymmetric path: 2 mss t1 t2 / (t1 + t2). Fitted to the mean
    intensity after the pulse (200 pings for the surface; 40 realisations of the bottom, with
    the report's meanSquareSlope = 0.045)."""
    tails = asiaex["tails"]
    assert tails["surface"]["fitted_ms"] == pytest.approx(tails["surface"]["rule_of_thumb_ms"], rel=0.15)
    assert tails["bottom"]["fitted_ms"] == pytest.approx(tails["bottom"]["rule_of_thumb_ms"], rel=0.2)
    assert tails["bottom"]["mean_square_slope"] == 0.045
    # rmsHeight = 0.012 m: far below wavelength / sin(grazing), "mostly coherent".
    assert tails["bottom"]["coherent_fraction"] > 0.9


def test_asiaex_surface_doppler_spread_is_under_10_hz(asiaex):
    """ "the Doppler spreading width for the surface peak is less than 10 Hz". Measured from
    the decorrelation of the tail from ping to ping (800 pings 50 ms apart), for the default
    AR(1) process and for the Gaussian one, against each one's closed form."""
    d = asiaex["surface_doppler_spread"]
    for name in ("lorentzian", "gaussian"):
        assert d[name]["measured_hz"] < d["tr0702_bound_hz"]
        assert d[name]["measured_hz"] == pytest.approx(d[name]["closed_form_hz"], rel=0.25)
    # The rule as printed on p. 101 (divided by the sine) is also under 10 Hz here: the
    # remark does not tell the two readings apart.
    assert d["lorentzian"]["closed_form_hz"] < d["p101_rule_hz"] < d["tr0702_bound_hz"]


def test_asiaex_two_bounce_tail_sum_against_convolve(asiaex):
    """ "Now turn your attention to the multiple-bounce peaks ... The intensity is too high
    immediately after the pulse and too low later." TR 0702 does not say how SST combined two
    bounces. Ours by default adds the mean delays into one exponential (`sum`);
    `convolve` convolves the bounces' responses.

    With two bounces that both scatter (the variant), `sum` has the stated defect. With the
    ASIAEX bottom as printed (93 % coherent, long decay) its error has the other sign: every
    scattered part is given the bottom's delay although 7 % of it met a rough bottom."""
    rough, two = asiaex["two_rough_bounces"], asiaex["two_bounces"]
    assert rough["sum_over_convolve_first_half_of_pulse_db"] > 1.0             # too high at once
    assert rough["sum_over_convolve_2_to_5_ms_after_pulse_db"] < -0.4          # too low later
    assert abs(rough["tail_energy_sum_over_convolve_db"]) < 1.0                # same energy
    assert two["coherent_fraction_bottom"] > 0.9 > 0.5 > two["coherent_fraction_surface"]
    assert two["mean_delay_sum_ms"] > 3.0 * two["mean_delay_convolve_ms"]
    assert two["sum_over_convolve_early_db"] < -1.0 and two["sum_over_convolve_late_db"] > 1.0
    assert two["tail_centroid_sum_ms"] > two["tail_centroid_convolve_ms"] + 0.5
    assert abs(two["tail_energy_sum_over_convolve_db"]) < 1.0


# =================================================================================================
# Closed forms through the whole simulator
# =================================================================================================

def test_image_equations_as_printed_match_the_provider():
    """TR 0702 eqs. (36), (37), (40), (41) as printed: equal to a leg-by-leg unfolding and to
    the provider's images for every bounce sequence of up to three bounces (docs/PHYSICS_NOTES.md
    Sec. 11 item 24 asked for eq. (36) to be re-derived before it is relied on)."""
    found = sv.check_image_model()
    assert len(found["sequences"]) == 7
    assert found["eq36_vs_unfolding_m"] < 1e-9
    assert found["delay_s"] < 1e-12
    assert found["launch"] < 1e-12 and found["arrival"] < 1e-12
    assert found["bounce_fraction"] < 1e-9


def test_echo_level_is_the_sonar_equation():
    """A highlight of 7 dB, scenario -> record -> baseband -> pulse compression: the peak is
    SL - 2 TL + TS in free field, and in the image medium each of the six groups of eigenray
    pairs has the level and the delay of eqs. (36)-(38) and (66)."""
    found = sv.check_echo_level()
    assert found["free_field"]["max_level_error_db"] < 0.05
    assert found["free_field"]["max_time_error_us"] < 2.0
    assert set(found["image_source"]["pairs"]) == {"dd", "ds", "ss", "db", "sb", "bb"}
    assert found["image_source"]["max_level_error_db"] < 0.05
    assert found["image_source"]["max_time_error_us"] < 2.0


def test_doppler_bank_finds_a_moving_highlight_through_a_sampled_pulse():
    """A highlight target closing at 6 m/s, an m-sequence given as samples, the Doppler bank:
    the range rate, the time scale of eq. (68), and the level."""
    found = sv.check_sampled_doppler()
    assert found["synthesis"] == "numpy_sampled"
    assert found["bank_range_rate"] == found["true_range_rate"]
    assert found["bank_range_rate_fine"] == pytest.approx(found["true_range_rate"], abs=0.11)
    assert found["eq68_time_scale"] == pytest.approx(found["exact_time_scale"], rel=1e-12)
    assert found["peak_without_doppler_over_best"] < 0.3
    assert found["peak_over_sonar_equation"] == pytest.approx(1.0, abs=0.02)


@pytest.fixture(scope="module")
def sediment():
    return sv.check_sediment_reverberation()


def test_sediment_reverberation_follows_the_scattering_strength_integral(sediment):
    """`medium_sand` by name through run_scenario: the mean intensity against time is
    sigma(grazing) pi c T / R^3 with sigma from the model, over grazing angles of 8 to 42
    degrees (48 ping-channels, 12 range blocks)."""
    assert sediment["sediment"] == "medium_sand"
    d = sediment["scattering_function_vs_integral_db"]
    assert abs(d["mean"]) < 0.7 and d["min"] > -1.5 and d["max"] < 1.5
    lo, hi = sediment["grazing_range_deg"]
    assert lo < 10.0 and hi > 40.0


def test_point_scatterers_and_scattering_function_agree_for_medium_sand(sediment):
    """The same sediment as scatterers in the scene (TR 0702 Sec. 11.1) and as a reverberation
    boundary (Sec. 11.3), both through run_scenario, and both against the integral."""
    d = sediment["point_scatterers_vs_integral_db"]
    assert abs(d["mean"]) < 0.7 and d["min"] > -2.0 and d["max"] < 2.0
    both = sediment["point_scatterers_vs_scattering_function_db"]
    assert abs(both["mean"]) < 1.0 and both["min"] > -2.0 and both["max"] < 2.0


def test_signal_to_reverberation_after_pulse_compression():
    """A 20 dB target in `medium_sand` reverberation, LFM pulses of 1, 2 and 4 kHz, pulse
    compression: the ratio is TS - 10 log10(sigma pi c R / B) and grows by 3 dB per octave
    (TR 0702 p. 70)."""
    found = sv.check_signal_to_reverberation()
    assert found["max_error_db"] < 1.0
    assert np.allclose(found["gain_per_octave_db"], 10.0 * np.log10(2.0), atol=0.8)
    for row in found["bands"].values():
        assert row["echo_peak_over_sonar_equation"] == pytest.approx(1.0, abs=0.01)


# =================================================================================================
# The matrix
# =================================================================================================

def test_matrix_has_a_row_and_a_verdict_for_every_section(pursuit, asiaex, sediment):
    rows = sv.matrix({"pursuit": pursuit, "asiaex": asiaex,
                      "checks": {"image_model": sv.check_image_model(), "absorption": sv.check_absorption(),
                                 "echo_level": sv.check_echo_level(), "sampled_doppler": sv.check_sampled_doppler(),
                                 "sediment_reverberation": sediment, "signal_to_reverberation": sv.check_signal_to_reverberation()}})
    text = sv.markdown(rows)
    where = " ".join(r["where"] for r in rows)
    for section in ("Sec. 2", "Sec. 3", "Sec. 4", "Sec. 5", "Sec. 6", "Sec. 7", "Sec. 8", "Sec. 9", "Sec. 10", "Sec. 11", "Sec. 12"):
        assert section in where, section
    assert len(rows) >= 50
    for r in rows:
        assert r["result"] and "(not run)" not in r["result"], r["where"]
        assert r["check"]
    # Every test of this file that the matrix names exists.
    import re
    names = set(re.findall(r"test_sst_validation\.py::(test_\w+)", text)) | set(re.findall(r"::(test_\w+) \(xfail\)", text))
    for name in names:
        assert name in globals(), name
    # The readings of the inconsistent passages: items 17-24 of PHYSICS_NOTES Sec. 11.
    assert [r["item"] for r in sv.readings()] == list(range(17, 25))
    assert "NOT a numerical reproduction" in text
    # Without running anything the matrix still lists every row.
    assert len(sv.matrix()) == len(rows)
