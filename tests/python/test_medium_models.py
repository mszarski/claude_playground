"""Medium helpers added from the org-arl survey: sea state, sea water density, Thorp absorption
and Wood's equation for bubbly water."""

import math

import numpy as np
import pytest

from gpusonar import acoustics
from gpusonar.config import load_scenario_config
from gpusonar.sim import run_scenario, timespread


# --- sea state -----------------------------------------------------------------------------------

def test_sea_state_table():
    """The ten values UnderwaterAcoustics.jl prints (src/stdlib.jl:413-422), typed in again."""
    printed = [0.8, 2.6, 4.4, 6.9, 9.8, 12.6, 19.3, 26.5, 30.6, 32.9]
    assert [acoustics.sea_state_wind_speed(n) for n in range(10)] == printed
    assert acoustics.sea_state_wind_speed(3.0) == 6.9
    assert all(b > a for a, b in zip(printed, printed[1:]))
    for bad in (-1, 10, 2.5, "x"):
        with pytest.raises(ValueError):
            acoustics.sea_state_wind_speed(bad)


def test_resolve_sea_state_everywhere_and_conflicts():
    cfg = {"time_spread": {"surface": {"sea_state": 3, "bubble_loss_db": {"model": "hall_novarini", "sea_state": 4}}},
           "reverberation": [{"model": "chapman_harris", "boundary_z": 0.0, "sea_state": 2},
                             {"boundary_z": 30.0, "mu_db": -27.0}],
           "receiver": {"ambient_noise": {"sea_state": 5, "shipping": 0.5}},
           "propagation": {"wind_speed": 6.9, "sea_state": 3}, "seed": 4}
    out = acoustics.resolve_sea_state(cfg)
    assert out["time_spread"]["surface"] == {"wind_speed": 6.9, "bubble_loss_db": {"model": "hall_novarini", "wind_speed": 9.8}}
    assert out["reverberation"][0] == {"model": "chapman_harris", "boundary_z": 0.0, "wind_speed": 4.4}
    assert out["reverberation"][1] == cfg["reverberation"][1]
    assert out["receiver"]["ambient_noise"] == {"wind_speed": 12.6, "shipping": 0.5}
    assert out["propagation"] == {"wind_speed": 6.9}                      # both given and agreeing
    assert "sea_state" in cfg["time_spread"]["surface"]                   # the input is not modified
    plain = {"a": [1, {"wind_speed": 5.0}], "b": "sea_state"}
    assert acoustics.resolve_sea_state(plain) == plain                    # nothing to do: equal content
    with pytest.raises(ValueError, match=r"scenario\.time_spread\.surface.*6\.9.*wind_speed is 7\.5"):
        acoustics.resolve_sea_state({"time_spread": {"surface": {"sea_state": 3, "wind_speed": 7.5}}})
    with pytest.raises(ValueError, match="0 to 9"):
        acoustics.resolve_sea_state({"reverberation": [{"sea_state": 12}]})


def test_boundary_parameters_from_a_resolved_sea_state():
    surface = acoustics.resolve_sea_state({"sea_state": 4})
    assert timespread.boundary_parameters(surface, surface=True) == timespread.boundary_parameters({"wind_speed": 9.8}, surface=True)


def test_scenario_with_sea_state_equals_the_wind_speed_one():
    """`sea_state: 3` in a scenario's time-spread section gives the samples of `wind_speed: 6.9`,
    bit for bit, and not those of another wind. (Reverberation items and the ambient noise go
    through the same pass: test_resolve_sea_state_everywhere_and_conflicts.)"""
    def samples(key, value):
        cfg = load_scenario_config("point_grid_fls_300k")
        cfg["fidelity"] = "image_source"
        cfg["propagation"] = {"surface_z": 0.0, "bottom_z": 120.0, "bottom_coefficient": 0.5, "max_order": 1}
        cfg["scene"][0].update(ranges=cfg["scene"][0]["ranges"][:1], bearings_deg=[0.0])
        cfg["time_spread"] = {"surface": {key: value}, "taps": 2}
        cfg["receiver"] = {"noise_rms": 0.0, "adc": False}
        out = run_scenario(cfg).records[0].samples.array
        assert np.abs(out).max() > 0.0
        return out
    a, b, c = samples("sea_state", 1), samples("wind_speed", 2.6), samples("wind_speed", 4.4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_ambient_noise_spectrum_from_a_sea_state():
    from gpusonar.sim import simulator
    f = np.array([1e3, 7e3, 30e3])
    by_state = simulator._noise_spectrum(acoustics.resolve_sea_state({"sea_state": 5, "shipping": 0.4}))(f)
    by_wind = simulator._noise_spectrum({"wind_speed": 12.6, "shipping": 0.4})(f)
    assert np.array_equal(by_state, by_wind)
    assert not np.array_equal(by_state, simulator._noise_spectrum({"shipping": 0.4})(f))   # not the 5 m/s default


# --- density -------------------------------------------------------------------------------------

def test_water_density():
    # The value both packages document: arlpy.uwa.density() -> 1022.7 at 27 deg C, 35 ppt.
    assert round(acoustics.water_density(27.0, 35.0), 1) == 1022.7
    # Pure water (S = 0) is the first polynomial alone. Its density has its maximum just below
    # 4 deg C, within 0.03 kg/m3 of 1000 (the old definition of the kilogram: general knowledge,
    # not from the source of the coefficients).
    t = np.linspace(0.0, 10.0, 1001)
    rho = np.array([acoustics.water_density(v, 0.0) for v in t])
    assert 3.9 < t[rho.argmax()] < 4.1 and abs(rho.max() - 1000.0) < 0.03
    # Salt adds roughly 0.8 kg/m3 per PSU, and warm water is lighter.
    assert 0.74 < (acoustics.water_density(10.0, 35.0) - acoustics.water_density(10.0, 0.0)) / 35.0 < 0.82
    assert acoustics.water_density(25.0, 35.0) < acoustics.water_density(5.0, 35.0)
    # The reference water of the UnderwaterAcoustics.jl seabed table (1023 kg/m3) is its own
    # default water: 27 deg C and 35 ppt to the nearest kg/m3.
    assert round(acoustics.water_density(27.0, 35.0)) == 1023
    # 1027.67547 kg/m3 at 5 deg C, 35 PSU and atmospheric pressure is, from memory, the check value
    # published with the 1980 equation of state; it was not read in a document here.
    assert acoustics.water_density(5.0, 35.0) == pytest.approx(1027.67547, abs=1e-5)


def test_water_density_against_arlpy():
    uwa = pytest.importorskip("arlpy.uwa")
    for t, s in ((0.0, 0.0), (10.0, 35.0), (27.0, 35.0), (18.0, 7.0), (30.0, 40.0)):
        assert acoustics.water_density(t, s) == pytest.approx(float(uwa.density(t, s)), rel=1e-13)



# --- sound speed and Francois-Garrison against arlpy (audit B1, B2, A11) ---------------------------

def test_sea_water_sound_speed_is_mackenzie_as_arlpy_computes_it_and_prints_it():
    """arlpy `uwa.soundspeed` is an independent transcription of Mackenzie (1981) (REFERENCES.md: arlpy oracle).
    Its docstring prints 1539.1 m/s at 27 deg C, 35 ppt, 10 m and 1534.6 at 25 deg C, 20 m."""
    uwa = pytest.importorskip("arlpy.uwa")
    for t in (-1.0, 0.0, 4.0, 10.0, 15.0, 25.0, 30.0):
        for sal in (5.0, 25.0, 35.0, 40.0):
            for d in (0.0, 10.0, 500.0, 2000.0, 8000.0):
                assert acoustics.sound_speed(t, sal, d) == pytest.approx(float(uwa.soundspeed(t, sal, d)), rel=1e-13, abs=0.0), (t, sal, d)
    assert round(acoustics.sound_speed(27.0, 35.0, 10.0), 1) == 1539.1
    assert round(acoustics.sound_speed(25.0, 35.0, 20.0), 1) == 1534.6


def test_brackish_sound_speed_is_our_blend_and_has_no_step_at_5_psu():
    """Audit A11: below 5 PSU the speed is OUR OWN mix (acoustics.sound_speed docstring), linear in salinity from
    Marczak's pure water at S = 0 to Mackenzie at 5 PSU. Before, it stepped down by 0.81 / 1.31 / 1.67 m/s at
    4 / 10 / 20 deg C crossing 5 PSU. Marczak's own published check values were not available to test against."""
    marczak = (1.402385e3, 5.038813, -5.799136e-2, 3.287156e-4, -1.398845e-6, 2.787860e-9)     # the paper's coefficients as coded
    for t in (0.0, 4.0, 10.0, 20.0, 30.0):
        assert acoustics.sound_speed(t, 0.0) == pytest.approx(sum(a * t**k for k, a in enumerate(marczak)), abs=1e-9)
        for d in (0.0, 100.0):
            below, at = acoustics.sound_speed(t, 5.0 - 1e-9, d), acoustics.sound_speed(t, 5.0, d)
            assert abs(at - below) < 1e-6, (t, d)                                       # continuous
            c = [acoustics.sound_speed(t, s, d) for s in (0.0, 1.0, 2.5, 4.0)]
            assert np.diff(c) / np.diff([0.0, 1.0, 2.5, 4.0]) == pytest.approx([(at - c[0]) / 5.0] * 3, rel=1e-9)   # linear in S
    # the size of the old step, so that the docstring's numbers stay true
    old = {t: sum(a * t**k for k, a in enumerate(marczak)) + 1.340 * 5.0 for t in (4.0, 10.0, 20.0)}
    assert [round(old[t] - acoustics.sound_speed(t, 5.0), 2) for t in (4.0, 10.0, 20.0)] == [0.81, 1.31, 1.67]


def test_francois_garrison_at_sonar_frequencies_against_arlpy():
    """Audit B2: the published-table check stops at 10 kHz and about 1 deg C; the sonars run at 100 to 750 kHz.
    arlpy `uwa.absorption` (Francois-Garrison, an independent transcription) over 10 kHz to 1 MHz, cold and warm
    water, surface to 1000 m. Not at exactly 20 deg C: arlpy changes pure-water branch below 20, ours at 20 and
    below. arlpy's docstring prints -10.71 dB over 1 km at 50 kHz for its default water (27 deg C, 35 ppt, 10 m, pH 8.1)."""
    uwa = pytest.importorskip("arlpy.uwa")
    f = np.geomspace(1e4, 1e6, 41)
    for t, s, d, ph in ((2.0, 34.0, 0.0, 8.0), (10.0, 35.0, 50.0, 8.0), (15.0, 32.0, 500.0, 7.8), (27.0, 35.0, 10.0, 8.1), (30.0, 38.0, 1000.0, 8.2)):
        ours = acoustics.absorption_francois_garrison(f, t, s, d, ph) * 1e3                       # dB/km
        theirs = -20.0 * np.log10(np.array([uwa.absorption(x, 1000.0, t, s, d, ph) for x in f]))
        assert ours == pytest.approx(theirs, rel=1e-9), (t, s, d, ph)
    assert round(float(acoustics.absorption_francois_garrison(5e4, 27.0, 35.0, 10.0, 8.1)) * 1e3, 2) == 10.71

# --- Thorp ---------------------------------------------------------------------------------------

def test_thorp_absorption():
    # TR 0702 eq. (42) by hand at 8 kHz: 0.1 * 64 / 65 + 40 * 64 / 4164 dB/kyd, and a kiloyard is 914.4 m.
    per_kyd = 0.1 * 64.0 / 65.0 + 40.0 * 64.0 / 4164.0
    assert acoustics.absorption_thorp(8e3) * 1e3 == pytest.approx(per_kyd / 0.9144, rel=1e-12)
    # The number our validation of TR 0702 quotes: 0.78 dB/km at 8 kHz.
    assert round(float(acoustics.absorption(8e3, model="thorp")) * 1e3, 2) == 0.78
    # It ignores the water's state, and it is not the default.
    assert acoustics.absorption(8e3, 2.0, 30.0, 500.0, 7.8, model="thorp") == acoustics.absorption(8e3, model="thorp")
    assert acoustics.absorption(8e3) == acoustics.absorption_francois_garrison(8e3) != acoustics.absorption(8e3, model="thorp")
    # Limits of the expression: f^2 growth at low frequency (the first relaxation's 0.1 f^2 plus
    # 40 f^2 / 4100), and the sum of the two plateaus, 40.1 dB/kyd, at high frequency.
    f = 10.0
    assert acoustics.absorption_thorp(f) * 1e6 / acoustics.YARDS_PER_KM == pytest.approx((0.1 + 40.0 / 4100.0) * (f / 1e3) ** 2, rel=1e-3)
    with pytest.warns(UserWarning, match="pure-water term"):
        assert acoustics.absorption_thorp(1e8) * 1e6 / acoustics.YARDS_PER_KM == pytest.approx(40.1, rel=1e-3)
    # The comparison the validation of TR 0702 quotes at 8 kHz: 0.78 against Francois-Garrison's 0.68
    # dB/km for the conditions Thorp's fit is stated for (4 deg C, about 1 km depth).
    assert round(float(acoustics.absorption(8e3, 4.0, 35.0, 1000.0)) * 1e3, 2) == 0.68
    out = acoustics.absorption(np.array([4e3, 8e3, 10e3]), model="thorp")
    assert out.shape == (3,) and np.all(np.diff(out) > 0)
    with pytest.raises(ValueError):
        acoustics.absorption(8e3, model="thorpe")


def test_scenario_takes_thorp_by_name():
    cfg = load_scenario_config("point_grid_fls_300k")
    base = run_scenario(cfg).records[0].samples.array
    cfg["absorption"] = "thorp"
    with pytest.warns(UserWarning, match="pure-water term"):          # a 300 kHz sonar: above our 100 kHz warning limit (TR 0702 states none)
        assert not np.array_equal(run_scenario(cfg).records[0].samples.array, base)


# --- Wood ----------------------------------------------------------------------------------------

def test_bubbly_sound_speed_limits_and_closed_form():
    c, c_gas, ratio = 1500.0, 340.0, 1000.0
    assert acoustics.bubbly_sound_speed(0.0, c, c_gas, ratio) == pytest.approx(c, rel=1e-14)
    assert acoustics.bubbly_sound_speed(1.0, c, c_gas, ratio) == pytest.approx(c_gas, rel=1e-14)
    # Wood's equation written independently: mean density times mean compressibility.
    rho_w, rho_g = 1000.0, 1.0
    g = np.array([1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.5, 0.9, 0.999])
    density = g * rho_g + (1 - g) * rho_w
    compressibility = g / (rho_g * c_gas**2) + (1 - g) / (rho_w * c**2)
    assert np.allclose(acoustics.bubbly_sound_speed(g, c, c_gas, ratio), 1.0 / np.sqrt(density * compressibility), rtol=1e-13)
    # The value arlpy documents: bubble_soundspeed(1e-5) -> 1402.133 for its default water
    # (27 deg C, 35 ppt, 10 m: Mackenzie gives 1539.09 m/s).
    assert acoustics.bubbly_sound_speed(1e-5, acoustics.sound_speed(27.0, 35.0, 10.0)) == pytest.approx(1402.133, abs=2e-3)
    # The minimum: far below both speeds, at a void fraction of one half to within the small
    # terms, where the speed is 2 c_gas sqrt(rho_g / rho_w) to the same order.
    fine = np.linspace(0.3, 0.7, 4001)
    speeds = acoustics.bubbly_sound_speed(fine, c, c_gas, ratio)
    assert abs(fine[speeds.argmin()] - 0.5) < 2e-3
    assert speeds.min() == pytest.approx(2.0 * c_gas / math.sqrt(ratio), rel=2e-3) and speeds.min() < 0.07 * c_gas
    # A little gas matters a great deal: one part in ten thousand takes 40 % off.
    assert 0.55 < acoustics.bubbly_sound_speed(1e-4, c, c_gas, ratio) / c < 0.62
