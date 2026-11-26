"""The `sediment` tier's opt-in fluid penetration (`penetration: fluid`, G-3 of docs/tasks/SST_GAPS_JACKSON.md).

Above the critical angle a refracted path into a fluid sediment carries, per leg, the flat-interface
pressure transmission coefficient V_wp = 1 + V (Jackson and Richardson 2007, eqs. (8.47), (8.55)), the
reciprocity factor 1 / a_rho of the return leg (8.68), split as 1 / sqrt(a_rho) on each leg, and the
attenuation exp(-k_p'' L) along the sediment leg with k_p'' = k_w delta / nu ((8.18), (8.20), (8.26)).
Every expected value below is computed here from the book's equations as the note
docs/references/jackson_richardson_2007/03_wave_theories_reflection_penetration.md (sections 2.2.1, 2.2.3,
2.2.4, 3.5 and WP1) gives them, not from the code under test. Default: today's tier, unchanged.
"""

import warnings

import numpy as np
import pytest

from gpusonar.config import load_scenario_config
from gpusonar.sim import SedimentProvider, engine, make_provider, run_scenario
from gpusonar.sim import bottom as bottom_model
from gpusonar.sim.propagation import fluid_transmission, two_way_paths

C = 1500.0
ENGINES = ["numpy"] + (["cpp"] if engine.HAVE_CPP else [])
LN10 = np.log(10.0)


def sand():
    """medium_sand of configs/sediments.yaml over 1500 m/s water: {speed 1767, density 1.845, attenuation dB/wavelength}."""
    return bottom_model.fluid_halfspace("medium_sand", C)


def delta_of(b):
    return b["attenuation"] * LN10 / (40.0 * np.pi)          # (8.27): dB per wavelength = 40 pi delta / ln 10


def z_wp(grazing, b):
    """(8.53) with (8.25), (8.46): z_wp = a_rho a_p sin(theta_w) / sin(theta_p)."""
    a_rho = b["density"] / b.get("water_density", 1.0)
    a_p = (b["speed"] / C) / (1.0 + 1j * delta_of(b))
    sin_p = np.sqrt(1.0 - a_p**2 * np.cos(grazing) ** 2 + 0j)
    sin_p = np.where(sin_p.imag < 0, -sin_p, sin_p)
    return a_rho * a_p * np.sin(grazing) / sin_p, a_rho, a_p, sin_p


def v_wp(grazing, b):
    z = z_wp(grazing, b)[0]
    return 2.0 * z / (z + 1.0)                               # (8.55)


def alpha_db_per_m(b, f):
    """(8.26): alpha_p = 40 pi f delta_p / (nu_p c_w ln 10) dB/m."""
    return 40.0 * np.pi * f * delta_of(b) / ((b["speed"] / C) * C * LN10)


def fluid(b=None, f=20e3, **kw):
    b = sand() if b is None else b
    return SedimentProvider(C, b["speed"], 0.0, penetration="fluid", bottom=b, frequency=f, **kw)


# --- the coefficient --------------------------------------------------------------------------

def test_transmission_is_one_plus_reflection_and_conserves_energy():
    """V_wp = 1 + V (8.47) and, lossless, 1 - |V|^2 = |V_wp|^2 Re{sin theta_p / (a_rho a_p)} / sin theta_w (note 2.2.3)."""
    b = dict(sand(), attenuation=0.0)
    crit = np.arccos(C / b["speed"])
    g = np.linspace(crit + 1e-6, np.pi / 2, 400)
    t = fluid_transmission(g, C, **b)
    np.testing.assert_allclose(t, v_wp(g, b), rtol=1e-13)
    from gpusonar.sim.propagation import rayleigh_reflection
    v = rayleigh_reflection(g, C, **b)
    np.testing.assert_allclose(t, 1.0 + v, rtol=0, atol=1e-15)
    _, a_rho, a_p, sin_p = z_wp(g, b)
    flux = np.abs(t) ** 2 * np.real(sin_p / (a_rho * a_p)) / np.sin(g)
    np.testing.assert_allclose(1.0 - np.abs(v) ** 2, flux, rtol=0, atol=1e-12)


def test_matched_sediment_transmits_unity():
    """a_rho = 1, nu = 1 (and no loss): V = 0, V_wp = 1 at every angle (note WP1)."""
    g = np.linspace(0.01, np.pi / 2, 50)
    np.testing.assert_allclose(fluid_transmission(g, C, speed=C, density=1.0, attenuation=0.0), 1.0, atol=1e-15)


# --- the provider -----------------------------------------------------------------------------

@pytest.mark.parametrize("eng", ENGINES)
def test_worked_case_medium_sand_0_3_m(eng):
    """[N3 3.5]: 0.3 m of medium sand, vertical: 3.0 dB one way, 6 dB two-way at 20 kHz, 9 dB at 30 kHz of
    ATTENUATION; the interface's two-way factor |V_wp(90)|^2 / a_rho = 1.017 of TRANSMISSION. The note writes
    that as "+0.07 dB", 10 log10 of the factor; the factor multiplies PRESSURE (a product of two pressure
    coefficients), so it is 20 log10(1.017) = +0.146 dB. The total change against today's tier is their sum
    (spreading is unchanged): -5.85 dB at 20 kHz, -8.85 dB at 30 kHz."""
    b = sand()
    src, pt = np.array([0.0, 0.0, -1.5]), np.array([[0.0, 0.0, 0.3]])
    today = SedimentProvider(C, b["speed"], 0.0, engine=eng).eigenrays(src, pt).amplitude[0, 0]
    for f, att_two_way in ((20e3, 6.0), (30e3, 9.0)):
        new = fluid(b, f, engine=eng).eigenrays(src, pt).amplitude[0, 0]
        two_way_db = 2 * 20 * np.log10(abs(new / today))      # monostatic: the same leg down and back
        attenuation_db = 2 * 0.3 * alpha_db_per_m(b, f)
        transmission_db = 20 * np.log10(abs(v_wp(np.pi / 2, b)) ** 2 / (b["density"]))
        assert attenuation_db == pytest.approx(att_two_way, abs=0.01)      # 10.00 dB/m at 20 kHz (the table's input)
        assert abs(v_wp(np.pi / 2, b)) ** 2 / b["density"] == pytest.approx(1.017, abs=0.0005)
        assert transmission_db == pytest.approx(0.146, abs=0.001)
        assert two_way_db == pytest.approx(transmission_db - attenuation_db, abs=1e-10)


@pytest.mark.parametrize("eng", ENGINES)
def test_normal_incidence_is_exact(eng):
    """Straight down: grazing 90 deg, sediment path = depth, delay h / c_w + d / c_s, amplitude closed form."""
    b, f, h, d = sand(), 25e3, 2.0, 0.45
    e = fluid(b, f, engine=eng).eigenrays([1.0, -2.0, -h], [[1.0, -2.0, d]])
    t = h / C + d / b["speed"]
    assert e.delay[0, 0] == pytest.approx(t, rel=1e-15)
    expected = v_wp(np.pi / 2, b) / np.sqrt(b["density"]) * 10 ** (-alpha_db_per_m(b, f) * d / 20) / (C * t)
    np.testing.assert_allclose(e.amplitude[0, 0], expected, rtol=1e-13)
    # c t = h + d c_w / c_s is the distance of the paraxial refracted spherical wave on the axis: the
    # spreading of the tier is exact here (and only here; see docs/PHYSICS_NOTES.md).
    assert C * t == pytest.approx(h + d * C / b["speed"], rel=1e-15)


@pytest.mark.parametrize("eng", ENGINES)
def test_lossless_matched_limit_is_todays_tier(eng):
    """Zero loss and a sediment equal to the water: the new amplitudes are today's."""
    rng = np.random.default_rng(4)
    pts = np.column_stack([rng.uniform(-3, 3, 200), rng.uniform(-3, 3, 200), rng.uniform(-0.5, 1.0, 200)])
    src = np.array([0.2, 0.1, -1.5])
    matched = {"speed": C, "density": 1.0, "attenuation": 0.0}
    a = SedimentProvider(C, C, 0.0, penetration="fluid", bottom=matched, frequency=20e3, engine=eng,
                         subcritical="ray").eigenrays(src, pts)
    b = SedimentProvider(C, C, 0.0, engine=eng).eigenrays(src, pts)
    np.testing.assert_array_equal(a.delay, b.delay)
    np.testing.assert_allclose(a.amplitude, b.amplitude, rtol=1e-14)
    # Lossless but a real contrast: only the interface factor changes the amplitude.
    s = dict(sand(), attenuation=0.0)
    with pytest.warns(RuntimeWarning, match="least-time ray's level"):
        c = fluid(s, engine=eng, subcritical="ray").eigenrays(src, pts[pts[:, 2] > 0.05])
    d = SedimentProvider(C, s["speed"], 0.0, engine=eng).eigenrays(src, pts[pts[:, 2] > 0.05])
    g = np.arcsin(np.abs(d.launch[:, 0, 2]))
    np.testing.assert_allclose(c.amplitude[:, 0], d.amplitude[:, 0] * v_wp(g, s) / np.sqrt(s["density"]), rtol=1e-12)


@pytest.mark.parametrize("eng", ENGINES)
def test_points_in_the_water_are_untouched(eng):
    src = np.array([0.0, 0.0, -3.0])
    pts = np.array([[1.0, 0.5, -1.0], [2.0, 0.0, 0.0], [0.5, 0.0, -2.0]])    # above and ON the interface
    a = fluid(engine=eng).eigenrays(src, pts)
    b = SedimentProvider(C, sand()["speed"], 0.0, engine=eng).eigenrays(src, pts)
    np.testing.assert_array_equal(a.amplitude, b.amplitude)


@pytest.mark.parametrize("eng", ENGINES)
def test_reciprocal_under_swapping_source_and_receiver(eng):
    """Two-way product V_wp(theta_1) V_wp(theta_2) / a_rho exp(-k''(L_1 + L_2)) (8.68), the same with tx and rx swapped."""
    b, f = sand(), 18e3
    tx, rx = np.array([-0.4, 0.2, -1.5]), np.array([0.7, -0.3, -1.2])
    pts = np.array([[0.1, 0.0, 0.3], [0.5, 0.4, 0.22], [-0.6, -0.2, 0.5]])
    p = fluid(b, f, engine=eng)
    q1, p1, a1, *_ = two_way_paths(p, tx, rx, pts, np.ones(3), engine=eng)
    q2, p2, a2, *_ = two_way_paths(p, rx, tx, pts, np.ones(3), engine=eng)
    np.testing.assert_allclose(a1, a2, rtol=1e-13)
    # Independently: the two legs from their geometry.
    plain = SedimentProvider(C, b["speed"], 0.0, engine=eng)
    out, back = plain.eigenrays(tx, pts), plain.eigenrays(rx, pts)
    k2 = 2 * np.pi * f / C * delta_of(b) / (b["speed"] / C)
    from gpusonar.sim.propagation import snell_crossing
    legs = []
    for s, e in ((tx, out), (rx, back)):
        _, cross = snell_crossing(s, pts, C, b["speed"], 0.0, engine="numpy")
        g = np.arcsin(np.abs(e.launch[:, 0, 2]))
        legs.append((v_wp(g, b), np.linalg.norm(pts - cross, axis=1)))
    expected = legs[0][0] * legs[1][0] / b["density"] * np.exp(-k2 * (legs[0][1] + legs[1][1]))
    np.testing.assert_allclose(a1, expected, rtol=1e-12)


@pytest.mark.parametrize("eng", ENGINES)
def test_continuous_as_the_grazing_angle_falls_to_the_critical_angle(eng):
    """The coefficient is continuous at the critical angle (from above), and so is the amplitude of the rays
    whose water grazing angle approaches it (farther and farther points just under the interface)."""
    b = sand()
    crit = np.arccos(C / b["speed"])
    t_crit = fluid_transmission(crit, C, **b)
    steps = [abs(fluid_transmission(crit + e, C, **b) - t_crit) for e in (1e-2, 1e-4, 1e-6, 1e-8)]
    assert all(x > y for x, y in zip(steps, steps[1:])) and steps[-1] < 1e-5
    lossless = dict(b, attenuation=0.0)
    t0 = fluid_transmission(crit, C, **lossless)
    assert abs(fluid_transmission(crit + 1e-10, C, **lossless) - t0) < 1e-3     # sqrt branch point: ~ sqrt(eps)
    assert abs(t0 - 2.0) < 1e-6                                                     # |V| = 1 there: V_wp = 2
    # Rays: 5 mm deep, ranges where the water leg's grazing falls toward the critical angle.
    x = np.linspace(1.0, 6.0, 2001)
    pts = np.column_stack([x, np.zeros_like(x), np.full_like(x, 0.005)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e = fluid(b, 20e3, engine=eng, subcritical="ray").eigenrays([0.0, 0.0, -1.5], pts)
    g = np.arcsin(np.abs(e.launch[:, 0, 2]))
    assert np.all(g > crit) and g[-1] - crit < np.radians(2.0)
    ratio = np.abs(e.amplitude[:, 0] * C * e.delay[:, 0])                  # amplitude on top of spreading
    assert np.max(np.abs(np.diff(ratio))) < 5e-3                           # no jump between neighbours 2.5 mm apart


@pytest.mark.parametrize("eng", ENGINES)
def test_attenuation_scales_with_frequency(eng):
    """Constant delta: attenuation in dB linear in f ((8.26); the book: "at best an approximation", p. 251)."""
    b = sand()
    src, pt = [0.0, 0.0, -1.5], [[0.3, 0.1, 0.4]]
    plain = SedimentProvider(C, b["speed"], 0.0, engine=eng).eigenrays(src, pt)
    db = {f: 20 * np.log10(abs(fluid(b, f, engine=eng).eigenrays(src, pt).amplitude[0, 0] / plain.amplitude[0, 0]))
          for f in (10e3, 20e3, 40e3)}
    # Take off the interface factor (independent of frequency: delta is) so that the rest is attenuation alone.
    g = np.arcsin(abs(plain.launch[0, 0, 2]))
    tdb = 20 * np.log10(abs(v_wp(g, b)) / np.sqrt(b["density"]))
    att = {f: tdb - v for f, v in db.items()}
    assert att[20e3] / att[10e3] == pytest.approx(2.0, rel=1e-10)
    assert att[40e3] / att[10e3] == pytest.approx(4.0, rel=1e-10)


# --- what is refused, and the sub-critical points -----------------------------------------------

def test_sub_critical_points_get_the_evanescent_field_and_are_said_once():
    """Points whose straight line from the source meets the interface below the critical angle are reached by
    no real refracted plane wave (the flat-interface field is evanescent, [N3 3.5]): with the option on they get
    the zeroth-order field (15.1), V_wp(theta_d) / sqrt(a_rho) exp(-k_w Im{sqrt(1/a_p^2 - cos^2 theta_d)} d), never
    the least-time ray's level, and a warning says so once per provider; `refuse` makes it an error."""
    b = sand()
    pts = np.array([[0.0, 0.0, 0.3], [40.0, 0.0, 0.3]])                   # 90 deg and 14 deg (critical 31.9 deg)
    p = fluid(b)
    with pytest.warns(RuntimeWarning, match="1 of 2 points below the critical angle"):
        p.eigenrays([0.0, 0.0, -10.0], pts)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p.eigenrays([0.0, 0.0, -10.0], pts)                                  # said once per provider
        fluid(b).eigenrays([0.0, 0.0, -10.0], pts[:1])                       # none below: no warning
    with pytest.raises(ValueError, match="below the critical angle"):
        fluid(b, subcritical="refuse").eigenrays([0.0, 0.0, -10.0], pts)


@pytest.mark.parametrize("eng", ENGINES)
def test_evanescent_decay_is_the_notes(eng):
    """[N3 3.5]: medium sand, 10 deg: 145 / 364 / 728 / 2183 dB/m at 8 / 20 / 40 / 120 kHz (|P0| at 0.3 m: -43, -109,
    -218, -654 dB). The provider's sub-critical amplitude against the closed form, and against the note's rates."""
    b = sand()
    g = np.radians(10.0)
    src = np.array([0.0, 0.0, -1.0])
    d = 0.3
    pt = np.array([[(1.0 + d) / np.tan(g), 0.0, d]])                     # the straight line at exactly 10 deg
    plain = SedimentProvider(C, b["speed"], 0.0, engine=eng).eigenrays(src, pt)
    _, a_rho, a_p, _ = z_wp(g, b)
    for f, note_rate in ((8e3, 145.0), (20e3, 364.0), (40e3, 728.0), (120e3, 2183.0)):
        root = np.sqrt(1.0 / a_p**2 - np.cos(g) ** 2 + 0j)
        rate = 20.0 / LN10 * 2 * np.pi * f / C * abs(root.imag)          # dB/m
        assert rate == pytest.approx(note_rate, rel=0.005)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e = fluid(b, f, engine=eng).eigenrays(src, pt)
        expected = v_wp(g, b) / np.sqrt(a_rho) * 10 ** (-rate * d / 20) * plain.amplitude[0, 0]
        np.testing.assert_allclose(e.amplitude[0, 0], expected, rtol=1e-11)
        assert e.delay[0, 0] == plain.delay[0, 0]                         # the least-time ray's delay is kept


def test_refusals():
    b = sand()
    with pytest.raises(ValueError, match="attenuation"):                  # no assumed loss (as D-104 for a bottom)
        SedimentProvider(C, 1700.0, 0.0, penetration="fluid", bottom={"speed": 1700.0, "density": 1.8}, frequency=2e4)
    with pytest.raises(ValueError, match="bottom"):
        SedimentProvider(C, 1700.0, 0.0, penetration="fluid", frequency=2e4)
    with pytest.raises(ValueError, match="frequency"):
        SedimentProvider(C, b["speed"], 0.0, penetration="fluid", bottom=b)
    with pytest.raises(ValueError, match="sediment_speed"):
        SedimentProvider(C, 1700.0, 0.0, penetration="fluid", bottom=b, frequency=2e4)
    with pytest.raises(ValueError, match="transmission"):
        SedimentProvider(C, b["speed"], 0.0, transmission=0.5, penetration="fluid", bottom=b, frequency=2e4)
    with pytest.raises(ValueError, match="penetration"):
        SedimentProvider(C, 1700.0, 0.0, penetration="fluidd")
    with pytest.raises(ValueError, match="penetration: fluid"):
        SedimentProvider(C, 1700.0, 0.0, bottom=b)
    with pytest.raises(ValueError, match="did you mean 'attenuation'"):
        SedimentProvider(C, 1700.0, 0.0, penetration="fluid", bottom={"speed": 1700.0, "density": 1.8, "atenuation": 0.8},
                         frequency=2e4)
    with pytest.raises(ValueError, match="did you mean 'penetration'"):
        make_provider("sediment", C, {"sediment_speed": 1700.0, "interface_z": 0.0, "penetraton": "fluid"})


def test_named_sediment_and_speed_from_the_bottom():
    """{sediment: name} resolves through configs/sediments.yaml; sediment_speed may then be left out."""
    p = make_provider("sediment", C, {"interface_z": 0.0, "penetration": "fluid", "bottom": {"sediment": "medium_sand"},
                                      "frequency": 2e4})
    assert p.sediment_speed == pytest.approx(sand()["speed"], rel=1e-15)
    assert p.bottom["density"] == pytest.approx(1.845, rel=1e-12)


# --- the scenario path -------------------------------------------------------------------------

def _small(name, **prop):
    cfg = load_scenario_config(name)
    cfg["platform"]["raster"] = {"x": [0.0, 0.0001, 0.1], "y": [0.0, 0.0001, 0.1]}
    cfg["propagation"].update(prop)
    return cfg


def test_scenario_option_off_is_the_default_and_on_needs_a_bottom():
    with pytest.raises(Exception, match="bottom"):
        run_scenario(_small("buried_cylinder", penetration="fluid"))
    medium = {"speed": 1700.0, "density": 1.845, "attenuation": sand()["attenuation"]}
    off = run_scenario(_small("buried_cylinder")).records[0].samples.array
    on = run_scenario(_small("buried_cylinder", penetration="fluid", bottom=medium)).records[0].samples.array
    # The sonar's band centre (20 kHz) is filled in as the frequency; the echo of the cylinder 0.3 m down at
    # nadir loses about 6 dB two-way of attenuation and gains a little at the interface.
    change = 20 * np.log10(np.abs(on).max() / np.abs(off).max())
    assert -7.0 < change < -5.0


def test_proud_scatterers_of_a_buried_scene_do_not_move():
    """The interface's own scatterers (on z = interface_z, not under it) give bit-identical records with the option
    on: only buried points change."""
    def records(**prop):
        cfg = load_scenario_config("buried_objects")
        cfg["scene"] = [i for i in cfg["scene"] if i["label"] == "interface"]
        cfg["platform"]["raster"] = {"x": [-0.6, 0.6001, 0.6], "y": [0.0, 0.0001, 0.6]}
        cfg["propagation"].update(prop)
        return [np.asarray(r.samples.array) for r in run_scenario(cfg).records]
    off = records()
    on = records(penetration="fluid", bottom={"sediment": "medium_sand", "speed": 1700.0})
    assert len(off) == len(on) == 3
    for a, b in zip(off, on):
        np.testing.assert_array_equal(a, b)
