"""Rock as an elastic solid (docs/references/CONFORMANCE.md A8; docs/tasks/SST_GAPS_JACKSON.md G-4).

The flat-interface reflection coefficient of a fluid over an elastic half-space, Jackson and Richardson,
"High-Frequency Seafloor Acoustics" (2007) [J&R], (9.34)-(9.40) with the losses (9.23), as the note
docs/references/jackson_richardson_2007/03_wave_theories_reflection_penetration.md [N3] Sec. 2.3 gives them
(confirmed there). The formula is retyped below in the book's notation, independently of bottom.py, and checked
a second time against a boundary-condition solve written out here from first principles (ours: continuity of
normal displacement and normal stress, zero shear stress, potentials; the energy fluxes from the same solve).
Numbers marked "printed" are on the book's page as [N3] reads it; "the note's computation" are [N3]'s; "the
audit's" are CONFORMANCE.md A8's (computed by an agent, `rock.py`).

Conventions [N3 2.1]: exp(-i w t); grazing angles; a_rho = rho / rho_w, a_q = nu_q / (1 + i delta_q) (9.23),
q = p (compressional), t (shear); nu_q is the speed ratio to the WATER's sound speed.
"""
import math
import os
import warnings

import numpy as np
import pytest

from gpusonar.sim import bottom
from gpusonar.sim.propagation import ImageSourceProvider, rayleigh_reflection

# The `rock` row's fluid numbers (configs/sediments.yaml: 2557.5 kg/m3, 3820 m/s, delta 0.01374 on 1023 kg/m3,
# 1528 m/s water; UnderwaterAcoustics.jl "Rock").
ROCK = dict(rho=2557.5 / 1023.0, nu=3820.0 / 1528.0, delta=0.01374)
ANGLES = [5.0, 10.0, 20.0, 30.0, 40.0, 45.0, 60.0, 90.0]


def loss_db(v):
    return -20.0 * np.log10(np.abs(v))


# --- J&R (9.34)-(9.40), retyped -------------------------------------------------------------------

def sin_theta(a_q, theta):
    """sin theta_q from cos theta_q = a_q cos theta_w (9.38), the root with Im >= 0 (decaying into the bottom)."""
    s = np.sqrt(1.0 - a_q**2 * np.cos(theta) ** 2 + 0j)
    return np.where(s.imag < 0.0, -s, s)


def v_ww_elastic(theta, a_rho, nu_p, delta_p, nu_t, delta_t):
    a_p, a_t = nu_p / (1.0 + 1j * delta_p), nu_t / (1.0 + 1j * delta_t)        # (9.23)
    z_wp = a_rho * a_p * np.sin(theta) / sin_theta(a_p, theta)                 # (9.36)
    z_wt = a_rho * a_t * np.sin(theta) / sin_theta(a_t, theta)                 # (9.37)
    cos2t = 2.0 * a_t**2 * np.cos(theta) ** 2 - 1.0                            # (9.39)
    sin2t_sq = 1.0 - cos2t**2                                                  # (9.40)
    z_e = z_wp * cos2t**2 + z_wt * sin2t_sq                                    # (9.35)
    return (z_e - 1.0) / (z_e + 1.0)                                           # (9.34)


def solve_interface(theta, a_rho, nu_p, delta_p, nu_t, delta_t):
    """Ours, from first principles: water (z < 0) over a solid (z > 0), z down, potentials
    phi_w = e^{i(xi x + g_w z)} + R e^{i(xi x - g_w z)}, phi = T e^{i(xi x + g_p z)}, psi = S e^{i(xi x + g_s z)},
    u = grad phi + curl(psi y). At z = 0: u_z continuous, sigma_zz = -p continuous, sigma_xz = 0.
    Returns R (the pressure reflection coefficient) and the vertical energy fluxes per unit incident flux of
    the reflected, compressional and shear waves (rho Re(g) |amplitude|^2, normalised)."""
    w, c_w = 1.0, 1.0
    c_p, c_s = c_w * nu_p / (1.0 + 1j * delta_p), c_w * nu_t / (1.0 + 1j * delta_t)
    mu = a_rho * c_s**2
    lam = a_rho * c_p**2 - 2.0 * mu
    xi, g_w = w / c_w * math.cos(theta), w / c_w * math.sin(theta)

    def down(k):
        g = np.sqrt(k**2 - xi**2 + 0j)
        return g if g.imag >= 0.0 else -g

    g_p, g_s = down(w / c_p), down(w / c_s)
    k_w2, k_p2 = (w / c_w) ** 2, (w / c_p) ** 2
    a = np.array([[1j * g_w, 1j * g_p, 1j * xi],
                  [-k_w2, lam * k_p2 + 2.0 * mu * g_p**2, 2.0 * mu * xi * g_s],
                  [0.0, -2.0 * mu * xi * g_p, mu * (g_s**2 - xi**2)]], dtype=complex)
    r, t, s = np.linalg.solve(a, np.array([1j * g_w, k_w2, 0.0], dtype=complex))
    return r, abs(r) ** 2, a_rho * g_p.real * abs(t) ** 2 / g_w, a_rho * g_s.real * abs(s) ** 2 / g_w


# --- The formula --------------------------------------------------------------------------------------------

CASES = [(*ROCK.values(), 1.25, 0.085), (*ROCK.values(), 1900.0 / 1528.0, 0.085), (2.5, 2.5, 0.02, 1.3, 0.1),
         (2.0, 1.1, 0.01, 0.167, 0.01), (2.5, 2.5, 0.0, 1.3, 0.0), (1.845, 1.1782, 0.01624, 0.1, 0.1)]


@pytest.mark.parametrize("case", CASES)
def test_elastic_coefficient_is_9_34_to_9_40(case):
    """Ours (bottom.reflection with nu_t) equal to the retyped (9.34)-(9.40) and to the boundary-condition solve,
    0.05 to 90 degrees (rock: Fig. 9.4's and our row's; Fig. 9.3's sand; lossless rock; sand with a slow shear)."""
    theta = np.radians(np.linspace(0.05, 90.0, 900))
    ours = bottom.reflection(theta, *case)
    assert np.max(np.abs(ours - v_ww_elastic(theta, *case))) < 1e-12
    solved = np.array([solve_interface(t, *case)[0] for t in theta[::9]])
    assert np.max(np.abs(ours[::9] - solved)) < 1e-12


def test_propagation_twin_is_the_same_coefficient():
    """propagation.rayleigh_reflection (m/s and dB per wavelength) is bottom.reflection (ratios)."""
    theta = np.radians(np.linspace(0.0, 90.0, 721))
    c = 1528.0
    for rho, nu, delta, nu_t, delta_t in CASES:
        ours = rayleigh_reflection(theta, c, nu * c, rho, delta * bottom.DB_PER_WAVELENGTH,
                                   shear_speed=nu_t * c, shear_attenuation=delta_t * bottom.DB_PER_WAVELENGTH)
        assert np.max(np.abs(ours - bottom.reflection(theta, rho, nu, delta, nu_t, delta_t))) < 1e-12


def test_zero_shear_is_the_fluid_coefficient():
    """nu_t = 0 is the fluid coefficient bit for bit (the fluid expression itself); as nu_t -> 0 the elastic one
    tends to it (G-4: 1e-12); at normal incidence the shear speed does not matter (cos 2 theta_t = -1)."""
    theta = np.radians(np.linspace(0.0, 90.0, 721))
    fluid = bottom.reflection(theta, **ROCK)
    assert np.array_equal(bottom.reflection(theta, **ROCK, nu_t=0.0, delta_t=0.085), fluid)
    assert np.max(np.abs(bottom.reflection(theta, **ROCK, nu_t=1e-9, delta_t=0.085) - fluid)) < 1e-12
    args = (theta, 1528.0, 3820.0, 2.5, 0.5)
    assert np.array_equal(rayleigh_reflection(*args, shear_speed=0.0, shear_attenuation=0.5), rayleigh_reflection(*args))
    assert np.max(np.abs(rayleigh_reflection(*args, shear_speed=1e-6, shear_attenuation=0.5) - rayleigh_reflection(*args))) < 1e-12
    normal = [bottom.reflection(math.pi / 2, **ROCK, nu_t=nu_t, delta_t=0.085) for nu_t in (0.0, 1.0, 1.25, 1.5)]
    assert np.max(np.abs(np.array(normal) - normal[0])) < 1e-15


def test_lossless_energy_is_conserved():
    """Lossless rock (nu_p 2.5, nu_t 1.25): 1 = |V|^2 + the compressional and shear fluxes (ours, from the
    boundary-condition solve) at every angle; |V| = 1 below both critical angles (all transmitted waves are
    evanescent); above the shear critical angle the shear wave takes energy."""
    case = (2.5, 2.5, 0.0, 1.25, 0.0)
    crit_t = math.degrees(math.acos(1.0 / 1.25))
    for deg in np.linspace(0.5, 89.5, 179):
        r, refl, p, s = solve_interface(math.radians(deg), *case)
        assert refl + p + s == pytest.approx(1.0, abs=1e-12)
        v = bottom.reflection(math.radians(deg), *case)
        assert abs(v) ** 2 == pytest.approx(refl, abs=1e-12)
        if deg < crit_t:
            assert abs(v) == pytest.approx(1.0, abs=1e-12) and s == pytest.approx(0.0, abs=1e-12)
        else:
            assert s > 1e-4
    assert crit_t == pytest.approx(36.87, abs=0.01)


def test_fig_9_4_rock():
    """Fig. 9.4 (p. 275): rock nu_p 2.5, a_rho 2.5, delta_p 0.02, nu_t 1.3, delta_t 0.1 (printed). Critical
    angles arccos(1/2.5) = 66.4 and arccos(1/1.3) = 39.7 degrees (printed); the minimum |V| 0.052 at 35.7 degrees
    (the note's computation)."""
    assert math.degrees(math.acos(1 / 2.5)) == pytest.approx(66.4, abs=0.05)
    assert math.degrees(math.acos(1 / 1.3)) == pytest.approx(39.7, abs=0.05)
    theta = np.radians(np.linspace(20.0, 50.0, 30001))
    v = np.abs(bottom.reflection(theta, 2.5, 2.5, 0.02, 1.3, 0.1))
    assert v.min() == pytest.approx(0.052, abs=0.0005)
    assert math.degrees(theta[v.argmin()]) == pytest.approx(35.7, abs=0.05)


def test_fig_9_3_sand_shear_is_a_small_correction():
    """Fig. 9.3 (p. 275): sand nu_p 1.1, a_rho 2.0, delta_p 0.01, nu_t 0.167 (250 m/s, "rather large"),
    delta_t 0.01 (printed): "a small difference". G-4 asks for under 0.5 dB; ours: at most 0.506 dB, at 36.1
    degrees (the note's own table, [N3] Sec. 3.3, has 0.5 dB at 40 and 45 degrees), so the bound is 0.51 dB."""
    theta = np.radians(np.linspace(0.5, 90.0, 40000))
    diff = loss_db(bottom.reflection(theta, 2.0, 1.1, 0.01, 0.167, 0.01)) - loss_db(bottom.reflection(theta, 2.0, 1.1, 0.01))
    assert np.max(np.abs(diff)) == pytest.approx(0.506, abs=0.001) and np.min(diff) > -1e-12


# --- The numbers the audit and the notes give ---------------------------------------------------------------

def test_audit_per_bounce_losses_are_reproduced():
    """CONFORMANCE.md A8 (the audit's, `rock.py`): our row with V_s = 0.5 V_p (nu_t 1.25) and shear delta 0.085,
    1.55 / 2.86 / 5.29 / 13.68 / 8.96 / 6.27 / 4.52 / 2.80 dB; with delta_t 0.1, 1.81 / 3.35 / 6.13 / 15.43 /
    9.24 / 6.41 / 4.39 / 2.80; at 10 degrees over J&R's 40-60 %: 6.83 / 3.35 / 2.05 dB. As a fluid: 0.01 / 0.02 /
    0.03 / 0.03 / 0.03 / 0.03 / 0.06 / 2.80 dB."""
    g = np.radians(ANGLES)
    np.testing.assert_allclose(loss_db(bottom.reflection(g, **ROCK, nu_t=1.25, delta_t=0.085)),
                               [1.55, 2.86, 5.29, 13.68, 8.96, 6.27, 4.52, 2.80], atol=0.005)
    np.testing.assert_allclose(loss_db(bottom.reflection(g, **ROCK, nu_t=1.25, delta_t=0.1)),
                               [1.81, 3.35, 6.13, 15.43, 9.24, 6.41, 4.39, 2.80], atol=0.005)
    at10 = [loss_db(bottom.reflection(math.radians(10.0), **ROCK, nu_t=r * 2.5, delta_t=0.1)) for r in (0.4, 0.5, 0.6)]
    np.testing.assert_allclose(at10, [6.83, 3.35, 2.05], atol=0.005)
    np.testing.assert_allclose(loss_db(bottom.reflection(g, **ROCK)), [0.01, 0.02, 0.03, 0.03, 0.03, 0.03, 0.06, 2.80],
                               atol=0.005)
    # The audit's "peak 16.7 dB at 35.7 degrees" is the loss AT 35.7 degrees (Fig. 9.4's angle); this case's
    # largest loss is 28.8 dB at 33.4 degrees (ours).
    assert loss_db(bottom.reflection(math.radians(35.7), **ROCK, nu_t=1.25, delta_t=0.085)) == pytest.approx(16.67, abs=0.01)
    theta = np.radians(np.linspace(30.0, 40.0, 100001))
    lo = loss_db(bottom.reflection(theta, **ROCK, nu_t=1.25, delta_t=0.085))
    assert lo.max() == pytest.approx(28.76, abs=0.01) and math.degrees(theta[lo.argmax()]) == pytest.approx(33.43, abs=0.01)


def test_note_rock_cases_are_reproduced():
    """[N3] Sec. 3.3 (the note's computation, 0.1 dB): Fig. 9.4 rock; our rock at V_s 0.4 / 0.5 / 0.6 V_p with
    delta_t 0.1; Olson et al.'s "generic rock" (3600 m/s, 1900 m/s, 2500 kg/m3, delta 0.0018 / 0.085) on our
    1528 m/s, 1023 kg/m3 water. Settles C1 (SST_GAPS_JACKSON.md): [N3]'s numbers follow from the stated inputs."""
    g = np.radians(ANGLES)
    rows = {(2.5, 2.5, 0.02, 1.3, 0.1): [1.8, 3.1, 5.0, 10.1, 11.5, 6.8, 4.3, 2.8],
            (*ROCK.values(), 1.0, 0.1): [3.1, 6.8, 18.8, 11.7, 7.9, 7.1, 3.6, 2.8],
            (*ROCK.values(), 1.25, 0.1): [1.8, 3.3, 6.1, 15.4, 9.2, 6.4, 4.4, 2.8],
            (*ROCK.values(), 1.5, 0.1): [1.4, 2.0, 2.4, 3.2, 8.9, 17.3, 3.8, 2.8],
            (2500.0 / 1023.0, 3600.0 / 1528.0, 0.0018, 1900.0 / 1528.0, 0.085): [1.4, 2.6, 5.1, 13.7, 9.0, 6.4, 4.5, 3.0]}
    for case, printed in rows.items():
        np.testing.assert_allclose(loss_db(bottom.reflection(g, *case)), printed, atol=0.051)


# --- The rock row -------------------------------------------------------------------------------------------

def test_rock_row_is_an_elastic_solid():
    """configs/sediments.yaml `rock`: shear 1900 m/s and shear delta 0.085 (Olson et al.'s generic rock, via the
    file's header), flagged unconfirmed: a warning names `shear`. fluid_halfspace stays the fluid (the sediment tier
    and the live chain take it); halfspace adds the shear. `pebbles` stays a fluid (no source for its shear)."""
    with pytest.warns(UserWarning, match="shear"):
        par = bottom.sediment_reflection("rock")
    assert par == pytest.approx(dict(ROCK, nu_t=1900.0 / 1528.0, delta_t=0.085), rel=1e-15)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert bottom.sediment_reflection("rock", shear=False) == pytest.approx(ROCK, rel=1e-15)
        fluid = bottom.fluid_halfspace("rock", 1500.0)
    assert set(fluid) == {"speed", "density", "attenuation"}
    with pytest.warns(UserWarning, match="shear"):
        half = bottom.halfspace("rock", 1500.0)
    assert half == pytest.approx(dict(fluid, shear_speed=1900.0 / 1528.0 * 1500.0,
                                      shear_attenuation=0.085 * bottom.DB_PER_WAVELENGTH), rel=1e-15)
    assert 0.4 <= 1900.0 / 3820.0 <= 0.6                                  # J&R Sect. 5.6, p. 169: 40-60 % (printed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert "nu_t" not in bottom.sediment_reflection("pebbles")
        assert bottom.halfspace("pebbles", 1500.0) == bottom.fluid_halfspace("pebbles", 1500.0)
        # Per bounce, the row as it is now (ours): before (fluid) -> after (elastic).
        g = np.radians(ANGLES)
        np.testing.assert_allclose(loss_db(bottom.reflection(g, **bottom.sediment_reflection("rock"))),
                                   [1.56, 2.90, 5.45, 14.57, 8.78, 6.25, 4.54, 2.80], atol=0.005)


def test_fluid_rows_are_unchanged():
    """Every row without shear: sediment_reflection, halfspace and the coefficient are what they were (the
    sha256 of each before and after the change is in the A8 report; here: no shear keys, halfspace is
    fluid_halfspace)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in bottom.sediment_table():
            try:
                par = bottom.sediment_reflection(name)
            except ValueError:
                continue
            if name == "rock":
                continue
            assert set(par) == set(bottom.REFLECTION_KEYS), name
            assert bottom.halfspace(name, 1528.0) == bottom.fluid_halfspace(name, 1528.0), name


# --- The image-source tier: both engines --------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["numpy", "cpp"])
def test_image_source_rock_bounce_is_elastic(engine):
    """{sediment: rock}: the one-bottom-bounce path carries the elastic coefficient at its grazing angle;
    {sediment: rock, shear_speed: 0} is the fluid row as before."""
    c, src, pt = 1528.0, np.array([0.0, 0.0, 10.0]), np.array([[60.0, 0.0, 20.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        el = ImageSourceProvider(c, surface_z=None, bottom_z=30.0, max_order=1, bottom={"sediment": "rock"}, engine=engine)
        fl = ImageSourceProvider(c, surface_z=None, bottom_z=30.0, max_order=1, bottom={"sediment": "rock", "shear_speed": 0.0},
                                 engine=engine)
        e, f = el.eigenrays(src, pt), fl.eigenrays(src, pt)
        par = bottom.sediment_reflection("rock")
    j = int(np.flatnonzero(e.n_bottom[0] == 1)[0])
    length = np.hypot(60.0, 20.0 + 10.0)
    grazing = math.atan2(30.0, 60.0)
    assert e.amplitude[0, j] * length == pytest.approx(complex(bottom.reflection(grazing, **par)), abs=1e-12)
    assert f.amplitude[0, j] * length == pytest.approx(complex(bottom.reflection(grazing, **ROCK)), abs=1e-12)
    assert abs(e.amplitude[0, j]) < 0.85 * abs(f.amplitude[0, j])
    with pytest.raises(ValueError, match="shear"):
        ImageSourceProvider(c, surface_z=None, bottom_z=30.0, max_order=1, bottom=dict(speed=3820.0, density=2.5, shear_speed=-1.0),
                            engine=engine).eigenrays(src, pt)


# --- Bellhop: the written file, and Bellhop itself (a separate GPL program) -----------------------------------

def test_bellhop_writes_the_shear_column(tmp_path):
    """An elastic bottom writes depth, cp, cs, rho, alpha_p, alpha_s (dB per wavelength, 'CVW'); a fluid writes
    the line as before (shear speed 0.0, no shear attenuation); `elastic=False` keeps the rock row a fluid."""
    from gpusonar.sim.bellhop import OceanEnvironment, write_env
    from gpusonar.sim.bellhop3d import write_env3d

    def line(env):
        write_env(str(tmp_path / "x.env"), env, 8000.0, 10.0, [20.0], [100.0])
        text = (tmp_path / "x.env").read_text().splitlines()
        return text[text.index("'A' 0.0") + 1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rock = OceanEnvironment(depth=50.0, sound_speed=1528.0, sediment="rock")
        rock_fluid = OceanEnvironment(depth=50.0, sound_speed=1528.0, sediment="rock", elastic=False)
        sand = OceanEnvironment(depth=50.0, sound_speed=1528.0, sediment="medium_sand")
    ap, at = 0.01374 * bottom.DB_PER_WAVELENGTH, 0.085 * bottom.DB_PER_WAVELENGTH
    assert line(rock) == f" 50.0000  3820.0000 1900.0000 2.5000 {ap:.4f} {at:.4f} /"
    assert line(rock_fluid) == f" 50.0000  3820.0000 0.0 2.5000 {ap:.4f} /"
    assert line(sand).split()[2] == "0.0" and len(line(sand).split()) == 6
    explicit = OceanEnvironment(depth=50.0, sound_speed=1500.0, bottom_speed=3600.0, bottom_density=2.5,
                                bottom_attenuation=0.1, bottom_shear_speed=1900.0, bottom_shear_attenuation=2.5)
    assert line(explicit) == " 50.0000  3600.0000 1900.0000 2.5000 0.1000 2.5000 /"
    with pytest.raises(ValueError, match="not both"):
        OceanEnvironment(depth=50.0, sound_speed=1528.0, sediment="rock", bottom_shear_speed=1900.0)
    kw = dict(frequency=8000.0, profile=([0.0, 50.0], [1500.0, 1500.0]), depth=50.0, source_depth=10.0, receiver_depths=[20.0],
              ranges=[100.0], bearings_deg=[0.0], run_type="a", mode="3", n_alpha=11, alpha_deg=(-10.0, 10.0), n_beta=3,
              beta_deg=(-1.0, 1.0), box_km=(1.0, 1.0))
    write_env3d(str(tmp_path / "y.env"), bottom=(3600.0, 2.5, 0.1, 1900.0, 2.5), **kw)
    assert " 50.0000  3600.0000 1900.0000 2.5000 0.1000 2.5000 /" in (tmp_path / "y.env").read_text().splitlines()
    write_env3d(str(tmp_path / "y.env"), bottom=(3600.0, 2.5, 0.1), **kw)
    assert " 50.0000  3600.0000 0.0 2.5000 0.1000 /" in (tmp_path / "y.env").read_text().splitlines()


@pytest.mark.skipif(not os.environ.get("GPUSONAR_BELLHOP"), reason="Bellhop not installed (GPUSONAR_BELLHOP)")
def test_bellhop_reflects_off_rock_as_ours():
    """Bellhop (GPL, run as a separate process) over the elastic rock row, isovelocity 1528 m/s water 200 m deep,
    source and receiver at 100 m, 2 kHz: the one-bottom-bounce arrival's amplitude x path length and phase against
    our coefficient at its grazing angle. Measured (2026-11-26): within 0.22 dB (20 degrees; 0.05 dB or less at 5,
    10, 45, 60) and 2.6 degrees; at 30 degrees 1.2 dB and 6.2 degrees (|V| falls 15 dB between 30 and 33 degrees there, and
    Bellhop's beams average over angle). The fluid row (elastic=False) agrees within 0.01 dB everywhere."""
    from gpusonar.sim.bellhop import OceanEnvironment, run_bellhop

    angles = (60.0, 45.0, 30.0, 20.0, 10.0, 5.0)
    ranges = np.array([200.0 / math.tan(math.radians(d)) for d in angles])       # increasing, as Bellhop takes them
    for elastic in (True, False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = OceanEnvironment(depth=200.0, sound_speed=1528.0, sediment="rock", elastic=elastic)
            par = bottom.sediment_reflection("rock", shear=elastic)
        arr = run_bellhop(env, 2000.0, 100.0, [100.0], ranges, n_beams=4001, angles=(-89.0, 89.0))
        for j, (deg, r) in enumerate(zip(angles, ranges)):
            pick = np.flatnonzero((arr.n_surface[0][j] == 0) & (arr.n_bottom[0][j] == 1))
            assert len(pick) == 1
            amp = arr.amplitude[0][j][pick[0]] * math.hypot(r, 200.0)
            ours = complex(bottom.reflection(math.radians(deg), **par))
            db, phase = abs(loss_db(amp) - loss_db(ours)), abs(math.degrees(np.angle(amp / ours)))
            if not elastic:
                assert db < 0.01 and phase < 0.5, (deg, db, phase)
            elif deg == 30.0:
                assert db < 1.5 and phase < 8.0, (deg, db, phase)
            else:
                assert db < 0.3 and phase < 3.0, (deg, db, phase)
