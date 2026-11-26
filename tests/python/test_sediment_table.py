"""The named sediments (configs/sediments.yaml): every row loads, density, speed and loss are
those of UnderwaterAcoustics.jl (with one corrected typing error), the two rows that are not
fluids (rock, pebbles) serve the reflection coefficient and are refused by the scattering
models, and the two rows from arXiv:1809.08459 agree with that table. The scattering
parameters of the other rows are checked in test_sediment_literature.py."""

import cmath
import math
import warnings

import numpy as np
import pytest

from gpusonar.sim import bottom, build_scene, find_bellhop
from gpusonar.sim.bellhop import BellhopProvider, OceanEnvironment
from gpusonar.sim.propagation import ImageSourceProvider, make_provider, rayleigh_reflection

# UnderwaterAcoustics.jl src/stdlib.jl:115-136 typed in again here, independently of the YAML:
# name, density kg/m3, sound speed m/s, loss parameter. Reference water 1023 kg/m3, 1528 m/s.
JULIA_TABLE = {
    "rock": (2557.50, 3820.00, 0.01374), "pebbles": (2557.50, 2750.40, 0.01374),
    "sandy_gravel": (2549.32, 2073.50, 0.01705), "very_coarse_sand": (2455.20, 1996.64, 0.01667),
    "muddy_sandy_gravel": (2367.22, 1952.48, 0.0163), "coarse_sand": (2282.31, 1910.46, 0.01638),
    "gravelly_muddy_sand": (2200.47, 1870.42, 0.01645), "medium_sand": (1887.44, 1800.29, 0.01624),
    "muddy_gravel": (1652.15, 1741.31, 0.0161), "fine_sand": (1484.37, 1691.95, 0.01602),
    "muddy_sand": (1369.80, 1650.24, 0.01728), "very_fine_sand": (1297.16, 1614.79, 0.01875),
    "clayey_sand": (1252.15, 1583.62, 0.02019), "coarse_silt": (1222.49, 1555.35, 0.02158),
    "sandy_silt": (1195.89, 1527.85, 0.01261), "medium_silt": (1175.43, 1510.43, 0.00676),
    "sandy_mud": (1175.43, 1508.59, 0.00386), "fine_silt": (1174.40, 1506.76, 0.00306),
    "sandy_clay": (1173.38, 1504.93, 0.00242), "very_fine_silt": (1173.38, 1503.09, 0.00194),
    "silty_clay": (1172.36, 1501.11, 0.00163), "clay": (1171.34, 1497.44, 0.00148),
}
FULL_ROWS = ("medium_sand", "very_fine_silt")          # rows whose rho, nu, delta come from arXiv:1809.08459
JULIA_ROWS = sorted(set(JULIA_TABLE) - set(FULL_ROWS))  # rows whose rho, nu, delta come from UnderwaterAcoustics.jl
REFLECTION_ONLY = ["pebbles", "rock"]                  # not fluid sediments: no w2, gamma2, sigma2
# The one place the file departs from the Julia table: its line for SandyGravel prints 2073.50 m/s
# (ratio 1.3570); three other prints of the row give 1.337 (the entry's note), which is carried.
SANDY_GRAVEL_SPEED = 1.337 * 1528.0
# Named site rows (A6 of docs/references/CONFORMANCE.md Sec. 1.1): one measured site each, never a class.
# Jackson and Briggs (1992) Table V and Williams and Jackson (1998) Table I: the notes
# docs/references/bottom_backscatter_jackson_briggs_1992.md ([JB]) 5.1 and
# docs/references/bottom_bistatic_williams_jackson_1998.md ([WJ]) 5.5.
JB_ROWS = ("jackson_briggs_1992_quinault", "jackson_briggs_1992_arafura", "jackson_briggs_1992_san_francisco")
WJ_ROWS = ("williams_jackson_1998_panama_city",)
SITE_ROWS = ("gallaudet_2001_site",) + JB_ROWS + WJ_ROWS
# The seven rows of 5.5 phi and finer (docs/tasks/SEDIMENT_DEFAULTS_DECISION.md, option E) and their class
# volume parameter, unchanged by A6 (very_fine_silt: arXiv:1809.08459's -28.6 dB/m over 1.4 dB/m).
SOFT_ROWS = {"medium_silt": 0.001, "sandy_mud": 0.001, "fine_silt": 0.001, "sandy_clay": 0.001,
             "very_fine_silt": 10.0 ** (-28.6 / 10.0) / 1.4, "silty_clay": 0.001, "clay": 0.001}


def quiet(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kw)


def test_every_row_loads_and_the_table_is_complete():
    table = bottom.sediment_table()
    assert set(table) == set(JULIA_TABLE) | {"placeholder_sand"} | set(SITE_ROWS)
    for name, entry in table.items():
        par = quiet(bottom.sediment_reflection, name)
        # rock is an elastic solid (A8; tests/python/test_elastic_bottom.py): also its shear ratio and loss
        assert set(par) == {"rho", "nu", "delta"} | ({"nu_t", "delta_t"} if name == "rock" else set())
        assert all(math.isfinite(v) and v > 0 for v in par.values())
        assert bottom.is_reflection_only(name) == (name in REFLECTION_ONLY) == bool(entry.get("reflection_only", False))
        assert "source" in entry and "status" in entry
    for name in JULIA_ROWS:
        rho, c, delta = JULIA_TABLE[name]
        c = SANDY_GRAVEL_SPEED if name == "sandy_gravel" else c
        par = quiet(bottom.sediment_reflection, name)
        assert par["rho"] == pytest.approx(rho / 1023.0, rel=1e-12)
        assert par["nu"] == pytest.approx(c / 1528.0, rel=3e-6)
        assert par["delta"] == delta
        assert "stdlib.jl" in table[name]["source"] and "TR 9407" in table[name]["source"]


def test_confirmed_rows_do_not_warn_and_the_corrected_speed_is_documented():
    for name in JULIA_ROWS:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            bottom.sediment_reflection(name, shear=False)    # rock's shear (A8) is flagged unconfirmed and warns
    par = bottom.sediment_reflection("sandy_gravel")
    entry = bottom.sediment_table()["sandy_gravel"]
    # The Julia table's 1.3570 was a typing error: both values and the three sources are in the note.
    assert "unconfirmed_fields" not in entry and "1.3370" in entry["note"] and "1.3570" in entry["note"]
    assert "Cintron" in entry["note"] and "Snellen" in entry["note"]
    assert par["nu"] == pytest.approx(1.3370, abs=5e-6)


def test_one_convention_the_two_sources_give_the_same_ratios():
    """The complete rows come from arXiv:1809.08459 (kg/m3 and m/s, ratios formed with 1000 and
    1500); UnderwaterAcoustics.jl prints the same sediments for 1023 and 1528. The ratios agree
    to the digits printed, so both stand for one table of ratios, and the file holds ratios."""
    for name, rho_tol, delta_tol in (("medium_sand", 5e-4, 0.004), ("very_fine_silt", 5e-4, 0.025)):
        rho, c, delta = JULIA_TABLE[name]
        ours = bottom.sediment(name)
        assert ours["rho"] == pytest.approx(rho / 1023.0, abs=rho_tol)
        assert ours["nu"] == pytest.approx(c / 1528.0, abs=5e-4)
        assert ours["delta"] == pytest.approx(delta, rel=delta_tol)
    # Had the arXiv paper's kg/m3 been for 1023 kg/m3 water, the density ratio would be off by 2.3 %.
    assert abs(1845.0 / 1023.0 - 1887.44 / 1023.0) > 0.04


def hand_rayleigh(grazing, rho, nu, delta):
    """R = (rho sin g - P) / (rho sin g + P), P = sqrt(kappa^2 - cos^2 g), kappa = (1 + i delta) / nu,
    written out with scalar complex arithmetic (PHYSICS_NOTES 6.4)."""
    kappa = (1.0 + 1j * delta) / nu
    p = cmath.sqrt(kappa * kappa - math.cos(grazing) ** 2)
    p = -p if p.imag < 0 else p
    return (rho * math.sin(grazing) - p) / (rho * math.sin(grazing) + p)


@pytest.mark.parametrize("name", ["rock", "coarse_sand", "fine_sand", "sandy_silt", "clay"])
def test_reflection_of_named_rows_against_a_hand_evaluated_coefficient(name):
    rho, c, delta = JULIA_TABLE[name]
    rho, nu = rho / 1023.0, c / 1528.0
    par = bottom.sediment_reflection(name, shear=False)      # rock as a fluid (A8 made it elastic by default)
    half = bottom.fluid_halfspace(name, 1500.0)
    assert half["speed"] == pytest.approx(nu * 1500.0) and half["density"] == pytest.approx(rho)
    assert half["attenuation"] == pytest.approx(delta * 40.0 * math.pi / math.log(10.0))
    for deg in (2.0, 10.0, 25.0, 45.0, 70.0, 90.0):
        g = math.radians(deg)
        want = hand_rayleigh(g, rho, nu, delta)
        assert complex(bottom.reflection(g, **par)) == pytest.approx(want, abs=1e-12)
        assert complex(rayleigh_reflection(g, 1500.0, **half)) == pytest.approx(want, abs=1e-12)
    # Limits with known answers: normal incidence is the impedance contrast (no loss: delta -> 0),
    # and below the critical angle of a fast lossless bottom the reflection is total.
    z = rho * nu
    assert abs(hand_rayleigh(math.pi / 2, rho, nu, 0.0) - (z - 1.0) / (z + 1.0)) < 1e-12
    if nu > 1.0:
        assert abs(hand_rayleigh(0.5 * math.acos(1.0 / nu), rho, nu, 0.0)) == pytest.approx(1.0, abs=1e-12)


def test_reflection_loss_orders_the_sediments_as_expected():
    """At 60 degrees grazing (above every critical angle but rock's) the loss grows from gravel to clay."""
    g = math.radians(60.0)
    loss = {n: -20 * math.log10(abs(complex(bottom.reflection(g, **quiet(bottom.sediment_reflection, n)))))
            for n in ("sandy_gravel", "coarse_sand", "medium_sand", "fine_sand", "coarse_silt", "clay")}
    values = list(loss.values())
    assert values == sorted(values) and values[0] < 6.0 and values[-1] > 20.0


def test_scattering_models_refuse_reflection_only_rows():
    with pytest.raises(ValueError, match=r"rock.*reflection-only.*\['w2', 'gamma2', 'sigma2'\].*Lambert"):
        bottom.sediment("rock")
    # The scenario item may supply what the table lacks.
    got = bottom.sediment("rock", supplied={"w2": 1e-5, "gamma2": 3.0, "sigma2": 1e-3})
    assert set(got) == {"rho", "nu", "delta"}
    with pytest.raises(ValueError, match=r"reflection-only.*\['sigma2'\]"):
        bottom.sediment("rock", supplied={"w2": 1e-5, "gamma2": 3.0})
    seafloor = {"type": "seafloor", "x_range": [2.0, 6.0], "y_range": [-1.0, 1.0], "density": 20.0,
                "model": "fluid_sediment", "sediment": "rock"}
    with pytest.raises(ValueError, match="reflection-only"):
        build_scene([seafloor], np.random.default_rng(0))
    scene = build_scene([dict(seafloor, w2=1e-5, gamma2=3.0, sigma2=1e-3)], np.random.default_rng(0))
    params = scene.modifiers[0][2].params
    assert params["rho"] == pytest.approx(2557.50 / 1023.0) and params["w2"] == 1e-5
    # A complete row needs nothing more, by name: the rows from the arXiv paper and the rows
    # whose scattering parameters follow the grain size.
    for name in ("medium_sand", "fine_sand", "clay"):
        scene = build_scene([dict(seafloor, sediment=name)], np.random.default_rng(0))
        assert scene.modifiers[0][2].params == bottom.sediment(name)
    # An item's own key still wins over the table's.
    scene = build_scene([dict(seafloor, sediment="fine_sand", sigma2=5e-4)], np.random.default_rng(0))
    assert scene.modifiers[0][2].params["sigma2"] == 5e-4
    assert scene.modifiers[0][2].params["w2"] == bottom.sediment("fine_sand")["w2"]


def test_reverberation_item_refuses_a_reflection_only_row():
    from gpusonar.sim import run_scenario
    from gpusonar.config import load_scenario_config
    cfg = load_scenario_config("point_grid_fls_300k")
    cfg["reverberation"] = [{"model": "fluid_sediment", "boundary_z": 20.0, "sediment": "pebbles"}]
    with pytest.raises(ValueError, match=r"pebbles.*reflection-only.*w2"):
        run_scenario(cfg)


@pytest.mark.parametrize("engine", ["cpp", "numpy"])
def test_image_source_bottom_by_name_equals_the_numbers(engine):
    c = 1490.0
    par = bottom.sediment_reflection("coarse_sand")
    explicit = {"speed": par["nu"] * c, "density": par["rho"], "attenuation": par["delta"] * 40 * math.pi / math.log(10.0)}
    named = make_provider("image_source", c, {"surface_z": 0.0, "bottom_z": 60.0, "max_order": 4,
                                              "bottom": {"sediment": "coarse_sand"}, "engine": engine})
    given = ImageSourceProvider(c, surface_z=0.0, bottom_z=60.0, max_order=4, bottom=explicit, engine=engine)
    assert named.bottom == pytest.approx(explicit)
    pts = np.array([[300.0, 0.0, 30.0], [800.0, 50.0, 55.0], [40.0, -5.0, 10.0]])
    a, b = named.eigenrays(np.array([0.0, 0.0, 20.0]), pts), given.eigenrays(np.array([0.0, 0.0, 20.0]), pts)
    assert np.array_equal(a.amplitude, b.amplitude) and np.array_equal(a.delay, b.delay)
    assert np.any(a.n_bottom > 0)
    # A key beside the name wins.
    louder = ImageSourceProvider(c, bottom_z=60.0, bottom={"sediment": "coarse_sand", "attenuation": 0.0})
    assert louder.bottom["attenuation"] == 0.0 and louder.bottom["speed"] == pytest.approx(explicit["speed"])


def test_bellhop_environment_by_name():
    par = bottom.sediment_reflection("fine_sand")
    env = OceanEnvironment(depth=80.0, sound_speed=1510.0, sediment="fine_sand")
    assert env.bottom_speed == pytest.approx(par["nu"] * 1510.0) and env.bottom_density == pytest.approx(par["rho"])
    assert env.bottom_attenuation == pytest.approx(par["delta"] * bottom.DB_PER_WAVELENGTH)
    assert env.sediment_name == "fine_sand"
    # The speed ratio applies to the water at the bottom of a profile.
    env = OceanEnvironment(depth=100.0, ssp_depth=np.array([0.0, 100.0]), ssp_speed=np.array([1520.0, 1480.0]), sediment="clay")
    assert env.bottom_speed == pytest.approx(1497.44 / 1528.0 * 1480.0)
    with pytest.raises(ValueError, match="not both"):
        OceanEnvironment(depth=80.0, sediment="fine_sand", bottom_speed=1700.0)
    with pytest.raises(ValueError, match="no seabed is assumed"):    # no default seabed (D-146)
        OceanEnvironment(depth=80.0)


@pytest.mark.skipif(find_bellhop() is None, reason="Bellhop not installed (GPUSONAR_BELLHOP)")
def test_bellhop_run_with_a_named_sediment_equals_the_explicit_one():
    par = bottom.sediment_reflection("muddy_sand")
    named = OceanEnvironment(depth=60.0, sound_speed=1500.0, sediment="muddy_sand")
    given = OceanEnvironment(depth=60.0, sound_speed=1500.0, bottom_speed=par["nu"] * 1500.0, bottom_density=par["rho"],
                             bottom_attenuation=par["delta"] * bottom.DB_PER_WAVELENGTH)
    src = np.array([0.0, 0.0, 20.0])
    pts = np.array([[400.0, 0.0, 30.0], [900.0, 0.0, 45.0]])
    out = [BellhopProvider(env, frequency=7000.0, max_range=1000.0).eigenrays(src, pts) for env in (named, given)]
    assert np.array_equal(out[0].amplitude, out[1].amplitude) and np.array_equal(out[0].delay, out[1].delay)
    assert np.any(out[0].n_bottom[out[0].valid] > 0)
    # along_profile (dataclasses.replace) keeps the resolved numbers.
    relief = BellhopProvider(named, frequency=7000.0, max_range=1000.0).along_profile([[0.0, 60.0], [1000.0, 50.0]])
    assert relief.env.bottom_speed == named.bottom_speed and relief.env.bottom_density == named.bottom_density


# Lurton, Lamarche et al. (GeoHab Backscatter Working Group, 2015; doi:10.5281/zenodo.10089261),
# Table 2-1, typed in: phi range, density kg/m3, velocity m/s, absorption dB per wavelength, for
# "nominal values for c1 and rho1 (1500 m/s and 1000 kg/m3)".
GEOHAB_TABLE = {
    "clay": ((8.0, None), 1145.0, 1470.0, 0.08), "very_fine_silt": ((7.0, 8.0), 1147.0, 1476.0, 0.11),
    "fine_silt": ((6.0, 7.0), 1148.0, 1479.0, 0.17), "medium_silt": ((5.0, 6.0), 1149.0, 1482.0, 0.37),
    "coarse_silt": ((4.0, 5.0), 1195.0, 1523.0, 1.18), "very_fine_sand": ((3.0, 4.0), 1268.0, 1585.0, 1.02),
    "fine_sand": ((2.0, 3.0), 1451.0, 1661.0, 0.87), "medium_sand": ((1.0, 2.0), 1845.0, 1767.0, 0.89),
    "coarse_sand": ((0.0, 1.0), 2231.0, 1875.0, 0.89),
}


def test_a_second_open_source_agrees_on_nine_rows():
    """Density, speed and loss of nine rows against the GeoHab report's table, which states its
    reference water (1000 kg/m3, 1500 m/s): the ratios are the same to the printed digits, so
    the file's single convention (ratios) holds across all three sources. Coarse silt's speed
    is the one exception (0.26 %), recorded in the entry's note."""
    table = bottom.sediment_table()
    for name, (phi, density, speed, db_per_wavelength) in GEOHAB_TABLE.items():
        par = quiet(bottom.sediment_reflection, name)
        assert par["rho"] == pytest.approx(density / 1000.0, abs=6e-4), name
        tolerance = 3e-3 if name == "coarse_silt" else 6e-4
        assert par["nu"] == pytest.approx(speed / 1500.0, abs=tolerance), name
        assert par["delta"] * bottom.DB_PER_WAVELENGTH == pytest.approx(db_per_wavelength, abs=0.01), name
        assert tuple(table[name]["grain_size_phi"]) == phi
    assert abs(1555.35 / 1528.0 - 1523.0 / 1500.0) > 2e-3 and "1.0153" in table["coarse_silt"]["note"]
    # Finer sediment, larger phi: the ranges tile 0 to 8 in the order of the table.
    lows = [table[n]["grain_size_phi"][0] for n in ("coarse_sand", "medium_sand", "fine_sand", "very_fine_sand",
                                                    "coarse_silt", "medium_silt", "fine_silt", "very_fine_silt", "clay")]
    assert lows == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


def test_the_soft_rows_keep_their_class_values_and_carry_the_measured_range():
    """A6 (docs/references/CONFORMANCE.md Sec. 1.1; docs/tasks/SEDIMENT_DEFAULTS_DECISION.md option E): the
    seven rows of 5.5 phi and finer keep sigma2 0.001 (0.000986 for very_fine_silt) and the w2 rule, and gain
    `sigma2_range: [0.0008, 0.0037]` (the five J&R Table 14.1 sites by (14.10), an agents' computation,
    [N2] Sec. 3.3: 0.0031, 0.0037, 0.0008, 0.0017, 0.0016; and Gallaudet's fitted 0.003, printed, one site)
    with a `basis:` and a `w2_note:` (Briggs' spectra, J&R Table 6.1, +7.1 dB at 10 kHz and +2.2 dB at
    100 kHz above the rule, [N2] Sec. 3.2, ours). The loader carries the keys and does not use them."""
    table = bottom.sediment_table()
    for name, sigma2 in SOFT_ROWS.items():
        entry = table[name]
        assert entry["sigma2_range"] == [0.0008, 0.0037], name
        low, high = entry["sigma2_range"]
        assert low <= sigma2 <= high and low <= 0.003 <= high
        for site, value in (("Arafura", "0.0031"), ("San Francisco", "0.0037"), ("Orcas", "0.0008"),
                            ("Porto Venere", "0.0017"), ("Mariella", "0.0016"), ("Gallaudet", "0.003")):
            assert site in entry["basis"] and value in entry["basis"], (name, site)
        assert "Table 14.1" in entry["basis"] and "not independent" in entry["basis"] and "volume dominance" in entry["basis"]
        assert "Table 6.1" in entry["w2_note"] and "+7.1 dB" in entry["w2_note"] and "+2.2 dB" in entry["w2_note"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")                              # confirmed rows still do not warn
            par = bottom.sediment(name)
        assert par["sigma2"] == pytest.approx(sigma2, rel=1e-12) and par["gamma2"] == 3.25
        assert par["w2"] == pytest.approx(bottom.grain_size_scattering(entry["mean_grain_size_phi"])["w2"], rel=1e-3)
        assert set(par) == set(bottom.SEDIMENT_KEYS)                    # the six parameters only: the range is not used
        assert bottom._entry(name)["sigma2_range"] == [0.0008, 0.0037]  # but it is carried
    # No other row claims the soft rows' range.
    assert {n for n, e in table.items() if "sigma2_range" in e} == set(SOFT_ROWS)


def test_the_gallaudet_site_row_is_his_table_vi():
    """`gallaudet_2001_site`: Gallaudet (2001) Ch. 3 Table VI, p. 60 (printed; one site, NE Gulf of Mexico,
    68 kHz), the inputs of the composite model he fitted to measured backscatter, and his Mz 6.36 ("estimated
    from porosity", Table I(e), p. 55). Equal to the copy typed in test_gallaudet_boundaries.py. Its 0.003 is
    one fitted site, not a class value: the row says why it is not independent of the volume-dominance
    assumption."""
    from test_gallaudet_boundaries import BOTTOM_FIT, BOTTOM_FIT_PRINTED, MEAN_GRAIN_SIZE_PHI
    entry = bottom.sediment_table()["gallaudet_2001_site"]
    assert entry["status"] == "confirmed" and "Table VI" in entry["source"] and "Gallaudet" in entry["source"]
    assert entry["mean_grain_size_phi"] == MEAN_GRAIN_SIZE_PHI and "porosity" in entry["provenance"]["mean_grain_size_phi"]
    assert entry["scattering"]["w2_cm4"] == BOTTOM_FIT_PRINTED["w2_cm4"]
    assert "volume dominance" in entry["note"] and "one fitted site" in entry["note"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        par = bottom.sediment("gallaudet_2001_site")
    assert par == pytest.approx(BOTTOM_FIT, rel=1e-12)
    assert not bottom.is_reflection_only("gallaudet_2001_site")
    assert bottom.sediment_reflection("gallaudet_2001_site") == {k: BOTTOM_FIT[k] for k in ("rho", "nu", "delta")}


# --- The Jackson and Briggs (1992) and Williams and Jackson (1998) site rows ---------------------------------------
# Typed in again here from the notes, independently of the YAML ([JB] Sec. 2.1; [WJ] Sec. 2.1; confirmed, as printed).
# [JB] Table V (p. 973): nu, rho, delta, sigma2, gamma2 (along, across strike), w2 in cm^4 (along, across strike).
JB_TABLE_V = {
    "quinault": (1.113, 1.94, 0.0115, 0.001, (3.92, 3.67), (0.00598, 0.00422)),
    "arafura": (0.988, 1.39, 0.00553, 0.005, (3.18, 3.18), (0.00318, 0.00318)),
    "san_francisco_pre_storm": (1.002, 1.41, 0.0104, 0.002, (3.65, 3.38), (0.000846, 0.000912)),
    "san_francisco_post_storm": (1.002, 1.41, 0.0104, 0.002, (3.73, 3.56), (0.000826, 0.001296)),
}
# [JB] Tables II (surficial 0-2 cm, used) and I (whole core) mean grain size, phi.
JB_MZ = {"quinault": (2.97, 2.94), "arafura": (5.63, 5.24), "san_francisco": (6.39, 6.35)}
# Which Table V values each row carries ([JB] 5.1): Quinault across strike (Fig. 10; the along-strike exponent is
# called unreliable, p. 974), Arafura (one spectrum), San Francisco pre-storm along strike (Fig. 15).
JB_CHOICE = {"jackson_briggs_1992_quinault": ("quinault", 1, "quinault"),
             "jackson_briggs_1992_arafura": ("arafura", 0, "arafura"),
             "jackson_briggs_1992_san_francisco": ("san_francisco_pre_storm", 0, "san_francisco")}
# [WJ] Table I (p. 171), Panama City: rho, nu, delta, gamma3, w3 (cm^3), mu, gamma2, w2 (cm^4).
WJ_PANAMA_CITY = {"rho": 1.97, "nu": 1.126, "delta": 0.0166, "gamma3": 4.0, "w3_cm3": 0.0000161, "mu": -2.44,
                  "gamma2": 3.12, "w2_cm4": 0.00849}


def row_text(name):
    """The row's lines in configs/sediments.yaml, comments included (the alternatives the paper prints are there)."""
    from gpusonar import config
    lines = (config.config_dir() / "sediments.yaml").read_text().splitlines()
    start = lines.index(f"{name}:")
    end = next((i for i in range(start + 1, len(lines)) if lines[i] and not lines[i].startswith((" ", "#"))), len(lines))
    return "\n".join(lines[start:end])


@pytest.mark.parametrize("name", JB_ROWS)
def test_the_jackson_briggs_rows_are_table_v_as_printed(name):
    """[JB] 5.1: each row equals Table V as printed (the chosen column), Mz from Table II, status confirmed, one
    site, no warning; w2 is the printed cm^4 through bottom.w2_from_cm4; the row's comment keeps the other values the
    paper prints for the site and its note names the choice."""
    key, column, site = JB_CHOICE[name]
    nu, rho, delta, sigma2, gammas, w2s = JB_TABLE_V[key]
    entry = bottom.sediment_table()[name]
    assert (entry["rho"], entry["nu"], entry["delta"]) == (rho, nu, delta)
    assert entry["scattering"] == {"w2_cm4": w2s[column], "gamma2": gammas[column], "sigma2": sigma2}
    assert entry["mean_grain_size_phi"] == JB_MZ[site][0]
    assert entry["status"] == "confirmed" and "unconfirmed_fields" not in entry
    assert "Jackson and Briggs" in entry["source"] and "Table V" in entry["source"]
    note = " ".join(entry["note"].split())
    assert note.startswith("ONE SITE, not a class") and "one site" in " ".join(entry["provenance"]["sigma2"].split())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        par = bottom.sediment(name)
    assert par == {"rho": rho, "nu": nu, "delta": delta, "w2": bottom.w2_from_cm4(w2s[column], gammas[column]),
                   "gamma2": gammas[column], "sigma2": sigma2}
    assert not bottom.is_reflection_only(name)
    # The other printed values for the site stay in the row's comment.
    text = row_text(name)
    others = {"quinault": ["0.00598", "3.92", "2.94"], "arafura": ["0.004-0.006", "5.24"],
              "san_francisco": ["0.000912", "3.38", "0.000826", "3.73", "0.001296", "3.56", "6.35", "0.000885"]}[site]
    assert all(v in text for v in others), (name, [v for v in others if v not in text])


def test_the_jackson_briggs_rows_name_their_choices_and_caveats():
    """The notes the rows carry ([JB] 5.1 and Sec. 6): Quinault's across-strike choice, the insensitive sigma2 and the
    whole-core delta; Arafura's 0.004-0.006 fit, bimodal sediment, border of single scattering; San Francisco's
    pre-storm along-strike choice and the 4.6 % question on its w2."""
    table = bottom.sediment_table()
    q, a, s = (" ".join(table[n]["note"].split()) for n in JB_ROWS)
    assert "across strike" in q and "unreliable" in q and "insensitive" in q and "whole-core" in q
    assert "0.004-0.006" in a and "bimodal" in a and "single scattering" in a
    assert "pre-storm" in s and "along strike" in s and "0.000885" in s and "4.6 %" in s
    # Eckernfoerde ([WJ] Table I) has no row: no printed sigma2, gassy mud, w3 adjusted ([WJ] 5.5).
    assert not any("eckern" in n for n in table)


def test_the_williams_jackson_panama_city_row_is_table_i():
    """[WJ] 5.5: rho, nu, delta, gamma2, w2 as printed in Table I; the printed spectral volume parameters in
    `volume_spectral:` (carried, read by no model yet: [WJ] 5.3); `sigma2` is OUR equivalent (the paper prints none),
    so it is `unconfirmed_fields: [sigma2]` and warns, while the reflection numbers do not."""
    entry = bottom.sediment_table()["williams_jackson_1998_panama_city"]
    p = WJ_PANAMA_CITY
    assert (entry["rho"], entry["nu"], entry["delta"]) == (p["rho"], p["nu"], p["delta"])
    assert entry["scattering"] == {"w2_cm4": p["w2_cm4"], "gamma2": p["gamma2"], "sigma2": 0.00007}
    assert entry["volume_spectral"] == {"w3_cm3": p["w3_cm3"], "gamma3": p["gamma3"], "mu": p["mu"]}
    assert entry["status"] == "confirmed" and entry["unconfirmed_fields"] == ["sigma2"]
    assert "Williams and Jackson" in entry["source"] and "Table I" in entry["source"]
    assert "ours" in entry["provenance"]["sigma2"] and "mean_grain_size_phi" not in entry   # Table I prints no Mz
    note = " ".join(entry["note"].split())
    assert note.startswith("ONE SITE, not a class") and "Table 13.1" in note and "0.0001475" in note
    with pytest.warns(UserWarning, match=r"\['sigma2'\] could not be confirmed"):
        par = bottom.sediment("williams_jackson_1998_panama_city")
    assert par["w2"] == bottom.w2_from_cm4(0.00849, 3.12) and par["w2"] == pytest.approx(1.4754e-4, rel=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert bottom.sediment_reflection("williams_jackson_1998_panama_city") == {"rho": 1.97, "nu": 1.126, "delta": 0.0166}
    # The equivalent: [WJ] (21)-(22) in backscatter at 20 degrees, 40 kHz, c 1500 m/s (not printed: ours), divided
    # by the amplitude attenuation in dB/m, (20 / ln 10) k delta / nu ([JB] (3), (A11)). Written out here:
    # backscatter makes the bracket of (21) kappa^2 (mu - 1) and Delta k = 2 k (cos^2 + (Re P)^2)^(1/2).
    k = 2 * math.pi * 40e3 / 1500.0
    kappa = (1 + 1j * p["delta"]) / p["nu"]
    g = math.radians(20.0)
    pz = cmath.sqrt(kappa ** 2 - math.cos(g) ** 2)
    pz = -pz if pz.imag < 0 else pz
    dk = 2 * k * math.sqrt(math.cos(g) ** 2 + pz.real ** 2)
    w3 = p["w3_cm3"] * 10.0 ** (2 * p["gamma3"] - 6)                    # cm^3 at 1 rad/cm -> m^(3 - gamma3)
    sigma_v = math.pi / 2 * k ** 4 * abs(kappa ** 2 * (p["mu"] - 1)) ** 2 * w3 / dk ** p["gamma3"]
    assert 10 * math.log10(sigma_v) == pytest.approx(-28.3, abs=0.05)   # [WJ] Sec. 4 item 6: -28.3 dB/m
    equivalent = sigma_v / (20 / math.log(10) * k * p["delta"] / p["nu"])
    assert round(equivalent, 5) == 0.00007                               # 6.93e-5, ours


# Readings of the papers' printed model curves (figure: [JB] about +-1 dB, [WJ] about +-0.5 dB; never a bound tighter
# than +-1 dB) and our default model (small-slope plus volume) with the row, c 1500 m/s (ours: the notes' values to
# their printed 0.1 dB, so +-0.051 dB). Grazing angles in degrees.
FIGURES = {
    # name: [(label, frequency, angles, readings, ours)]
    "jackson_briggs_1992_quinault": [
        ("[JB] Fig. 10a, across strike, 25 kHz", 25e3, (20, 30, 40, 50, 60), (-29.8, -27.8, -26.5, -24.7, -21.5),
         (-29.0, -27.1, -26.3, -24.6, -21.1)),
        ("[JB] Fig. 10b, across strike, 35 kHz", 35e3, (20, 30, 40, 50, 60), (-29.2, -27.3, -26.3, -24.8, -21.3),
         (-28.5, -26.8, -26.0, -24.2, -20.5))],
    "jackson_briggs_1992_arafura": [
        ("[JB] Fig. 12, 20 kHz", 20e3, (40, 50, 60, 70), (-22.5, -21.5, -21.0, -20.5), (-21.9, -21.1, -20.5, -19.8)),
        ("[JB] Fig. 14, 20 degrees, 15 kHz", 15e3, (20,), (-25.1,), (-24.8,)),
        ("[JB] Fig. 14, 20 degrees, 25 kHz", 25e3, (20,), (-25.1,), (-24.8,)),
        ("[JB] Fig. 14, 20 degrees, 35 kHz", 35e3, (20,), (-25.1,), (-24.8,)),
        ("[JB] Fig. 14, 20 degrees, 45 kHz", 45e3, (20,), (-25.1,), (-24.8,))],
    "jackson_briggs_1992_san_francisco": [
        ("[JB] Fig. 15, pre-storm along strike, 40 kHz", 40e3, (10, 20, 30, 40, 50, 60, 70),
         (-31.5, -28.5, -27.0, -25.8, -25.0, -24.3, -23.5), (-31.5, -28.5, -26.8, -25.7, -24.9, -24.3, -23.4))],
    "williams_jackson_1998_panama_city": [
        ("[WJ] Fig. 7a, backscatter (phi 180), 5 deg, 40 kHz", 40e3, (5,), (-40.0,), (-40.1,))],
}
# Where our default model is known NOT to reproduce the figure, the difference is recorded, not hidden: ours minus
# the reading must have the sign and at most the size (+1 dB of reading error) the notes give. Each is a DIFFERS
# of the notes, kept visible here.
RECORDED_DIFFERENCES = {
    # [JB] Sec. 3: at 10 deg ours (small-slope) is 1.4-2 dB lower; their model is composite there.
    "jackson_briggs_1992_quinault": [("[JB] Fig. 10a, 10 deg", 25e3, 10, -34.5, (-3.0, 0.0)),
                                     ("[JB] Fig. 10b, 10 deg", 35e3, 10, -33.5, (-3.0, 0.0))],
    # [JB] Sec. 3: ours 1.2-1.9 dB higher at 10-20 deg and up to 1.0 at 30 (their volume term is slope-averaged
    # and shadowed, ours not: their reading, not checked).
    "jackson_briggs_1992_arafura": [("[JB] Fig. 12, 10 deg", 20e3, 10, -30.5, (0.0, 2.9)),
                                    ("[JB] Fig. 12, 20 deg", 20e3, 20, -26.0, (0.0, 2.9)),
                                    ("[JB] Fig. 12, 30 deg", 20e3, 30, -24.0, (0.0, 2.9))],
    # [WJ] Sec. 0 item 1 and Sec. 4 item 1: the paper's model (Kirchhoff and perturbation joined by (10)) lies below
    # our small-slope by up to 6.1 dB in roughness (20 deg); the total here, with our equivalent sigma2, is
    # +1.2 to +5.8 dB above the reading (ours). The figure test of the paper's own model is [WJ] 5.2.
    "williams_jackson_1998_panama_city": [("[WJ] Fig. 7a, 10 deg", 40e3, 10, -31.6, (0.0, 7.1)),
                                          ("[WJ] Fig. 7b, 20 deg", 40e3, 20, -29.5, (4.0, 7.1)),
                                          ("[WJ] Fig. 7c, 35 deg", 40e3, 35, -27.6, (2.0, 7.1)),
                                          ("[WJ] Fig. 7d, 60 deg", 40e3, 60, -21.9, (0.0, 7.1))],
}


@pytest.mark.parametrize("name", JB_ROWS + WJ_ROWS)
def test_each_site_row_through_the_default_model_against_the_papers_figure(name):
    """Each site row through `bottom.named_backscattering_strength_db` (the default roughness model, small-slope, plus
    the volume term) at c 1500 m/s: equal to the notes' computed values (ours, 0.051 dB) and within +-1 dB of the
    paper's model curve where the notes say it agrees (figure); where the notes record a difference, its sign and
    size."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                                 # Panama City's sigma2 is unconfirmed
        for label, f, angles, readings, ours in FIGURES[name]:
            got = bottom.named_backscattering_strength_db(name, np.radians(np.array(angles, dtype=float)), f, 1500.0)
            assert got == pytest.approx(np.array(ours), abs=0.051), (label, got)
            assert np.all(np.abs(got - np.array(readings)) <= 1.0), (label, got - np.array(readings))
        for label, f, angle, reading, (low, high) in RECORDED_DIFFERENCES.get(name, []):
            got = float(bottom.named_backscattering_strength_db(name, np.radians([float(angle)]), f, 1500.0)[0])
            assert low < got - reading <= high, (label, got - reading)
