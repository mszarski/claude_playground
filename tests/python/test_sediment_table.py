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


def quiet(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kw)


def test_every_row_loads_and_the_table_is_complete():
    table = bottom.sediment_table()
    assert set(table) == set(JULIA_TABLE) | {"placeholder_sand"}
    for name, entry in table.items():
        par = quiet(bottom.sediment_reflection, name)
        assert set(par) == {"rho", "nu", "delta"} and all(math.isfinite(v) and v > 0 for v in par.values())
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
            bottom.sediment_reflection(name)
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
    par = bottom.sediment_reflection(name)
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
    assert OceanEnvironment(depth=80.0).bottom_speed == 1600.0      # the default is untouched


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
