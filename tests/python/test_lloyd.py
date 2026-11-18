"""The surface-image (Lloyd-mirror) factor, the depth plan built on it, target classes and environments.

Each physics piece is held to a closed form, a limit, reciprocity or an independent computation:
the textbook single-frequency pattern, the positions of its nulls and peaks, the resolved-path limits of the band
(+6 dB peak, 10 log10(6) energy), the rough-surface limits (coherent share exp(-Gamma^2), energy 1 + loss), the
simulator's image-source paths, the rigid cylinder's textbook target strengths."""

import copy
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

from gpusonar import acoustics, depthplan, lloyd
from gpusonar import buildspec as bs
from gpusonar import target_classes as tc

ROOT = pathlib.Path(__file__).resolve().parents[2]
F_LO, F_HI = 4000.0, 6000.0


def tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "tools"))
    spec.loader.exec_module(mod)
    return mod


# --- the factor ---------------------------------------------------------------------------------------

def test_single_frequency_factor_is_the_textbook_lloyd_mirror():
    own, zt, r = np.array([5.0, 10.0, 40.0, 110.0, 225.0, 300.0]), 4.0, 12000.0
    h, s = lloyd.one_way(np.array([5000.0]), own, zt, r)
    assert np.all(s == 0.0)
    got = 40.0 * np.log10(np.abs(h[:, 0]))
    assert got == pytest.approx(lloyd.lloyd_two_way_db(5000.0, own, zt, r), abs=0.02)
    lobe = lloyd.first_lobe_own_depth(5000.0, zt, r)
    assert lobe["peak_m"] == pytest.approx(225.0) and lobe["depth_product_m2"] == pytest.approx(900.0)       # lambda R / 4 at 5 kHz and 12 km
    peak = lloyd.two_way(5000.0, 5000.0, lobe["peak_m"], zt, r, n_f=1)["narrowband_db"]
    assert float(peak) == pytest.approx(40.0 * np.log10(2.0), abs=0.02)                                       # +6 dB a leg
    assert float(lloyd.two_way(5000.0, 5000.0, lobe["zero_db_m"], zt, r, n_f=1)["narrowband_db"]) == pytest.approx(0.0, abs=0.05)
    assert float(lloyd.two_way(5000.0, 5000.0, lobe["first_null_m"], zt, r, n_f=1)["narrowband_db"]) < -60.0
    # the 12 km scenario's geometry (array 10 m, ship 4 m): about -17 dB a leg
    assert float(lloyd.lloyd_two_way_db(5000.0, 10.0, 4.0, r)) == pytest.approx(-34.2, abs=0.1)


def test_reciprocity_and_the_no_surface_limit():
    f = lloyd.band(F_LO, F_HI, 65)
    a, _ = lloyd.one_way(f, 37.0, 180.0, 5000.0, lloyd.Surface(wind_speed=8.0))
    b, _ = lloyd.one_way(f, 180.0, 37.0, 5000.0, lloyd.Surface(wind_speed=8.0))
    assert np.allclose(a, b)
    layer = lloyd.Layer(depth=120.0, bottom={"speed": 1650.0, "density": 1.9, "attenuation": 0.8})
    a, _ = lloyd.one_way(f, 20.0, 90.0, 6000.0, lloyd.FLAT, layer)
    b, _ = lloyd.one_way(f, 90.0, 20.0, 6000.0, lloyd.FLAT, layer)
    assert np.allclose(a, b)
    none = lloyd.summarise(*lloyd.one_way(f, 50.0, 20.0, 3000.0, lloyd.Surface(coefficient=0.0)))
    assert all(abs(float(none[k])) < 1e-9 for k in ("narrowband_db", "energy_db", "peak_db"))                 # one path: the free-field budget


def test_the_band_fills_the_nulls_between_lobes_but_not_the_one_at_the_surface():
    r, zt = 12000.0, 4.0
    null = lloyd.first_lobe_own_depth(5000.0, zt, r)["first_null_m"]
    t = lloyd.two_way(F_LO, F_HI, null, zt, r, n_f=257)
    assert float(t["narrowband_db"]) < -60.0 and -8.0 < float(t["peak_db"]) < -5.0 and -5.0 < float(t["energy_db"]) < -3.0
    # independent number for the energy: mean over the band of (2 - 2 cos(phi))^2 with phi = 2 pi f / 5 kHz
    phi = 2.0 * np.pi * np.linspace(F_LO, F_HI, 20001) / 5000.0
    assert float(t["energy_db"]) == pytest.approx(10.0 * np.log10(np.mean((2.0 - 2.0 * np.cos(phi)) ** 2)), abs=0.05)
    shallow = lloyd.two_way(F_LO, F_HI, 10.0, zt, r, n_f=257)                                                  # close to the surface the band changes nothing
    assert abs(float(shallow["peak_db"]) - float(shallow["narrowband_db"])) < 0.5 and float(shallow["peak_db"]) < -30.0
    # paths resolved by the pulse (path difference >> c / B): direct-surface and surface-direct arrive together, +6 dB; energy 1 + 4 + 1
    deep = lloyd.two_way(F_LO, F_HI, 250.0, 250.0, 3000.0, n_f=1025)
    assert float(deep["peak_db"]) == pytest.approx(20.0 * np.log10(2.0), abs=0.2) and float(deep["energy_db"]) == pytest.approx(10.0 * np.log10(6.0), abs=0.2)


def test_rough_surface_coherent_share_and_incoherent_limit():
    r, z1, z2 = 1000.0, 60.0, 60.0
    grazing = np.arcsin((z1 + z2) / np.hypot(r, z1 + z2))
    surf = lloyd.Surface(rms_height=0.2)
    assert float(surf.coherent(grazing, 5000.0)) == pytest.approx(float(acoustics.rough_coherent_reflection(grazing, 0.2, 5000.0)), rel=1e-12)
    h, s = lloyd.one_way(np.array([5000.0]), z1, z2, r, surf)
    ratio2 = (np.hypot(r, z1 - z2) / np.hypot(r, z1 + z2)) ** 2
    rho2 = float(acoustics.rough_coherent_reflection(grazing, 0.2, 5000.0)) ** 2
    assert float(s[0]) == pytest.approx((1.0 - rho2) * ratio2, rel=1e-9)                                       # what is not coherent is kept as power
    very = lloyd.two_way(F_LO, F_HI, z1, z2, r, lloyd.Surface(rms_height=3.0), n_f=257)
    assert float(very["peak_db"]) == pytest.approx(0.0, abs=0.05)                                              # the coherent echo is the direct path alone
    assert float(very["energy_db"]) == pytest.approx(20.0 * np.log10(1.0 + ratio2), abs=0.05)                  # mean intensity (1 + |R|^2) a leg
    assert float(very["coherent_share"]) == pytest.approx(1.0 / (1.0 + ratio2), abs=0.01)
    # the wind law is only the default for the height: an explicit height wins
    assert lloyd.Surface(wind_speed=10.0).height() == pytest.approx(acoustics.rms_wave_height(10.0)) and lloyd.Surface(wind_speed=10.0, rms_height=0.1).height() == 0.1
    # at 12 km the grazing angle is so small that a 5 m/s sea is a mirror
    assert float(lloyd.two_way(F_LO, F_HI, 225.0, 4.0, 12000.0, lloyd.Surface(wind_speed=5.0))["coherent_share"]) > 0.99


def test_bubble_loss_weakens_the_image_and_fills_the_null_by_the_named_law():
    from gpusonar.sim import surface as surface_model
    r, z1, z2 = 12000.0, 10.0, 4.0
    surf = lloyd.Surface(wind_speed=9.8, bubble_model="apl_1994")
    grazing = np.arcsin((z1 + z2) / np.hypot(r, z1 + z2))
    loss = float(surface_model.apl_1994_bubble_loss_db(grazing, 9.8, 5000.0))
    assert float(surf.bubble_loss_db(grazing, 5000.0)) == pytest.approx(loss)
    h, _ = lloyd.one_way(np.array([5000.0]), z1, z2, r, lloyd.Surface(wind_speed=9.8, rms_height=0.0, bubble_model="apl_1994"))
    m = 10.0 ** (-loss / 20.0) * np.hypot(r, z1 - z2) / np.hypot(r, z1 + z2)
    k_delta = 2.0 * np.pi * 5000.0 / 1500.0 * (np.hypot(r, z1 + z2) - np.hypot(r, z1 - z2))
    assert abs(h[0]) ** 2 == pytest.approx(1.0 + m * m - 2.0 * m * np.cos(k_delta), rel=1e-9)                 # the docstring's closed form
    assert float(lloyd.two_way(F_LO, F_HI, z1, z2, r, surf)["peak_db"]) > float(lloyd.two_way(F_LO, F_HI, z1, z2, r)["peak_db"]) + 10.0
    with pytest.raises(ValueError):
        lloyd.Surface(wind_speed=5.0, bubble_model="nonsense").bubble_loss_db(0.1, 5000.0)
    with pytest.raises(ValueError):
        lloyd.Surface.from_config({"sea_state": 3, "wind_speed": 2.0})
    assert lloyd.Surface.from_config({"sea_state": 5}).wind_speed == acoustics.sea_state_wind_speed(5)


def test_layer_image_series_matches_the_simulators_images_and_carries_more_than_spherical_spreading():
    from gpusonar.sim.propagation import ImageSourceProvider
    bottom = {"speed": 1650.0, "density": 1.9, "attenuation": 0.8}
    prov = ImageSourceProvider(1500.0, surface_z=0.0, bottom_z=100.0, max_order=6, bottom=bottom, engine="numpy")
    src, pt = np.array([0.0, 0.0, 30.0]), np.array([[0.0, 3000.0, 60.0]])
    rays = prov.eigenrays(src, pt)
    f = lloyd.band(F_LO, F_HI, 33)
    direct = np.linalg.norm(pt[0] - src)
    want = np.array([np.sum(rays.amplitude[0] * direct * np.exp(-2j * np.pi * fi * (rays.delay[0] - direct / 1500.0))) for fi in f])
    got, _ = lloyd.one_way(f, 30.0, 60.0, 3000.0, lloyd.FLAT, lloyd.Layer(depth=100.0, bottom=bottom, max_order=6))
    assert np.allclose(got, want, rtol=1e-6, atol=1e-9)
    duct = lloyd.two_way(F_LO, F_HI, 30.0, 4.0, 12000.0, lloyd.FLAT, lloyd.Layer(depth=100.0, bottom=bottom, max_grazing_deg=30.0), n_f=513)
    assert 12.0 < float(duct["one_way_energy_db"]) < 17.0                  # Bellhop's figure for this water is 14 dB less loss each way (DESIGN Sec. 5)
    with pytest.raises(ValueError):
        lloyd.one_way(f, 130.0, 60.0, 3000.0, lloyd.FLAT, lloyd.Layer(depth=100.0, bottom=bottom))


def test_projector_below_the_array_matters_in_the_null():
    r, zt = 12000.0, 4.0
    same = float(lloyd.two_way(F_LO, F_HI, 10.0, zt, r)["peak_db"])
    lower = float(lloyd.two_way(F_LO, F_HI, 10.0, zt, r, projector_below_m=1.5)["peak_db"])
    assert lower - same == pytest.approx(20.0 * np.log10(11.5 / 10.0), abs=0.1)                               # the outbound leg alone gains (z_tx / z_rx)^2 in power


def test_factor_matches_the_simulator_through_run_scenario_on_both_engines():
    v = tool("surface_image_validation")
    for own, zt in ((30.0, 4.0), (225.0, 4.0)):
        model = lloyd.two_way(F_LO, F_HI, own, zt, 12000.0, n_f=257, projector_below_m=v.PROJECTOR_BELOW)
        for engine in ("cpp", "numpy"):
            p1, e1 = v.echo(v.scenario(own, zt, 12000.0, True), engine)
            p0, e0 = v.echo(v.scenario(own, zt, 12000.0, False), engine)
            assert 10.0 * np.log10(p1 / p0) == pytest.approx(float(model["peak_db"]), abs=0.5), (own, engine)
            assert 10.0 * np.log10(e1 / e0) == pytest.approx(float(model["energy_db"]), abs=0.7), (own, engine)


def test_validation_file_holds_the_model_to_the_simulator_the_chain_and_bellhop():
    V = json.loads((ROOT / "docs" / "design" / "surface_image_validation.json").read_text())
    assert max(V["largest_difference_db"].values()) < 1.3
    chain = V["through_chain"]
    rows = [r for r in chain["rows"] if r["chain_factor_db"] is not None]
    assert len(rows) >= 5 and max(abs(r["chain_factor_db"] - r["model_peak_db"]) for r in rows) < 1.5        # 1.33 dB with the multiscale detector (D-93) and the coloured noise (D-103); under 1.0 before
    if V.get("bellhop"):
        for case in V["bellhop"].values():
            assert case["one_way_energy_difference_db"]["largest"] < 0.7


# --- target classes -----------------------------------------------------------------------------------

def test_cylinder_target_strength_is_the_textbook_figure_at_broadside_near_and_far_and_at_the_end():
    cf = tc.cylinder_closed_forms(10.0, 2.0, 5000.0, 12000.0)
    assert cf["broadside_far_field_db"] == pytest.approx(10.0 * np.log10(1.0 * 100.0 / (2.0 * 0.3)))
    one = tc.cylinder_ts_db(10.0, 2.0, [0.0, 90.0], 4990.0, 5010.0, None)
    assert one[0] == pytest.approx(cf["broadside_far_field_db"], abs=0.2) and one[1] == pytest.approx(cf["end_on_disc_db"], abs=0.3)
    long_ = tc.cylinder_closed_forms(100.0, 10.0, 5000.0, 12000.0)
    assert long_["far_field_beyond_m"] > 12000.0
    near = tc.cylinder_ts_db(100.0, 10.0, [0.0], F_LO, F_HI, 12000.0)[0]
    assert near == pytest.approx(long_["broadside_near_field_db"], abs=1.0) and near < long_["broadside_far_field_db"] - 5.0   # a R / 4 inside L^2 / lambda
    off = tc.cylinder_ts_db(10.0, 2.0, [30.0, 45.0], F_LO, F_HI, 12000.0)
    assert np.all(off < cf["broadside_far_field_db"] - 30.0)               # a bare smooth cylinder returns almost nothing off broadside


def test_target_classes_are_checked_resolved_and_the_shipped_ones_are_marked_assumed():
    classes = tc.load("reference_classes")
    assert [c["name"] for c in classes] == ["ship", "xl_auv", "large_submerged"] and tc.problems(classes) == []
    assert all(c["status"] == "assumed" for c in classes)                   # nobody vouches for any target strength shipped here
    res = tc.resolve(classes, F_LO, F_HI, 12000.0)
    by = {c["name"]: c for c in res}
    assert by["ship"]["ts_db"] == [25.0, 25.0] and by["xl_auv"]["ts_db"][0] == 5.0
    for name in ("xl_auv", "large_submerged"):                              # the typed high end is the cylinder's broadside figure
        assert by[name]["ts_db"][1] == pytest.approx(by[name]["by_aspect"][0]["ts_db"], abs=0.5), name
    model = tc.resolve([{"name": "hull", "cylinder": {"length_m": 30.0, "diameter_m": 4.0}, "depth_m": [10.0, 50.0]}], F_LO, F_HI)[0]
    assert model["status"] == "model" and model["ts_db"][0] < model["ts_db"][1]
    bad = [{"name": "a b", "depth_m": [5.0, 1.0]}, {"name": "x", "target_strength_db": [10.0, 5.0], "depth_m": [1.0, 2.0], "status": "sure", "colour": 1},
           {"name": "x", "cylinder": {"length_m": 2.0, "diameter_m": 5.0}, "depth_m": [1.0, 2.0], "status": "confirmed"}]
    where = [w for w, _ in tc.problems(bad)]
    for want in ("targets.a b.name", "targets.a b.target_strength_db", "targets.a b.depth_m", "targets.x.target_strength_db", "targets.x.status", "targets.x.colour", "targets.x.name", "targets.x.cylinder"):
        assert want in where, want


# --- environments and the plan ------------------------------------------------------------------------

def test_environments_are_checked_in_the_spec_sheets_words():
    envs = depthplan.load_environments("illustrative")
    assert depthplan.environment_problems(envs) == [] and envs[0].get("water_depth_m") is None
    bad = [{"name": "a", "bottom": "medium_sand"}, {"name": "b", "water_depth_m": 50.0, "bottom": "porridge", "sea_state": 3, "wind_speed_m_s": 4.0, "rms_wave_height_m": 8.0,
                                                    "own_depth_limits_m": [60.0, 80.0], "bubble_model": "foam"},
           {"name": "c", "profile": {"depth_m": [0.0, 10.0], "speed_m_s": [1500.0]}}]
    where = [w for w, _ in depthplan.environment_problems(bad)]
    for want in ("environments.a.bottom", "environments.b.bottom", "environments.b.sea_state", "environments.b.rms_wave_height_m", "environments.b.own_depth_limits_m",
                 "environments.b.bubble_model", "environments.c.profile"):
        assert want in where, want
    assert depthplan.environment_problems([{"name": "ok", "water_depth_m": 80.0, "bottom": "medium_sand", "sea_state": 3}]) == []


def test_build_spec_validates_targets_and_environments_and_they_change_no_generated_file():
    spec = bs.load("popoto_face_3x32")
    assert spec["targets"] == "reference_classes" and spec["environments"] == "illustrative"
    issues = bs.validate(spec)
    assert not [i for i in issues if i.level == "error"]
    warned = [i.where for i in issues if i.level == "warning"]
    assert "targets.ship" in warned and "environments.deep_no_bottom_return" in warned      # placeholders are said to be placeholders
    bare = copy.deepcopy(spec)
    bare.pop("targets"), bare.pop("environments")
    # Changed with the owner's decision to switch the coverage picture on for the face builds: the spec now names one of its
    # environments as the SITE of `chain.coverage` (operation.coverage), which does go into the deployment file, and a spec
    # without that environment is refused. The statement below is about the planning inputs, so it is made without the site.
    with pytest.raises(ValueError, match="not one of the build's environments"):
        bs.files(bare)
    spec["operation"].pop("coverage"), bare["operation"].pop("coverage")
    with_, without = bs.files(spec), bs.files(bare)               # only the submerged-mover trial is added; no sonar or deployment file changes
    assert {k: v for k, v in with_.items() if "submerged" not in k} == without and [k for k in with_ if k not in without] == ["scenarios/popoto_face_3x32_submerged.yaml"]
    broken = copy.deepcopy(spec)
    broken["targets"] = [{"name": "t", "depth_m": [1.0, 2.0]}]
    broken["environments"] = "no_such_file"
    errors = [i.where for i in bs.validate(broken) if i.level == "error"]
    assert "targets.t.target_strength_db" in errors and "environments" in errors
    inline = copy.deepcopy(spec)
    inline["targets"] = [{"name": "t", "target_strength_db": 12.0, "depth_m": [10.0, 40.0], "status": "confirmed"}]
    inline["environments"] = [{"name": "e", "water_depth_m": 80.0, "bottom": "medium_sand", "wind_speed_m_s": 3.0, "own_depth_limits_m": [5.0, 60.0], "illustrative": False}]
    issues = bs.validate(inline)
    assert not [i for i in issues if i.level == "error"] and not [i for i in issues if i.where.startswith(("targets", "environments"))]


def test_performance_is_unchanged_without_an_environment_and_gains_the_surface_term_with_one():
    spec = bs.load("popoto_face_1x32")
    d = bs.derive(spec)
    plain = bs.performance(d)
    assert "surface" not in plain
    env = depthplan.load_environments("illustrative")[0]
    with_env = bs.performance(d, environment=env, own_depth_m=10.0)
    assert {k: v for k, v in with_env.items() if k != "surface"} == plain
    ship = with_env["surface"]["classes"]["ship"]
    r = "12000.0"
    assert ship["snr_free_field_db"][r] == pytest.approx(plain["row_snr_db"]["ship"][r])              # one row's SNR: the rows are in the threshold, not in the SNR
    want = float(lloyd.two_way(F_LO, F_HI, 10.0, 4.0, 12000.0, depthplan.surface_of(env), projector_below_m=d["projector_m"][2])["peak_db"])
    assert ship["factor_db"][r] == pytest.approx(want, abs=1e-3) and ship["factor_db"][r] < -30.0
    need = plain["detection"]["threshold_db"]                                                            # the build's own detector's, not a typed 13 dB
    assert with_env["surface"]["threshold_db"] == pytest.approx(need) and need == pytest.approx(15.6, abs=0.3)
    assert ship["margin_db"][r] == pytest.approx(ship["snr_free_field_db"][r] + want - need, abs=1e-3)
    eq, gains = bs.budget(d)
    assert float(eq.snr(12000.0)) + 25.0 == pytest.approx(ship["snr_free_field_db"][r]) and gains["wind_speed_m_s"] == 5.0


@pytest.fixture(scope="module")
def deep_plan():
    spec = bs.load("popoto_face_3x32")
    d = bs.derive(spec)
    env = depthplan.load_environments("illustrative")[0]
    return d, env, depthplan.plan(d, env, bs.target_classes_of(spec))


def test_plan_recommends_the_first_lobe_for_the_ship_and_shows_the_loss_at_ten_metres(deep_plan):
    d, env, P = deep_plan
    ship = P["classes"]["ship"]
    rec = ship["recommended"]
    # worst over hull depths 2 to 6 m: the best own depth lies between the first-lobe peaks for 6 m and for 2 m, and the factor is positive
    lo, hi = (lloyd.first_lobe_own_depth(5000.0, z, 12000.0)["peak_m"] for z in (6.0, 2.0))
    assert lo <= rec["own_depth_m"] <= hi and rec["factor_db"] > 3.0 and rec["margin_db"] == pytest.approx(ship["free_field_margin_db"] + rec["factor_db"])
    assert ship["at_own_depth"]["10"]["margin_db"] < -30.0 and rec["margin_db"] > 0.0          # lost at 10 m, held at depth
    band_lo, band_hi = rec["held_band_m"][0]                       # the first lobe; deeper lobes hold it again
    assert 80.0 < band_lo < rec["own_depth_m"] < band_hi and ship["shallowest_own_depth_held_m"] == band_lo
    assert P["platform_max_depth_m"] == 500.0 and P["own_depths_m"][-1] == 500.0 and rec["inside_platform_limit"] is True      # the owner: "Depth no further than 500m"
    assert depthplan.own_depth_grid({"name": "e", "own_depth_limits_m": [5.0, 800.0]}, max_depth=500.0)[-1] == 500.0
    assert depthplan.own_depth_grid({"name": "e", "water_depth_m": 100.0}, max_depth=500.0)[-1] == 90.0 and depthplan.own_depth_grid({"name": "e"})[-1] == 300.0
    own, at = np.asarray(P["own_depths_m"]), np.asarray(ship["margin_at_range_worst_db"])
    i = int(np.argmin(np.abs(own - rec["own_depth_m"])))
    zt = np.array(ship["target_depths_used_m"])
    eq, _ = bs.budget(d)
    want = float(lloyd.two_way(F_LO, F_HI, own[i], zt, 12000.0, depthplan.surface_of(env), n_f=129)["peak_db"].min()) + float(eq.snr(12000.0)) + 25.0 - P["threshold_db"]
    assert at[i] == pytest.approx(want, abs=1e-3)
    # the threshold is the three-row face's own detector's (11.6 dB of one row's SNR for a detection probability of 0.5), not a typed 13 dB at a summed map
    assert P["threshold_db"] == pytest.approx(bs.detection(d)["threshold_db"]) == pytest.approx(11.6, abs=0.3) and P["detection"]["rows"] == 3
    assert rec["detection_probability"] > 0.99 and ship["free_field_detection_probability"] == pytest.approx(0.99, abs=0.02)


def test_plan_deep_targets_are_spared_the_nulls_and_two_classes_share_a_band(deep_plan):
    _, _, P = deep_plan
    big = P["classes"]["large_submerged"]
    own = np.asarray(P["own_depths_m"])
    f = np.asarray(big["factor_at_range_worst_db"])
    assert np.all(f[own >= 100.0] > 3.0) and np.ptp(f[own >= 100.0]) < 4.0                     # resolved paths: about +6 dB whatever the depth
    both = P["together"]["ship + large_submerged"]
    assert both["held_band_m"] and both["margin_of_the_weaker_db"] > 0.0
    lo, hi = both["held_band_m"][0]
    ship_lo, ship_hi = P["classes"]["ship"]["recommended"]["held_band_m"][0]
    assert lo >= ship_lo and hi <= ship_hi
    assert P["classes"]["xl_auv"]["recommended"]["margin_db"] < 0.0 and P["classes"]["xl_auv"]["margin_high_ts_db"] == pytest.approx(17.0)
    steps = P["classes"]["ship"]["depth_diversity"]["steps"]
    assert steps and steps[-1]["covered_share"] == 1.0 and len(steps) <= 3
    assert depthplan._intervals(np.array([1.0, 2.0, 3.0, 4.0]), np.array([True, True, False, True])) == [[1.0, 2.0], [4.0, 4.0]]
    cover = depthplan._cover(np.array([[1, 1, 0, 0], [0, 0, 1, 0], [0, 1, 1, 1]], dtype=bool), np.array([10.0, 20.0, 30.0]))
    assert [s["own_depth_m"] for s in cover] == [30.0, 10.0] and cover[-1]["covered_share"] == 1.0


def test_plan_in_shallow_water_over_sand_holds_the_ship_at_any_depth():
    spec = bs.load("popoto_face_1x32")
    env = next(e for e in depthplan.load_environments("illustrative") if e["name"] == "shelf_100m_sand")
    classes = [c for c in bs.target_classes_of(spec) if c["name"] == "ship"]
    P = depthplan.plan(bs.derive(spec), env, classes, own_depths=[10.0, 50.0, 85.0], ranges=[6000.0, 12000.0])
    ship = P["classes"]["ship"]
    assert "image series" in P["method"] and min(ship["margin_at_range_worst_db"]) > 5.0      # ducted: tens of dB over the free-field budget
    with pytest.raises(ValueError):
        depthplan.own_depth_grid({"name": "puddle", "water_depth_m": 12.0})


def test_planner_command_prints_a_plan_and_refuses_a_bad_environment(tmp_path, capsys):
    planner = tool("popoto_depth_planner")
    out = tmp_path / "plan"
    assert planner.main(["--build", "popoto_static_3x32", "--water-depth", "80", "--bottom", "medium_sand", "--sea-state", "3", "--rms-height", "0.15", "--own-depth", "5", "60",
                         "--ts", "10", "--target-depth", "20", "60", "--range", "2500", "--out", str(out), "--no-figure"]) == 0
    text = capsys.readouterr().out
    assert "Best own depth" in text and "rms wave height 0.15 m" in text and "rms" in text.split("Source level:")[1]
    J = json.loads(out.with_suffix(".json").read_text())
    assert J["plans"]["given"]["classes"]["given"]["recommended"]["own_depth_m"] <= 60.0 and "rms" in J["source_level_convention"]
    assert planner.main(["--build", "popoto_static_3x32", "--water-depth", "80", "--bottom", "porridge", "--ts", "10", "--target-depth", "20", "60"]) == 2
    assert "porridge" in capsys.readouterr().out


# --- the owner's decisions carried by the spec ---------------------------------------------------------

def test_source_level_is_an_rms_level_by_name_and_the_old_name_is_refused():
    spec = bs.load("popoto_face_3x32")
    assert "source_level_db_rms" in spec["projector"] and "rms" in bs.SCHEMA["projector"]["source_level_db_rms"][1]
    old = copy.deepcopy(spec)
    old["projector"]["source_level_db"] = old["projector"].pop("source_level_db_rms")
    msgs = [i.message for i in bs.validate(old) if i.where == "projector.source_level_db"]
    assert msgs and "source_level_db_rms" in msgs[0]


def test_recording_and_compute_options_feed_the_deployment_file_and_warn_when_a_cpu_is_too_slow():
    one, three = bs.load("popoto_face_1x32"), bs.load("popoto_face_3x32")
    assert bs.recording_mode(three) == "raw" and bs.compute_option(three) == "gpu" and bs.compute_option(one) == "decimate_first_cpu"
    assert bs.deploy_config(one)["chain"]["decimate"] == {} and "decimate" not in bs.deploy_config(one)["source"]         # raw recording, the chain decimates
    assert "decimate" not in bs.deploy_config(three)["chain"]
    dec = copy.deepcopy(one)
    dec["storage"]["record"] = "decimated"
    cfg = bs.deploy_config(dec)
    assert cfg["source"]["decimate"] == {} and "decimate" not in cfg["chain"]
    assert "storage.record" in [i.where for i in bs.validate(dec) if i.level == "warning"]
    d_raw, d_dec = bs.derive(one), bs.derive(dec)
    assert d_raw["record_bytes_per_hour_chosen"] == d_raw["record_bytes_per_hour"] and d_dec["record_bytes_per_hour_chosen"] == d_dec["record_bytes_per_hour_decimated"]
    S = json.loads((ROOT / "docs" / "design" / "popoto_stage2.json").read_text())["decimation"]["plan"]["stages"]
    assert d_raw["decimation_factor"] == next(iter(S.values()))["default"]["factor"]
    assert d_raw["record_bytes_per_hour"] / d_raw["record_bytes_per_hour_decimated"] == pytest.approx(next(iter(S.values()))["default"]["storage_ratio"], rel=0.01)
    slow = copy.deepcopy(three)
    slow["compute"]["option"] = "decimate_first_cpu"
    msgs = [i.message for i in bs.validate(slow) if i.where == "compute.option"]
    assert msgs and "not real time" in msgs[0]
    assert not [i for i in bs.validate(three) if i.where == "compute.option"]
    missing = copy.deepcopy(three)
    missing["compute"].pop("option")
    assert bs.compute_option(missing) == "gpu"
