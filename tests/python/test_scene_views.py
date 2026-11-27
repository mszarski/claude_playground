"""The scene trace and its 2-D page for every scenario (python/gpusonar/viz, tools/scenario_view.py; packages V1 and
V2 of docs/tasks/VISUALIZATION.md).

What is held: every one of the scenario files has a family and yields a trace and a models list without a
simulation; the drawn paths of the new tiers are the provider's (general planes: specular points on their planes and
the path as long as the image's, both engines; the sediment tier's refracted ray: the provider's delay, the crossing
on the interface, both engines; Bellhop over a range-dependent bed and its 2-D projection of a 3-D bed: bounce points
on the bed); the trace's two-way delays are the simulator's pairs at each tier, both engines; one-way sources and the
projector's direct sound are the simulator's one-way paths; each boundary model a scenario uses is listed and drawn;
the page of each family builds. Tolerances: rounding (the GPU job's C++ and NumPy differ there).
"""

import os
import re
import warnings

import numpy as np
import pytest
from gpusonar import acoustics
from gpusonar import environment as site
from gpusonar.config import deep_merge, load_scenario_config
from gpusonar.sim import engine as sim_engine
from gpusonar.sim import propagation as prop
from gpusonar.sim.simulator import run_scenario
from gpusonar.sonar import SonarModel
from gpusonar.viz import catalogue, page2d
from gpusonar.viz import trace as tr

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SCENARIOS = sorted(f[:-5] for f in os.listdir(os.path.join(ROOT, "configs", "scenarios")) if f.endswith(".yaml"))


def _resolved(name, overrides=None, engine="numpy"):
    """(resolved config, sonar, provider) of a scenario as run_scenario builds them, without simulating."""
    cfg = acoustics.resolve_sea_state(load_scenario_config(name, overrides, expand=False))
    sonar = SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides"))
    env = site.resolve(cfg.get("environment"), strict=True)
    cfg = dict(site.apply_to_scenario(cfg, env, sonar), environment=env)
    return cfg, sonar, prop.make_provider(cfg.get("fidelity", "free_field"), env["sound_speed"], cfg.get("propagation"), engine=engine)


def _run(name, engine=None, **overrides):
    try:
        return run_scenario(name, overrides=overrides or None, engine=engine)
    except sim_engine.EngineUnavailable as exc:          # a module without the C++ simulator
        pytest.skip(str(exc))


def _need_bellhop():
    from gpusonar.sim import bellhop
    if bellhop.find_bellhop() is None:
        pytest.skip("Bellhop is not installed (GPUSONAR_BELLHOP)")


# --- the catalogue ---------------------------------------------------------------------------------------------------

def test_every_scenario_has_a_family_as_the_catalogue_lists_it():
    # docs/handoff/SIMULATION.md section 4 groups the scenarios under headings; catalogue.FAMILIES writes the same
    # groups out (the last heading split in two). Each heading's scenarios ("`a`, `_b`" means a and a's stem + _b)
    # must be one family's.
    assert len(SCENARIOS) == 98
    assert all(catalogue.scenario_family(n)[1] != "other" for n in SCENARIOS)
    assert sorted(n for _, _, names in catalogue.FAMILIES for n in names) == SCENARIOS
    text = open(os.path.join(ROOT, "docs", "handoff", "SIMULATION.md"), encoding="utf-8").read()
    text = text.split("## 4. The scenario catalogue")[1].split("### How a scenario")[0]
    groups, current = {}, None
    for line in text.splitlines():
        if line.startswith("### "):
            current = line[4:]
            groups[current] = set()
        elif line.startswith("| `") and current:
            names = re.findall(r"`([a-z0-9_]+)`", line.split("|")[1])
            first = names[0]
            for n in names:
                if n.startswith("_"):
                    stem = first
                    while stem and stem + n not in SCENARIOS and "_" in stem:
                        stem = stem.rsplit("_", 1)[0]
                    n = stem + n
                groups[current].add(n)
    for heading, names in groups.items():
        fams = {catalogue.scenario_family(n)[1] for n in names if n in SCENARIOS}
        if heading.startswith("Fjord"):
            assert fams == {"fjord", "ssp_range"}, heading
        else:
            assert len(fams) == 1, (heading, fams)
    assert set(catalogue.FAMILY_EXAMPLE) == {s for _, s, _ in catalogue.FAMILIES}


@pytest.mark.parametrize("name", SCENARIOS)
def test_every_scenario_yields_a_trace_and_a_models_list(name):
    # No simulation and no provider call: the geometry, environment, models and beams of the page's ping.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = tr.static_trace(name)
    assert t["kind"] == tr.TRACE_KIND and t["scenario"] == name
    assert t["models"][0]["model"] == t["tier"]
    for m in t["models"]:
        assert m["source"] and m["status"] and m["key"], m
    assert t["view"]["note"] and 0 <= t["view"]["ping_of_run"] < max(t["view"]["n_pings"], 1)
    assert t["beams"] is not None and t["beams"]["projector_text"].startswith("projector:")
    if t["sonar"]["mode"] in ("flank", "fls"):
        assert t["beams"]["chain"]["bearings_deg"] or t["sonar"]["mode"] == "fls"
    env = t["environment"]
    if t["tier"] in ("eigenray", "eigenray3d"):
        assert env["ssp"] is not None and env["bottom"]["speed"] > 0
    if t["tier"] == "sediment":
        assert env["interface"] is not None
    assert env["absorption_db_per_km"] is not None


def test_sas_pages_show_the_middle_ping_and_say_so():
    raw = load_scenario_config("stripmap_points", None, expand=False)
    plan = catalogue.view_plan(raw)
    assert plan["ping"] == 45 and len(plan["track"]) == 90
    np.testing.assert_allclose(plan["overrides"]["platform"]["position"], plan["track"][45])
    assert "middle" in plan["note"]
    legs = catalogue.view_plan(load_scenario_config("sas_minsas_survey", None, expand=False))
    assert legs["ping"] == 40 and len(legs["track"]) == 160 and legs["overrides"]["platform"]["legs"][0]["pings"] == 1
    assert catalogue.view_plan(load_scenario_config("flank_patrol", None, expand=False))["ping"] == 0


# --- the new tiers' paths ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["numpy", "cpp"])
def test_general_plane_paths_bounce_on_their_planes_and_are_as_long_as_the_image(engine):
    cfg, sonar, p = _resolved("fls_120k_quay_multipath", engine=engine)
    src = np.array([0.0, 0.0, 3.1])
    pts = np.array([[45.0, 8.0, 6.0], [60.0, -6.0, 11.0], [30.0, 15.0, 2.0]])
    try:
        eig = p.eigenrays(src, pts)
    except sim_engine.EngineUnavailable as exc:
        pytest.skip(str(exc))
    walls = 0
    for s in range(len(pts)):
        rays = tr._leg_rays(p, eig, s, src, pts[s], sonar.center_frequency, True, pts)
        assert len(rays) == int(eig.valid[s].sum()) > 3
        for r in rays:
            assert r["polyline"] is not None, r["path"]
            line = np.asarray(r["polyline"])
            np.testing.assert_allclose(line[0], src, atol=1e-9)
            np.testing.assert_allclose(line[-1], pts[s], atol=1e-9)
            assert np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1)) == pytest.approx(p.sound_speed * r["delay_s"], rel=1e-9)
            assert sum(b["boundary"] == "surface" for b in r["bounces"]) == r["n_surface"]
            assert sum(b["boundary"] == "bottom" for b in r["bounces"]) == r["n_bottom"]
            for b in r["bounces"]:
                plane = next(pl for pl in p._planes.planes if pl.kind == b["boundary"])
                assert abs(float(plane.distance(np.asarray(b["xyz"])[None])[0])) < 1e-6
                assert 0.0 <= b["grazing_deg"] <= 90.0 and b["loss_db"] is not None
            walls += r.get("walls", 0)
    assert walls > 0                                     # the quay wall's images are drawn


@pytest.mark.parametrize("engine", ["numpy", "cpp"])
def test_sediment_refracted_path_is_the_providers(engine):
    cfg, sonar, p = _resolved("buried_cylinder", engine=engine)
    src = np.array([0.1, -0.2, -1.5])
    pts = np.array([[0.0, 0.0, 0.3], [0.6, 0.4, 0.5], [0.2, 0.0, -0.4]])  # two buried, one in the water
    try:
        eig = p.eigenrays(src, pts)
    except sim_engine.EngineUnavailable as exc:
        pytest.skip(str(exc))
    for s in range(len(pts)):
        (r,) = tr._leg_rays(p, eig, s, src, pts[s], sonar.center_frequency, True, pts)
        line = np.asarray(r["polyline"])
        seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
        if pts[s, 2] > p.interface_z:
            assert len(line) == 3 and line[1, 2] == pytest.approx(p.interface_z, abs=1e-12)
            t = seg[0] / p.sound_speed + seg[1] / p.sediment_speed
            (b,) = r["bounces"]
            assert b["boundary"] == "interface" and b["loss_db"] == pytest.approx(tr.interface_loss_db(p, np.radians(b["grazing_deg"])))
        else:
            assert len(line) == 2 and not r["bounces"]
            t = seg[0] / p.sound_speed
        assert t == pytest.approx(r["delay_s"], rel=1e-9)


def test_bellhop_rays_bounce_on_a_range_dependent_bed():
    _need_bellhop()
    cfg, sonar, p = _resolved("fjord_testauv_trial_1row")
    assert p.env.bathymetry is not None
    src, pt = np.array([0.0, 0.0, 10.0]), np.array([120.0, 485.0, 15.0])
    eig = p.eigenrays(src, pt[None])
    rays = tr._leg_rays(p, eig, 0, src, pt, sonar.center_frequency, True, pt[None])
    drawn = [r for r in rays if r["polyline"] is not None]
    assert len(drawn) >= 0.8 * len(rays) and any(r["n_bottom"] for r in drawn)
    rng = lambda xyz: float(np.hypot(xyz[0] - src[0], xyz[1] - src[1]))
    for r in drawn:
        for b in r["bounces"]:
            want = 0.0 if b["boundary"] == "surface" else float(p.env.depth_at(rng(b["xyz"])))
            assert b["xyz"][2] == pytest.approx(want, abs=0.05)
        assert abs(r["polyline_delay_s"] - r["delay_s"]) < 2e-3


def test_eigenray3d_projection_is_bellhops_ray_on_the_bearings_section():
    # The 2-D projection of an eigenray3d scene: Bellhop (2-D) on the 3-D bed's section along the bearing. Checked
    # against the 2-D provider on that same section (Bellhop3D itself is not run here: minutes a scene).
    _need_bellhop()
    from gpusonar.sim.bellhop import BellhopProvider
    cfg, sonar, p3 = _resolved("flank_ridge_3d")
    src = np.array([0.0, 0.0, 30.0])
    scene = cfg["scene"]
    pt = np.asarray(scene[0]["positions"][0], float) if scene[0].get("positions") else np.array([2000.0, 0.0, 50.0])
    bed = np.asarray(tr.section_bed(p3, src, pt), float)
    d = (pt[:2] - src[:2]) / np.linalg.norm(pt[:2] - src[:2])
    np.testing.assert_allclose(bed[:, 1], p3.bed.depth(src[0] + bed[:, 0] * d[0], src[1] + bed[:, 0] * d[1]))
    env2 = tr.section_env(p3, src, pt)
    p2 = BellhopProvider(env2, p3.frequency, float(np.linalg.norm(pt[:2]) * 1.2), angles=p3.angles)
    eig = p2.eigenrays(src, pt[None])
    rays = tr._leg_rays(p3, eig, 0, src, pt, sonar.center_frequency, True, pt[None])
    drawn = [r for r in rays if r["polyline"] is not None]
    assert drawn and all(r["path"].startswith("PROJECTION") for r in drawn)
    for r in drawn:
        for b in r["bounces"]:
            if b["boundary"] == "bottom":
                assert b["xyz"][2] == pytest.approx(float(env2.depth_at(np.hypot(*np.subtract(b["xyz"][:2], src[:2])))), abs=0.05)


# --- the trace's arrivals are the simulator's -------------------------------------------------------------------------

CASES = [
    ("stripmap_points", {}),                                                     # free field
    ("fls_120k_quay_multipath", {"pings": {"count": 1}, "scene": [               # general planes
        {"type": "points", "label": "sphere", "positions": [[45.0, 8.0, 6.0]], "target_strength_db": 0.0}]}),
    ("buried_cylinder", {}),                                                     # sediment
]


@pytest.mark.parametrize("engine", ["numpy", "cpp"])
@pytest.mark.parametrize("name,overrides", CASES, ids=[c[0] for c in CASES])
def test_trace_delays_are_the_simulators_pairs_at_each_tier(name, overrides, engine):
    plan = catalogue.view_plan(load_scenario_config(name, None, expand=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _run(name, engine=engine, **deep_merge(plan["overrides"], overrides))
        trace = tr.build_trace(result, engine=engine, view=plan)
    rec = result.records[0]
    tx, elements = np.asarray(rec.tx_position()), np.asarray(rec.rx_positions())
    ref = elements.mean(axis=0)
    pts = np.array([t["position"] for t in trace["targets"]])
    q, p, amp, _, _ = prop.two_way_paths(tr.provider_for(result, engine), tx, ref, pts, np.ones(len(pts)), engine=engine)
    keep = np.abs(amp) > 0
    c = result.config["environment"]["sound_speed"]
    if trace["tier"] == "sediment":                      # apparent points at the travel-time distance
        c = tr._inner(tr.provider_for(result, engine)).sound_speed
    sim = np.sort(np.linalg.norm(q[keep] - tx, axis=1) / c + np.linalg.norm(p[keep] - ref, axis=1) / c)
    ours = np.sort([pp["delay_s"] for pp in trace["pairs"]])
    assert len(ours) == len(sim) > 0
    np.testing.assert_allclose(ours, sim, rtol=0, atol=1e-9)
    assert not trace["missing"] or all("eigenray(s) not drawn" not in m for m in trace["missing"])


def test_trace_delays_are_the_simulators_pairs_at_the_eigenray_tier():
    _need_bellhop()
    result = _run("flank_target_1km", engine="numpy", reverberation=[], sonar_overrides={"array": {"count": 4}},
                  receiver={"ambient_noise": {"level_db": 0.0}})
    trace = tr.build_trace(result, engine="numpy")
    rec = result.records[0]
    tx, elements = np.asarray(rec.tx_position()), np.asarray(rec.rx_positions())
    ref = elements.mean(axis=0)
    pts = np.array([t["position"] for t in trace["targets"]])
    q, p, _, _, _ = prop.two_way_paths(tr.provider_for(result, "numpy"), tx, ref, pts, np.ones(len(pts)), engine="numpy")
    out = tr.provider_for(result, "numpy").eigenrays(tx, pts)
    c = out.sound_speed
    sim = np.sort(np.linalg.norm(q - tx, axis=1) / c + np.linalg.norm(p - ref, axis=1) / c)
    ours = np.sort([pp["delay_s"] for pp in trace["pairs"]])
    np.testing.assert_allclose(ours, sim, rtol=0, atol=1e-9)


def test_one_way_sources_and_the_direct_sound_are_the_simulators_paths():
    from gpusonar.sim import sources as source_models
    result = _run("sst_asiaex", engine="numpy")
    trace = tr.build_trace(result, engine="numpy")
    (src,) = trace["oneway"]
    provider = tr.provider_for(result, "numpy")
    rec = result.records[0]
    elements = np.asarray(rec.rx_positions())
    ref = elements.mean(axis=0)
    p, amp, _ = source_models.one_way_paths(provider, ref, np.asarray(src["position"]))
    want = np.sort(np.linalg.norm(p[:, None, :] - elements[None], axis=2) / provider.eigenrays(ref, ref[None] + 1.0).sound_speed, axis=0)
    got = np.sort(np.array([a["channel_delay_s"] for a in src["arrivals"]]), axis=0)
    np.testing.assert_allclose(got, want, atol=1e-9)
    # Time spreading: the arrivals are the wrapper's taps (what is synthesised), the rays the bounce eigenrays under them.
    assert sum(r["polyline"] is not None for r in src["rays"]) == len(src["rays"]) == 5 < len(src["arrivals"])
    # The projector heard directly (transmit.direct_path, with its boundaries): one-way from the projector.
    result = _run("popoto_testauv_direct_overload", engine="numpy")
    trace = tr.build_trace(result, engine="numpy")
    rec = result.records[0]
    elements, tx = np.asarray(rec.rx_positions()), np.asarray(rec.tx_position())
    provider = tr.provider_for(result, "numpy")
    p, _, _ = source_models.one_way_paths(provider, elements.mean(axis=0), tx)
    want = np.sort(np.linalg.norm(p[:, None, :] - elements[None], axis=2) / provider.sound_speed, axis=0)
    got = np.sort(np.array([a["channel_delay_s"] for a in trace["direct"]["arrivals"]]), axis=0)
    np.testing.assert_allclose(got, want, atol=1e-9)
    assert any(r["n_surface"] or r["n_bottom"] for r in trace["direct"]["rays"])


# --- the boundary and medium models a scenario uses -------------------------------------------------------------------

def _models(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = tr.static_trace(name)
    return t, {m["model"]: m for m in t["models"]}


def test_rough_sediment_small_slope_is_listed_and_its_curve_drawn():
    for name in ("flank_sst_shallow", "flank_sst_spread", "sst_sediment_reverb"):
        t, m = _models(name)
        assert m["fluid_sediment"]["value"]["roughness_model"] == "small_slope", name
        assert any(c["model"] == "fluid_sediment" and len(c["strength_db"]) == 90 for c in t["scattering"]), name


def test_surface_roughness_and_bubble_loss_are_listed_and_drawn():
    t, m = _models("flank_sst_spread")
    assert m["bubble_loss"]["value"] == 0.5 and "rough_surface_height" in m
    s = t["environment"]["surface"]
    assert s["rms_height"] > 0 and s["bubble_loss"]["loss_db"] == [0.5, 0.5]
    t, m = _models("asiaex_dahl_2004")
    assert m["rough_surface_height"]["value"]["rms_height"] == pytest.approx(0.3)
    assert t["environment"]["surface"]["rms_height"] == pytest.approx(0.3)


def test_penetration_general_planes_noise_statistics_direct_and_overload_are_listed():
    t, m = _models("buried_cylinder")
    assert m["penetration"]["value"]["penetration"] == "fluid" and "surface_pressure_release" not in m
    assert t["environment"]["interface"]["critical_deg"] == pytest.approx(np.degrees(np.arccos(1500.0 / 1700.0)))
    _, m = _models("fls_120k_quay_multipath")
    assert m["general_planes"]["value"]["walls"] == 1 and "scene_seafloor" in m
    _, m = _models("fjord_testauv_trial_1row")
    assert m["noise_statistics"]["value"]["model"] == "k" and "bathymetry" in m
    _, m = _models("ssp_front_testauv_trial_1row")
    assert "ssp_range" in m
    _, m = _models("popoto_testauv_direct_overload")
    assert "direct_path" in m and "overload" in m
    _, m = _models("fls_glints_on_sediment")
    assert m["scene_seafloor"]["value"]["model"] == "fluid_sediment"


def test_models_no_scenario_uses_are_listed_when_configured():
    # No scenario file uses these (counted 2026-11-27): the list still names them when a config does.
    base = {"fidelity": "image_source", "environment": {"sound_speed": 1500.0},
            "propagation": {"bottom_z": 50.0, "bottom": {"speed": 3000.0, "density": 2.5, "attenuation": 0.1, "shear_speed": 1400.0}},
            "reverberation": [{"model": "fluid_sediment", "sediment": "medium_sand", "roughness_model": "williams_jackson_1998"}]}
    names = {m["model"] for m in tr.models_in_force(base)}
    assert {"rayleigh_elastic", "williams_jackson_1998", "fluid_sediment"} <= names
    sed = {"fidelity": "sediment", "environment": {"sound_speed": 1500.0},
           "propagation": {"interface_z": 0.0, "penetration": "rough", "roughness": {"sediment": "medium_sand", "K0": 0.0}}}
    (pen,) = [m for m in tr.models_in_force(sed) if m["model"] == "penetration"]
    assert pen["value"]["roughness"] == {"sediment": "medium_sand", "K0": 0.0}


# --- the pages ---------------------------------------------------------------------------------------------------------

QUICK_PAGE = {"reverberation": [], "receiver": {"ambient_noise": {"level_db": 0.0}}}


@pytest.mark.parametrize("family", [s for _, s, _ in catalogue.FAMILIES])
def test_the_page_of_each_family_builds(family, tmp_path):
    # Quick form: the family's example scenario, one ping, no reverberation (it costs time and is not what a ray page
    # draws), the densest scenes thinned; Bellhop families skip without it.
    name = catalogue.FAMILY_EXAMPLE[family]
    cfg = load_scenario_config(name, None, expand=False)
    if cfg.get("fidelity") in ("eigenray", "eigenray3d"):
        _need_bellhop()
    plan = catalogue.view_plan(cfg, quick=True)
    over = deep_merge(plan["overrides"], QUICK_PAGE)
    if (cfg.get("receiver") or {}).get("ambient_noise") is None:
        over.pop("receiver")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _run(name, engine="numpy", **over)
        trace = tr.build_trace(result, engine="numpy", view=plan)
    drawn = page2d.draw_page(trace, result.records[0], str(tmp_path / "p.png"))
    assert os.path.getsize(tmp_path / "p.png") < 300_000
    assert drawn["rays"] + drawn.get("oneway_rays", 0) > 0
    assert len(trace["missing"]) == 0 or all("not computed" not in m for m in trace["missing"])
