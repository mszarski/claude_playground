"""The 3-D scene page (python/gpusonar/viz/page3d.py, scene3d.py; docs/tasks/VISUALIZATION.md package V4).

What is held: the page's embedded data are the trace's (ray count, delays, bounce points on the surface and bed
within rounding, the end points on the target) at each tier: free field, image (flat and general planes), sediment,
`eigenray` (Bellhop) and `eigenray3d` (Bellhop3D's own 3-D rays); the Bellhop3D ray reader on a hand-made file in
the format the Acoustics Toolbox writes, and its bounce count against the program's own; the shooting finds, for
image eigenrays on a level bed, Bellhop3D rays that bounce on the surface and the bed and arrive within a millisecond
of the image delay; page sizes stay under their limits (a CDN page 400 kB, an embedded page 1.2 MB) even for a trace
far larger than any scenario's; every scenario yields a 3-D page in quick form (no simulation); the pages load in
headless Chromium with no console error (only where Node and Playwright are installed: skipped otherwise).
"""

import hashlib
import json
import os
import shutil
import subprocess
import warnings

import numpy as np
import pytest
from gpusonar.sim import engine as sim_engine
from gpusonar.sim.simulator import run_scenario
from gpusonar.viz import page3d, scene3d
from gpusonar.viz import trace as tr

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SCENARIOS = sorted(f[:-5] for f in os.listdir(os.path.join(ROOT, "configs", "scenarios")) if f.endswith(".yaml"))
QUICK = {"reverberation": [], "receiver": {"ambient_noise": {"level_db": 0.0}}}


def _run(name, engine="numpy", **overrides):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return run_scenario(name, overrides=overrides or None, engine=engine)
    except sim_engine.EngineUnavailable as exc:
        pytest.skip(str(exc))


def _blob(path):
    text = open(path, encoding="utf-8").read()
    return json.loads(text.split('<script type="application/json" id="trace">', 1)[1].split("</script>", 1)[0].replace("<\\/", "</"))


def _page_equals_trace(trace, data, tol=0.006):
    """Every leg's page rays are the strongest drawable rays of the trace's leg, with the same delays, bounce counts
    and bounce points (to the page's rounding), and each ends where its trace ray ends."""
    n = 0
    for leg, pleg in zip(trace["legs"], data["legs"]):
        for which in ("out", "back"):
            drawable = sorted([r for r in leg[which] if r.get("polyline") is not None or r.get("bellhop3d")],
                              key=lambda r: -r["amplitude"])[:data["limits"]["max_rays"]]
            assert [r["k"] for r in pleg[which]] == [r["k"] for r in drawable]
            for r, p in zip(drawable, pleg[which]):
                src = r.get("bellhop3d") or r
                assert p["delay"] == pytest.approx(r["delay_s"], abs=1e-8)
                assert (p["ns"], p["nb"]) == (r["n_surface"], r["n_bottom"])
                assert len(p["bounces"]) == len(src["bounces"])
                for b, pb in zip(src["bounces"], p["bounces"]):
                    assert pb["b"] == b["boundary"]
                    assert np.allclose(pb["xyz"], b["xyz"], atol=tol)
                    assert any(np.allclose(v, pb["xyz"], atol=tol) for v in p["line"])     # a vertex of the drawn line
                assert np.allclose(p["line"][0], src["polyline"][0], atol=tol)
                assert np.allclose(p["line"][-1], src["polyline"][-1], atol=tol)
                n += 1
    return n


def _surface_and_bed(trace, data):
    """Bounce points of the page lie on the surface (z = surface_z) or on the flat bed (z = bottom_z)."""
    e = trace["environment"]
    for leg in data["legs"]:
        for which in ("out", "back"):
            for r in leg[which]:
                for b in r["bounces"]:
                    if b["b"] == "surface":
                        assert b["xyz"][2] == pytest.approx(e["surface_z"], abs=0.006)
                    elif b["b"] == "bottom" and e.get("bottom_z") is not None and not e.get("planes") and not e.get("bathymetry"):
                        assert b["xyz"][2] == pytest.approx(e["bottom_z"], abs=0.006)


# --- the page holds the trace, per tier ------------------------------------------------------------------------------

@pytest.mark.parametrize("name, overrides", [
    ("sst_echo_level", {"fidelity": "free_field"}),
    ("sst_echo_level", {}),                                          # image, flat
    ("buried_cylinder", {}),                                         # sediment
])
def test_page_data_equals_the_trace(name, overrides, tmp_path):
    from gpusonar.config import deep_merge, load_scenario_config
    from gpusonar.viz import catalogue
    plan = catalogue.view_plan(load_scenario_config(name, None, expand=False), quick=True)
    result = _run(name, **deep_merge(plan["overrides"], overrides))
    trace = tr.build_trace(result, engine="numpy", view=plan)
    scene3d.enrich(trace, name)
    info = page3d.write_page(trace, str(tmp_path / "p.html"))
    data = _blob(tmp_path / "p.html")
    n = _page_equals_trace(trace, data)
    assert n == info["rays"] > 0
    _surface_and_bed(trace, data)
    if overrides.get("fidelity") == "free_field":
        assert all(not r["bounces"] and len(r["line"]) == 2 for leg in data["legs"] for w in ("out", "back") for r in leg[w])
    assert data["chain"] is None                                  # reserved for the live chain's taps (V6)


def test_page_data_of_general_planes_and_their_planes(tmp_path):
    # The quay: a tilted bed and a wall as planes; the page carries the planes it draws and the wall bounces.
    from gpusonar.config import deep_merge, load_scenario_config
    from gpusonar.viz import catalogue
    name = "fls_120k_quay_multipath"
    plan = catalogue.view_plan(load_scenario_config(name, None, expand=False), quick=True)
    result = _run(name, **deep_merge(plan["overrides"], {"reverberation": []}))
    trace = tr.build_trace(result, engine="numpy", view=plan)
    page3d.write_page(trace, str(tmp_path / "p.html"))
    data = _blob(tmp_path / "p.html")
    assert _page_equals_trace(trace, data) > 0
    kinds = sorted(p["kind"] for p in data["environment"]["planes"])
    assert kinds == ["bottom", "surface", "wall"]
    assert any(b["b"] == "wall" for leg in data["legs"] for w in ("out", "back") for r in leg[w] for b in r["bounces"])


def test_page_data_of_the_eigenray_tier_is_bellhops_planar_rays(tmp_path):
    from gpusonar.sim import bellhop
    if bellhop.find_bellhop() is None:
        pytest.skip("Bellhop is not installed (GPUSONAR_BELLHOP)")
    from gpusonar.config import deep_merge, load_scenario_config
    from gpusonar.viz import catalogue
    name = "popoto_trial_100m"
    plan = catalogue.view_plan(load_scenario_config(name, None, expand=False), quick=True)
    result = _run(name, **deep_merge(plan["overrides"], QUICK))
    trace = tr.build_trace(result, engine="numpy", view=plan)
    page3d.write_page(trace, str(tmp_path / "p.html"))
    data = _blob(tmp_path / "p.html")
    assert _page_equals_trace(trace, data) > 0
    _surface_and_bed(trace, data)
    for leg in data["legs"]:                          # planar: every point of a ray on the vertical plane of its bearing
        t = np.asarray(trace["targets"][leg["target"]]["position"])
        for which, o in (("out", trace["sonar"]["projector"]), ("back", np.mean(trace["sonar"]["hydrophones"], axis=0))):
            u = (t[:2] - np.asarray(o)[:2]) / np.linalg.norm(t[:2] - np.asarray(o)[:2])
            for r in leg[which]:
                d = np.asarray(r["line"])[:, :2] - np.asarray(o)[:2]
                assert np.allclose(d[:, 0] * u[1] - d[:, 1] * u[0], 0.0, atol=0.02)


# --- Bellhop3D's rays ------------------------------------------------------------------------------------------------

RAYFILE = """ 'BELLHOP3D- hand-made                                                    '
   8000.0000000000000
           1           1           1
           2           1
   0.0000000000000000
   100.00000000000000
 'xyz'
  0.10000000000000001
           4           0           1
   0.0000000000000000        0.0000000000000000        50.000000000000000
   0.0000000000000000        250.00000000000000        75.000000000000000
   0.0000000000000000        500.00000000000000        100.00000000000000
   0.0000000000000000        500.00000000000000        100.00000000000000
 -0.10000000000000001
           6           1           0
   0.0000000000000000        0.0000000000000000        50.000000000000000
   0.0000000000000000        250.00000000000000        25.000000000000000
   0.0000000000000000        500.00000000000000        0.0000000000000000
   0.0000000000000000        500.00000000000000        0.0000000000000000
   10.000000000000000        750.00000000000000        25.000000000000000
   20.000000000000000        1000.0000000000000        50.000000000000000
"""


def test_bellhop3d_ray_reader_reads_the_documented_format(tmp_path):
    # The format: Bellhop/ReadEnvironmentBell.f90 (header) and WriteRay.f90 WriteRay3D (alpha in radians, then the
    # count and the surface and bottom bounces of the whole ray, then x y z per point; a reflection is the point
    # written twice). The second ray turns 10 m north after its surface bounce: out of its vertical plane.
    path = tmp_path / "run.ray"
    path.write_text(RAYFILE)
    f = scene3d.read_rays3d(str(path))
    assert (f["frequency"], f["n_alpha"], f["n_beta"], f["top"], f["bottom"]) == (8000.0, 2, 1, 0.0, 100.0)
    a, b = f["rays"]
    assert a["alpha_deg"] == pytest.approx(np.degrees(0.1)) and b["alpha_deg"] == pytest.approx(-np.degrees(0.1))
    assert (a["n_top"], a["n_bottom"], len(a["xyz"])) == (0, 1, 4) and (b["n_top"], b["n_bottom"], len(b["xyz"])) == (1, 0, 6)
    assert scene3d.ray_bounces(a["xyz"]) == [(2, "bottom")] and scene3d.ray_bounces(b["xyz"]) == [(2, "surface")]
    line, miss, bnc = scene3d.cut_ray(b["xyz"], [10.0, 750.0, 25.0])
    assert miss == pytest.approx(0.0, abs=1e-9) and np.allclose(line[-1], [10.0, 750.0, 25.0]) and bnc == [(2, "surface")]
    line, miss, bnc = scene3d.cut_ray(b["xyz"], [0.0, 250.0, 25.0])
    assert miss == pytest.approx(0.0, abs=1e-9) and bnc == []         # before the bounce: none counted
    assert scene3d.travel_time(b["xyz"][:3], lambda z: 1500.0) == pytest.approx(2 * np.hypot(250.0, 25.0) / 1500.0)
    with pytest.raises(ValueError):
        (tmp_path / "bad.ray").write_text(RAYFILE.replace("'xyz'", "'rz'"))
        scene3d.read_rays3d(str(tmp_path / "bad.ray"))


def test_bellhop3d_rays_on_a_level_bed_are_the_image_paths():
    # A level 100 m bed, isovelocity: the image tier's eigenrays (launch angles, bounce counts, delays) are exact;
    # Bellhop3D's rays shot about them must bounce on the surface and the bed with the same counts, pass close to the
    # target and arrive within 1 ms of the image delay. A separate process (GPL-3), as the provider runs it.
    from gpusonar.sim import bellhop3d as b3
    from gpusonar.sim import propagation as prop
    from gpusonar.sim.bellhop import OceanEnvironment
    if b3.find_bellhop3d() is None:
        pytest.skip("Bellhop3D is not installed (GPUSONAR_BELLHOP3D)")
    env = OceanEnvironment(depth=100.0, ssp_depth=[0.0, 100.0], ssp_speed=[1500.0, 1500.0], bottom_speed=1600.0,
                     bottom_density=1.8, bottom_attenuation=0.5)
    p = b3.Bellhop3DProvider(env, 8000.0, max_range=1200.0)
    img = prop.ImageSourceProvider(1500.0, surface_z=0.0, bottom_z=100.0, bottom_coefficient=0.5, max_order=2, engine="numpy")
    src, tgt = np.array([0.0, 0.0, 30.0]), np.array([300.0, 900.0, 60.0])
    eig = img.eigenrays(src, tgt[None])
    rays = tr._leg_rays(img, eig, 0, src, tgt, 8000.0, True, tgt[None])
    rays = [r for r in rays if r["n_surface"] + r["n_bottom"] <= 2]
    res = scene3d.bellhop3d_leg(p, src, tgt, rays, 8000.0)
    assert res["runs"] >= 1
    for r in rays:
        m = res["rays"][r["k"]]
        assert m is not None, r
        assert m["miss_m"] < 3.0
        assert abs(m["travel_time_s"] - r["delay_s"]) < 1e-3
        assert [b["boundary"] for b in m["bounces"]] == [b["boundary"] for b in r["bounces"]]
        for b, rb in zip(m["bounces"], r["bounces"]):
            assert b["xyz"][2] == pytest.approx(0.0 if b["boundary"] == "surface" else 100.0, abs=0.05)
            assert np.allclose(b["xyz"], rb["xyz"], atol=4.0)            # the image's bounce point, within the miss
            assert b["grazing_deg"] == pytest.approx(rb["grazing_deg"], abs=0.5)


def test_eigenray3d_page_uses_bellhop3ds_rays():
    # A hand-made eigenray3d trace whose rays carry `bellhop3d` (as add_bellhop3d_rays leaves them): the page draws
    # those, not the 2-D projection, and says so; a ray without one keeps its projection.
    line3 = [[0.0, 0.0, 50.0], [5.0, 500.0, 100.0], [5.0, 500.0, 100.0], [12.0, 1000.0, 60.0]]
    b3r = {"polyline": line3, "bounces": [{"xyz": line3[1], "boundary": "bottom", "grazing_deg": 5.7, "loss_db": 0.4}],
           "miss_m": 1.5, "travel_time_s": 0.6751, "alpha_deg": 5.7, "beta_deg": 89.4, "n_points": 4}
    ray = {"k": 0, "delay_s": 0.675, "amplitude": 1e-3, "n_surface": 0, "n_bottom": 1, "launch_deg": 5.7,
           "polyline": [[0.0, 0.0, 50.0], [0.0, 500.0, 100.0], [0.0, 1000.0, 60.0]],
           "bounces": [{"xyz": [0.0, 500.0, 100.0], "boundary": "bottom", "grazing_deg": 5.7, "loss_db": 0.4}],
           "path": "PROJECTION", "bellhop3d": b3r}
    ray2 = dict(ray, k=1, amplitude=5e-4, bellhop3d=None)
    trace = {"scenario": "hand", "tier": "eigenray3d", "ping": 0,
             "sonar": {"name": "s", "mode": "flank", "band_hz": [6e3, 1e4], "center_frequency": 8e3, "sample_kind": "x",
                       "sample_rate": 1.0, "n_channels": 2, "projector": [0.0, 0.0, 50.0],
                       "hydrophones": [[-1.0, 0.0, 50.0], [1.0, 0.0, 50.0]]},
             "environment": {"sound_speed": 1500.0, "surface_z": 0.0, "bottom_z": 100.0},
             "targets": [{"label": "t", "position": [12.0, 1000.0, 60.0], "amplitude": 1.0}],
             "legs": [{"target": 0, "out": [ray, ray2], "back": []}], "pairs": [], "models": [], "missing": []}
    data = page3d.page_data(trace)
    (a, b) = data["legs"][0]["out"]
    assert a["three_d"] and a["line"] == [line3[0], line3[1], line3[3]] and a["miss"] == 1.5     # the repeated point dropped and "Bellhop3D" in a["path"]
    assert not b.get("three_d") and b["path"] == "PROJECTION"
    assert _page_equals_trace(trace, data) == 2


# --- sizes, the vendored library, every scenario ---------------------------------------------------------------------

def _big_trace(n_targets=6, n_rays=30, n_points=4000, n_pairs=800, n_ch=96):
    rng = np.random.default_rng(1)
    def ray(k):
        r = np.linspace(0.0, 12000.0, n_points)
        line = np.column_stack([0.3 * r, r, 100.0 + 90.0 * np.sin(r / 700.0 + k) + rng.normal(0, 0.5, n_points)]).tolist()
        return {"k": k, "delay_s": 8.0 + 0.01 * k, "amplitude": 1.0 / (1 + k), "n_surface": k % 3, "n_bottom": k % 4,
                "launch_deg": 3.0, "polyline": line, "path": "synthetic",
                "bounces": [{"xyz": line[i], "boundary": "bottom", "grazing_deg": 4.0, "loss_db": 0.5} for i in range(100, n_points, 400)]}
    return {"scenario": "big", "tier": "eigenray", "ping": 0,
            "sonar": {"name": "s", "mode": "flank", "band_hz": [4e3, 6e3], "center_frequency": 5e3, "sample_kind": "x",
                      "sample_rate": 1.0, "n_channels": n_ch, "projector": [0.0, 0.0, 30.0],
                      "hydrophones": [[0.1 * i, 0.0, 30.0 + (i // 32)] for i in range(n_ch)]},
            "environment": {"sound_speed": 1500.0, "surface_z": 0.0, "bottom_z": 200.0},
            "targets": [{"label": f"t{i}", "position": [3600.0, 12000.0, 50.0], "amplitude": 1.0} for i in range(n_targets)],
            "legs": [{"target": i, "out": [ray(k) for k in range(n_rays)], "back": [ray(k) for k in range(n_rays)]} for i in range(n_targets)],
            "pairs": [{"target": 0, "out": 0, "back": 0, "delay_s": 16.0 + 1e-3 * i, "channel_delay_s": (16.0 + rng.random(n_ch)).tolist(),
                       "n_surface": 0, "n_bottom": 0, "level_db": -float(i)} for i in range(n_pairs)],
            "models": [], "missing": []}


def test_page_sizes_stay_under_their_limits(tmp_path):
    # Limits (ours): a CDN page under 400 kB, an embedded page under 1.2 MB (0.69 MB of it three.js). A trace far
    # larger than any scenario's (6 targets x 60 rays of 4000 points, 800 pairs x 96 hydrophones) is thinned to fit.
    trace = _big_trace()
    cdn = page3d.write_page(trace, str(tmp_path / "c.html"), three="cdn")
    assert cdn["bytes"] < page3d.PAGE_LIMIT_CDN and not cdn["limits"].get("over_budget")
    assert cdn["rays"] > 0 and cdn["pairs"] > 0
    if os.path.exists(page3d.VENDORED):
        emb = page3d.write_page(trace, str(tmp_path / "e.html"), three="embed")
        assert emb["embedded"] and emb["bytes"] < page3d.PAGE_LIMIT_EMBED
    # A small scene is not thinned at all.
    small = _big_trace(n_targets=1, n_rays=3, n_points=50, n_pairs=5, n_ch=4)
    info = page3d.write_page(small, str(tmp_path / "s.html"), three="cdn")
    assert info["limits"]["tolerance"] == 1.0 and info["rays"] == 6 and info["pairs"] == 5


def test_vendored_three_is_the_stated_file_and_the_default():
    if not os.path.exists(page3d.VENDORED):
        pytest.skip("third_party/three/three.module.min.js is not present")
    digest = hashlib.sha256(open(page3d.VENDORED, "rb").read()).hexdigest()
    readme = open(os.path.join(os.path.dirname(page3d.VENDORED), "README.md"), encoding="utf-8").read()
    assert digest in readme and os.path.getsize(page3d.VENDORED) == 691_648
    assert os.path.exists(os.path.join(os.path.dirname(page3d.VENDORED), "LICENSE"))
    assert page3d.three_source(None) == page3d.VENDORED and page3d.three_source("cdn") is None
    # Never imported by the package: no Python file under python/gpusonar names the vendored file but page3d.py.
    pkg = os.path.join(ROOT, "python", "gpusonar")
    users = [os.path.relpath(os.path.join(d, f), pkg) for d, _, fs in os.walk(pkg) for f in fs if f.endswith(".py")
             and "three.module" in open(os.path.join(d, f), encoding="utf-8").read()]
    assert users == [os.path.join("viz", "page3d.py")]


@pytest.mark.parametrize("name", SCENARIOS)
def test_every_scenario_yields_a_3d_page_in_quick_form(name, tmp_path):
    # No simulation and no provider call (static_trace), no Bellhop3D: geometry, boundaries, beams, models.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tr.static_trace(name)
        scene3d.enrich(trace, name, bellhop3d=False)
    info = page3d.write_page(trace, str(tmp_path / f"{name}.html"), three="cdn")
    assert info["bytes"] < page3d.PAGE_LIMIT_CDN
    data = _blob(tmp_path / f"{name}.html")
    assert data["models"] and data["sonar"]["hydrophones"] and data["scene3d"]["box"]
    if trace["tier"] == "eigenray3d":
        g = data["scene3d"]["bed_grid"]
        assert len(g["z"]) == len(g["y"]) and len(g["z"][0]) == len(g["x"])
        assert any("2-D projection" in n for n in data["scene3d"]["notes"])


def test_target_motion_over_the_pings():
    # popoto_far_mover: a target with a velocity; its positions over the pings are the traced one plus v t.
    trace = tr.static_trace("popoto_far_mover")
    cfg, _, scene = scene3d.resolved("popoto_far_mover")
    m = scene3d.motion(trace, cfg, scene)
    assert m["moving"] and len(m["times"]) == len(m["ping_index"]) > 1
    i = [t["label"] for t in trace["targets"]].index("small_vehicle") if any(t["label"] == "small_vehicle" for t in trace["targets"]) else 0
    pos = np.asarray(m["targets"][i])
    v = (pos[-1] - pos[0]) / (m["times"][-1] - m["times"][0])
    assert np.linalg.norm(v) > 0.1
    assert np.allclose(pos[0], trace["targets"][i]["position"], atol=0.01)


# --- the pages in a browser (opt-in: Node and Playwright with Chromium) ----------------------------------------------

def _browser():
    node = shutil.which("node")
    if node is None:
        return None, "Node is not installed"
    probe = subprocess.run([node, "-e", "require('playwright').chromium.launch({args:['--use-gl=angle','--use-angle=swiftshader',"
                            "'--enable-unsafe-swiftshader']}).then(b=>b.close()).then(()=>process.exit(0),()=>process.exit(3))"],
                           capture_output=True, text=True, timeout=120)
    if probe.returncode != 0:
        return None, "Playwright with Chromium is not available to Node"
    return node, None


def test_pages_load_in_a_browser_without_console_errors(tmp_path):
    node, why = _browser()
    if node is None:
        pytest.skip(why)
    if not os.path.exists(page3d.VENDORED):
        pytest.skip("third_party/three/three.module.min.js is not present (pages would need the network)")
    result = _run("sst_echo_level")
    trace = tr.build_trace(result, engine="numpy")
    scene3d.enrich(trace)
    page3d.write_page(trace, str(tmp_path / "sst_echo_level.html"), three="embed")
    for name in ("flank_seamount_3d", "buried_cylinder", "fls_120k_quay_multipath", "popoto_far_mover"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = tr.static_trace(name)
            scene3d.enrich(t, name, bellhop3d=False)
        page3d.write_page(t, str(tmp_path / f"{name}.html"), three="embed")
    out = subprocess.run([node, os.path.join(ROOT, "tools", "page3d_check.js"), str(tmp_path)], capture_output=True,
                         text=True, timeout=600)
    lines = [json.loads(x) for x in out.stdout.splitlines() if x.startswith("{")]
    assert out.returncode == 0 and len(lines) == 5, out.stdout + out.stderr
    assert all(x["ok"] and x["info"]["hydrophones"] > 0 for x in lines)
