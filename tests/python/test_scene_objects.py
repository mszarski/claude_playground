"""Extended scene objects: wires, plates, spheres, boxes, pool walls, seafloor relief;
surface normals and acoustic shadows."""

import numpy as np
import pytest

from gpusonar import recon
from gpusonar.sim import FreeFieldProvider, Scene, build_scene, make_pose, simulate_ping
from gpusonar.sim.propagation import two_way_paths
from gpusonar.sonar import SonarModel

C = 1500.0


def rng():
    return np.random.default_rng(4)


def test_wire_is_a_line_of_points():
    scene = build_scene([{"type": "wire", "start": [1.0, 0.0, -0.5], "end": [1.0, 0.0, 0.5], "spacing": 0.01,
                          "target_strength_db": -40.0}], rng())
    assert len(scene) == 101
    assert np.allclose(scene.positions[:, :2], [1.0, 0.0])
    assert np.allclose(np.diff(scene.positions[:, 2]), 0.01)
    assert np.allclose(np.abs(scene.amplitudes), 10 ** (-40 / 20))


def test_plate_scatters_only_from_its_front():
    # A 1 m plate in the plane x = 5, normal towards -x (towards a sonar at the origin).
    item = {"type": "plate", "origin": [5.0, -0.5, -0.5], "edge_u": [0.0, 0.0, 1.0], "edge_v": [0.0, 1.0, 0.0],
            "density": 400.0, "mu_db": -20.0}
    scene = build_scene([item], rng())
    assert np.allclose(scene.normals, [-1.0, 0.0, 0.0])
    provider = FreeFieldProvider(C)
    front = two_way_paths(provider, np.zeros(3), np.zeros(3), scene.positions, scene.amplitudes,
                          lambert=scene.lambert, normals=scene.normals)
    behind = two_way_paths(provider, np.array([10.0, 0, 0]), np.array([10.0, 0, 0]), scene.positions,
                           scene.amplitudes, lambert=scene.lambert, normals=scene.normals)
    assert len(front[2]) == len(scene) and len(behind[2]) == 0
    # Near normal incidence the Lambert factor is close to 1 (cos about 1), and the summed
    # cross-section is mu x area.
    # (two_way_paths returns amplitudes with spherical spreading taken out.)
    assert np.sum(np.abs(front[2]) ** 2) == pytest.approx(10 ** (-20 / 10) * 1.0, rel=0.25)


def test_sphere_and_box_shadows():
    scene = Scene()
    scene.occluders.append(("sphere", [5.0, 0.0, 0.0], 0.5))
    scene.occluders.append(("box", [4.0, 3.0, -0.5], [5.0, 4.0, 0.5]))
    sensor = np.zeros(3)
    pts = np.array([[8.0, 0.0, 0.0],      # straight behind the sphere
                    [8.0, 0.7, 0.0],      # just inside the sphere's shadow cone (edge at 0.80 m)
                    [8.0, 0.9, 0.0],      # just outside it
                    [3.0, 0.0, 0.0],      # in front of the sphere
                    [4.5, 0.0, 0.0],      # the sphere's near pole
                    [5.5, 0.0, 0.0],      # its far pole
                    [8.0, 6.2, 0.0],      # behind the box (the ray passes through y = 3.1..3.9)
                    [8.0, 8.5, 0.0],      # past the box
                    [4.0, 3.5, 0.0],      # on the box's front face
                    [5.0, 3.5, 0.0]])     # on its back face
    expected = [False, False, True, True, True, False, False, True, True, False]
    assert scene.visible(sensor, pts).tolist() == expected
    # A sensor inside a box sees nothing through its walls.
    assert not scene.visible(np.array([4.5, 3.5, 0.0]), np.array([[9.0, 3.5, 0.0]]))[0]


def test_sphere_points_have_outward_normals_and_half_is_lit():
    scene = build_scene([{"type": "sphere", "centre": [5.0, 0.0, 0.0], "radius": 0.5, "density": 2000.0}], rng())
    outward = scene.positions - np.array([5.0, 0.0, 0.0])
    assert np.allclose(np.linalg.norm(outward, axis=1), 0.5)
    assert np.allclose(scene.normals, outward / 0.5)
    q, p, amp, _, _ = two_way_paths(FreeFieldProvider(C), np.zeros(3), np.zeros(3), scene.positions,
                                    scene.amplitudes, lambert=scene.lambert, normals=scene.normals)
    lit = len(amp) / len(scene)
    assert 0.40 < lit < 0.50                        # slightly less than half from a finite distance
    # Occlusion agrees with the normals, up to a thin band at the limb that the occluder's
    # tolerance lets through (those points have no Lambert return anyway).
    assert scene.visible(np.zeros(3)).mean() == pytest.approx(lit, abs=0.035)


def test_pool_walls_face_inwards():
    scene = build_scene([{"type": "pool", "x_range": [0.0, 10.0], "y_range": [-3.0, 3.0], "z_range": [0.0, 2.0],
                          "density": 50.0}], rng())
    centre = np.array([5.0, 0.0, 1.0])
    assert np.all(((centre - scene.positions) * scene.normals).sum(axis=1) > 0)
    assert not np.any(np.isclose(scene.positions[:, 2], 0.0) & np.isclose(np.abs(scene.normals[:, 2]), 1.0))
    assert not scene.occluders


def test_ripples_modulate_the_seafloor_echo():
    """Ripples with crests across the line of sight: slopes facing the sonar return more
    than slopes facing away, which a flat floor cannot show."""
    item = {"type": "seafloor", "x_range": [4.0, 8.0], "y_range": [-0.5, 0.5], "density": 3000.0,
            "lambert_mu_db": -27.0, "ripples": {"amplitude": 0.05, "wavelength": 0.5, "direction_deg": 0.0}}
    scene = build_scene([item], rng())
    assert np.ptp(scene.positions[:, 2]) == pytest.approx(0.1, abs=0.005)
    assert np.all(scene.normals[:, 2] < 0)                        # normals point up (NED)
    sensor = np.array([0.0, 0.0, -1.0])
    _, _, amp, _, _ = two_way_paths(FreeFieldProvider(C), sensor, sensor, scene.positions,
                                    np.ones(len(scene)), lambert=scene.lambert, normals=scene.normals)
    facing = scene.normals[:, 0] < -0.2                           # slope tilted towards the sonar
    away = scene.normals[:, 0] > 0.2
    full = np.zeros(len(scene))
    seen = np.ones(len(scene), dtype=bool)
    r = np.linalg.norm(scene.positions - sensor, axis=1)
    cos = -(((scene.positions - sensor) / r[:, None]) * scene.normals).sum(axis=1)
    seen &= cos > 0
    full[seen] = np.abs(amp) * r[seen] ** 2
    assert full[facing].mean() > 2.0 * full[away].mean()


def test_object_casts_a_shadow_in_the_image():
    """A sphere on a flat seafloor, imaged by the forward-looking sonar: the floor behind
    the sphere is dark, the floor beside it is not."""
    sonar = SonarModel.load("fls_300k", {
        "receive": {"kind": "baseband", "sample_rate": 150e3, "window": {"range": [3.0, 10.5]}, "tvg": None}})
    scene = build_scene([
        {"type": "seafloor", "x_range": [3.5, 10.0], "y_range": [-2.0, 2.0], "density": 250.0, "lambert_mu_db": -27.0},
        {"type": "sphere", "centre": [5.0, 0.0, -0.3], "radius": 0.3, "density": 3000.0, "mu_db": -15.0},
    ], rng())
    pose = make_pose(position=(0.0, 0.0, -1.0))
    rec = simulate_ping(sonar, pose, scene, environment={"sound_speed": C}, absorption_model="none", adc=False)
    ranges = np.arange(4.0, 9.5, 0.03)
    bearings = np.radians(np.arange(-14.0, 14.01, 0.25))
    img = recon.fan_image([rec], ranges, bearings, interp="linear")
    power = np.abs(img.image) ** 2
    R, B = np.meshgrid(ranges, bearings, indexing="ij")
    x, y = R * np.cos(B), R * np.sin(B)
    shadow = (x > 6.2) & (x < 8.5) & (np.abs(y) < 0.15)
    beside = (x > 6.2) & (x < 8.5) & (np.abs(y) > 0.9) & (np.abs(y) < 1.6)
    sphere = (x > 4.6) & (x < 5.0) & (np.abs(y) < 0.2)
    contrast_db = 10 * np.log10(power[beside].mean() / power[shadow].mean())
    assert contrast_db > 10.0, contrast_db
    assert power[sphere].mean() > power[beside].mean()            # the object itself is a highlight
    # Without the occluder the same patch of floor is as bright as its surroundings.
    scene.occluders.clear()
    rec2 = simulate_ping(sonar, pose, scene, environment={"sound_speed": C}, absorption_model="none", adc=False)
    power2 = np.abs(recon.fan_image([rec2], ranges, bearings, interp="linear").image) ** 2
    assert abs(10 * np.log10(power2[beside].mean() / power2[shadow].mean())) < 4.0


# --- built-in scenarios that use these objects -----------------------------------------------

def test_pool_wire_scenario_resolves_the_three_wires():
    from gpusonar.sim import run_scenario
    result = run_scenario("pool_wires_l12_5d")
    assert result.scene.labels == ["wire_0p5m", "wire_1m", "wire_2m"]
    records = result.records
    centre = np.vstack([r.rx_positions() for r in records]).mean(axis=0)
    for wire_x, wire_y in ((0.5, 0.0), (1.0, 0.05), (2.0, -0.10)):
        target = np.array([wire_x, wire_y, 0.5])
        rng_m = np.linalg.norm(target - centre)
        bearing = np.arctan2(wire_y, wire_x)
        ranges = rng_m + np.arange(-0.004, 0.0041, 0.0002)
        img = recon.fan_image(records, ranges, [bearing], interp="linear")
        profile = np.abs(img.image[:, 0])
        assert abs(ranges[np.argmax(profile)] - rng_m) < 0.0006          # at the wire, within 0.6 mm
        assert profile.max() > 5.0 * np.median(profile)


def test_scenarios_with_motion_raster_and_sediment_run():
    from gpusonar.config import load_scenario_config
    from gpusonar.sim import run_scenario
    # Rail with position errors: truth differs from what navigation reports.
    cfg = load_scenario_config("rail_position_errors", {"pings": {"count": 12}})
    result = run_scenario(cfg)
    reported = np.array([np.array(r.geometry.rx_pose.position) for r in result.records])
    assert np.allclose(reported[:, 1:], [0.0, -0.684])                   # the commanded line
    assert np.allclose(np.diff(reported[:, 0]), 0.002)
    assert all(np.allclose(r.geometry.rx_pose.velocity, 0.0) for r in result.records)   # stop and go
    # Wreck with large motion: a short stretch, to check it runs and the wreck is in the scene.
    cfg = load_scenario_config("wreck_sas", {"pings": {"count": 2}})
    cfg["scene"][0]["density"] = 2.0
    cfg["scene"][1]["density"] = 4.0
    result = run_scenario(cfg)
    assert len(result.records) == 2 and result.scene.occluders[0][0] == "box"
    assert np.abs(result.records[0].samples.array).max() > 0
    # Buried cylinder: the raster and the sediment tier.
    cfg = load_scenario_config("buried_cylinder")
    cfg["platform"]["raster"] = {"x": [-0.15, 0.151, 0.15], "y": [-0.15, 0.151, 0.15]}
    result = run_scenario(cfg)
    assert len(result.records) == 9
    assert np.allclose(np.array(result.records[4].geometry.rx_pose.position), [0.0, 0.0, -1.5], atol=1e-9)


def test_seafloor_objects_scenario_has_shadows():
    from gpusonar.sim import run_scenario
    result = run_scenario("seafloor_objects_fls")
    scene = result.scene
    assert {"seafloor", "sphere", "box", "rocks"} <= set(scene.labels)
    assert len(scene.occluders) == 8                                     # sphere, box, six rocks
    hidden = ~scene.visible(np.array([0.0, 0.0, -1.0]))
    floor = np.isnan(scene.normals).any(axis=1)
    assert 0.02 < hidden[floor].mean() < 0.4                             # part of the floor is in shadow


@pytest.mark.skipif(not __import__("gpusonar").cuda_available(),
                    reason="four minutes of synthesis on a CPU; runs where there is a CUDA device")
@pytest.mark.skipif(not __import__("gpusonar").core.HAVE_MOTION, reason="built without Ceres")
def test_wreck_scenario_through_motion_estimation_and_imaging():
    """The whole SAS chain on an extended scene: simulate the wreck run with motion that
    navigation does not know, estimate the motion from the data, image, and compare with
    the image made with the true track."""
    import gpusonar as gs
    from gpusonar import motion
    from gpusonar.sim import run_scenario
    result = run_scenario("wreck_sas", backend=gs.core.Backend.Cuda)
    records, truth = result.records, result.true_positions
    recon.prepare(records)
    estimate = motion.estimate_motion(records)
    wavelength = 1500.0 / 200e3
    assert np.abs(truth[:, 1]).max() > wavelength                    # more than a wavelength of sway
    assert np.sqrt(np.mean((estimate.y - truth[:, 1]) ** 2)) < wavelength / 5
    assert np.sqrt(np.mean((estimate.z - truth[:, 2]) ** 2)) < wavelength / 4

    x, y = np.arange(6.0, 13.0, 0.025), np.arange(39.0, 46.5, 0.025)
    X, Y = np.meshgrid(x, y, indexing="ij")
    points = np.stack([X, Y, np.zeros_like(X)], axis=-1)

    def image(recs):
        return np.abs(recon.sas_image(recs, points, interp="linear", backend=gs.core.Backend.Cuda)) ** 2

    def sharpness(im):
        return float((im**2).sum() / im.sum() ** 2 * im.size)

    with_true = [r.copy() for r in records]
    velocity = np.gradient(truth, 0.2, axis=0)
    for r, p, v in zip(with_true, truth, velocity):
        geo = r.geometry
        pose = geo.rx_pose
        pose.position, pose.velocity = tuple(p), tuple(v)
        geo.rx_pose = geo.tx_pose = pose
        r.geometry = geo
    with_estimate = [r.copy() for r in records]
    motion.apply_motion(with_estimate, estimate)
    nav, est, true = image(records), image(with_estimate), image(with_true)
    assert sharpness(est) > 0.8 * sharpness(true)
    assert sharpness(est) > 2.0 * sharpness(nav)
    shadow = (X > 8.4) & (X < 10.6) & (Y > 42.3) & (Y < 44.0)        # behind the 1 m high box
    floor = (X > 8.4) & (X < 10.6) & (Y > 39.3) & (Y < 40.6)
    face = (X > 8.2) & (X < 10.8) & (Y > 40.7) & (Y < 41.3)
    assert 10 * np.log10(est[floor].mean() / est[shadow].mean()) > 10.0
    assert 10 * np.log10(est[face].mean() / est[floor].mean()) > 12.0


# --- cylinders, volume clutter, per-object bookkeeping ---------------------------------------

def test_cylinder_points_normals_and_shadow():
    item = {"type": "cylinder", "start": [5.0, -1.0, 0.0], "end": [5.0, 1.0, 0.0], "radius": 0.25, "density": 2000.0}
    scene = build_scene([item], rng())
    axis_dist = np.hypot(scene.positions[:, 0] - 5.0, scene.positions[:, 2])
    side = np.abs(scene.normals[:, 1]) < 0.5                                # caps have normals along the axis
    assert 0.8 < side.mean() < 0.95
    assert np.allclose(axis_dist[side], 0.25)                               # the side wall
    assert np.all(axis_dist[~side] <= 0.25 + 1e-9)                          # the end caps
    assert np.allclose(np.abs(scene.positions[~side][:, 1]), 1.0)
    radial = (scene.positions[side] - np.array([5.0, 0.0, 0.0])) * [1, 0, 1] / 0.25
    assert np.allclose(scene.normals[side], radial, atol=1e-9)
    # Shadow of the capsule: behind the cylinder is hidden, beside and beyond its ends is not.
    pts = np.array([[8.0, 0.0, 0.0], [8.0, 1.0, 0.0], [8.0, 0.0, 0.35], [8.0, 0.0, 0.6], [8.0, 2.4, 0.0],
                    [4.0, 0.0, 0.0], [4.75, 0.0, 0.0], [5.25, 0.0, 0.0]])
    assert scene.visible(np.zeros(3), pts).tolist() == [False, False, False, True, True, True, True, False]
    # About half the side wall faces a distant sonar.
    q, p, amp, _, _ = two_way_paths(FreeFieldProvider(C), np.array([-50.0, 0, 0]), np.array([-50.0, 0, 0]),
                                    scene.positions, scene.amplitudes, lambert=scene.lambert, normals=scene.normals)
    assert 0.40 < len(amp) / len(scene) < 0.50


def test_volume_clutter_and_label_masks():
    scene = build_scene([
        {"type": "volume", "label": "clutter", "min": [-1, -1, 0.1], "max": [1, 1, 0.5], "count": 500, "rms_amplitude": 0.01},
        {"type": "box", "label": "crate", "min": [2, 2, 0], "max": [3, 3, 1], "density": 100.0},
        {"type": "points", "label": "pinger", "positions": [[0, 0, 0]]},
    ], rng())
    masks = scene.label_masks()
    assert set(masks) == {"clutter", "crate", "pinger"}
    assert masks["clutter"].sum() == 500 and masks["pinger"].sum() == 1
    assert masks["crate"].sum() == len(scene) - 501                         # six faces under one label
    assert sum(m.sum() for m in masks.values()) == len(scene)
    clutter = scene.positions[masks["clutter"]]
    assert clutter[:, 2].min() >= 0.1 and clutter[:, 2].max() <= 0.5
    assert np.sqrt(np.mean(np.abs(scene.amplitudes[masks["clutter"]]) ** 2)) == pytest.approx(0.01, rel=0.1)
    assert not scene.lambert[masks["clutter"]].any()


# --- boats, waves and seabed clutter ---------------------------------------------------

def test_sea_surface_facets_follow_the_waves_and_face_down_into_the_water():
    item = {"type": "sea_surface", "label": "waves", "x_range": [0.0, 80.0], "y_range": [-20.0, 20.0], "density": 2.0,
            "mu_db": -22.0, "waves": [{"amplitude": 0.6, "wavelength": 40.0, "direction_deg": 0.0},
                                      {"amplitude": 0.15, "wavelength": 6.0, "direction_deg": 50.0}]}
    scene = build_scene([item], rng())
    z = scene.positions[:, 2]
    assert 0.5 < z.max() <= 0.75 and -0.75 <= z.min() < -0.5       # the two wave trains add
    assert np.all(scene.normals[:, 2] > 0.5)                       # z is down: normals point into the water
    assert np.allclose(np.linalg.norm(scene.normals, axis=1), 1.0)
    # A sonar below and to the -x side gets more from faces tilted towards it than from faces tilted away.
    tilt = scene.normals[:, 0]
    assert tilt.std() > 0.03 and abs(tilt.mean()) < 0.02
    flat = build_scene([{**item, "waves": []}], rng())
    assert np.allclose(flat.positions[:, 2], 0.0) and np.allclose(flat.normals, [0.0, 0.0, 1.0])


def test_boat_has_hull_keel_propeller_and_a_wake_that_stays_behind():
    item = {"type": "boat", "label": "launch", "origin": [100.0, 0.0, 0.0], "length": 10.0, "beam": 3.0, "draft": 0.6,
            "keel_depth": 1.2, "heading_deg": 0.0, "velocity": [4.0, 0.0, 0.0], "count": 400,
            "wake": {"length": 50.0, "count": 300}}
    scene = build_scene([item], rng())
    masks = scene.label_masks()
    assert set(masks) == {"launch", "launch_wake"}
    boat, wake = scene.positions[masks["launch"]], scene.positions[masks["launch_wake"]]
    assert boat[:, 0].min() > 94.0 and boat[:, 0].max() < 106.0
    assert boat[:, 2].max() == pytest.approx(1.8, abs=0.05)        # keel foot: draft plus keel depth
    assert boat[:, 2].min() >= -0.05                               # nothing above the waterline is modelled
    assert wake[:, 0].max() <= 95.0 and wake[:, 0].min() >= 45.0   # astern only
    assert np.all((wake[:, 2] > 0.0) & (wake[:, 2] <= 2.0))        # bubbles in the top two metres
    assert np.median(wake[:, 0]) > 70.0                            # densest near the stern
    moved = scene.at_time(2.0)
    assert np.allclose(moved[masks["launch"]] - boat, [8.0, 0.0, 0.0])
    assert np.allclose(moved[masks["launch_wake"]], wake)          # the wake is water, not boat
    bare = build_scene([{k: v for k, v in item.items() if k not in ("wake", "keel_depth")}], rng())
    assert set(bare.label_masks()) == {"launch"} and bare.positions[:, 2].max() < 1.1   # hull and propeller only


def test_debris_is_boxes_on_the_bed_that_cast_shadows():
    scene = build_scene([{"type": "debris", "label": "junk", "x_range": [10.0, 30.0], "y_range": [-5.0, 5.0], "depth": 40.0,
                          "count": 7, "size_range": [0.5, 1.5], "density": 20.0, "mu_db": -10.0}], rng())
    assert len(scene.occluders) == 7 and all(o[0] == "box" for o in scene.occluders)
    p = scene.positions
    assert p[:, 2].max() <= 40.0 + 1e-9 and p[:, 2].min() >= 40.0 - 0.9 - 1e-9      # at most 0.6 x 1.5 m tall
    assert p[:, 0].min() > 9.0 and p[:, 0].max() < 31.0
