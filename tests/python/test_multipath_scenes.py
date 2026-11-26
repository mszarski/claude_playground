"""sonar-0nsi: the three multipath scenes. Their `truth` entries are the image geometry of the scenario's own planes,
and a simulated image (CPU, host backend) puts each ghost there. The objects are replaced by point targets at their
centres, so that each peak is a point's; the full scenes are measured in docs/SCENARIOS.md.

- fls_120k_quay_multipath: a quay wall and a bed sloping 5 degrees across the look direction; the sphere's ghost via
  the bed lands at a different bearing than over a level bed (the out-of-plane turn of docs/tasks/PROPAGATION_3D.md).
- pool_fls_p4_2v_walls: ghost floats in the side walls of a 4 m lane.
- sas_auv_multipath and its free-field twin: copies of targets above the bed further out in ground range."""

import copy
import warnings

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import recon
from gpusonar.config import load_scenario_config
from gpusonar.sim.propagation import make_provider
from gpusonar.sim.simulator import run_scenario

HOST = gs.core.Backend.Host


def provider_of(cfg):
    return make_provider(cfg["fidelity"], cfg["environment"]["sound_speed"], cfg.get("propagation"))


def as_points(cfg, labels_centres):
    """The scenario with its scene replaced by 0 dB point targets, no noise."""
    cfg = copy.deepcopy(cfg)
    cfg["scene"] = [{"type": "points", "label": k, "positions": [list(v)], "target_strength_db": 0.0} for k, v in labels_centres.items()]
    cfg["receiver"] = {"adc": False}
    return cfg


def simulate(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                     # placeholder and unconfirmed-value notices
        return run_scenario(cfg, backend=HOST)


def peak(power, ranges, bearings, r, b, dr, db):
    m = (np.abs(ranges - r) <= dr)[:, None] & (np.abs(bearings - b) <= db)[None, :]
    i, j = np.unravel_index(np.argmax(np.where(m, power, -1.0)), power.shape)
    return ranges[i], bearings[j], power[i, j]


def fls_ghosts(cfg, centre):
    """{(via, path, on): (range, bearing deg)} from the provider: range half the two-way path, bearing the cone angle
    of the return leg's arrival (asin of its component along the array axis, y)."""
    sonar = np.asarray(cfg["platform"]["position"], dtype=float)
    rays = provider_of(cfg).eigenrays(sonar, np.asarray(centre, dtype=float)[None])
    ok = np.flatnonzero(rays.valid[0])
    lengths = rays.sound_speed * rays.delay[0, ok]
    cone = np.degrees(np.arcsin(rays.launch[0, ok, 1]))
    direct = int(np.argmin(lengths))
    return lengths, cone, direct, rays, ok


# --- FLS by a quay ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def quay():
    return load_scenario_config("fls_120k_quay_multipath")


def test_quay_truth_is_the_image_geometry(quay):
    tr = quay["truth"]
    lengths, cone, d, rays, ok = fls_ghosts(quay, tr["sphere"]["position"])
    assert lengths[d] == pytest.approx(tr["sphere"]["range_m"], abs=0.01) and cone[d] == pytest.approx(tr["sphere"]["bearing_deg"], abs=0.01)
    ns, nb = rays.n_surface[0, ok], rays.n_bottom[0, ok]
    one_wall = [k for k in range(len(ok)) if ns[k] == nb[k] == 0 and k != d]
    one_bed = [k for k in range(len(ok)) if nb[k] == 1 and ns[k] == 0 and lengths[k] < lengths[d] + 3.0]
    assert len(one_wall) == 1 and len(one_bed) == 1
    expect = {"wall": one_wall[0], "bed": one_bed[0]}
    for g in tr["ghosts"]:
        if g["via"] == "level_bed":
            level = copy.deepcopy(quay)
            level["propagation"]["bottom_slope"] = [0.0, 0.0]
            ll, lc, ld, lr, lok = fls_ghosts(level, tr["sphere"]["position"])
            k = [i for i in range(len(lok)) if lr.n_bottom[0, lok[i]] == 1 and lr.n_surface[0, lok[i]] == 0][0]
            assert (ll[k], lc[k]) == pytest.approx((g["range_m"], g["bearing_deg"]), abs=0.01)
            continue
        k = expect[g["via"]]
        rng = lengths[k] if g["path"] == "both" else 0.5 * (lengths[k] + lengths[d])
        assert rng == pytest.approx(g["range_m"], abs=0.01), g
        assert g["bearing_deg"] in (pytest.approx(cone[k], abs=0.01), pytest.approx(cone[d], abs=0.01)), g
    # The cross-slope moves the bed ghost by about 1.4 degrees (1.5 beams of 0.93 degrees) against a level bed.
    bed = [g for g in tr["ghosts"] if g["via"] == "bed" and g["path"] == "both"][0]
    lvl = [g for g in tr["ghosts"] if g["via"] == "level_bed"][0]
    assert lvl["bearing_deg"] - bed["bearing_deg"] > 1.4


def test_quay_ghosts_land_where_the_images_say(quay):
    """One ping, a 0 dB point at the sphere's centre: the direct echo, the ghosts via the wall (24 m across) and via
    the tilted bed are peaks of the fan image at the truth's range and bearing; over a level bed the bed ghost moves
    to the level bed's bearing; in free field there is none."""
    tr = quay["truth"]
    cfg = as_points(quay, {"sphere": tr["sphere"]["position"]})
    cfg["pings"]["count"] = 1
    ranges, bearings = np.arange(44.0, 57.0, 0.03), np.arange(-2.0, 40.0, 0.1)

    def fan(c):
        rec = simulate(c).records
        return np.abs(recon.fan_image(rec, ranges, np.radians(bearings), interp="sinc", window="hann", backend=HOST).image) ** 2

    tilted = fan(cfg)
    r0, b0, p0 = peak(tilted, ranges, bearings, tr["sphere"]["range_m"], tr["sphere"]["bearing_deg"], 0.5, 1.0)
    assert abs(r0 - tr["sphere"]["range_m"]) < 0.1 and abs(b0 - tr["sphere"]["bearing_deg"]) < 0.3
    for g in tr["ghosts"]:
        if g["via"] == "level_bed" or (g["path"] == "mixed" and abs(g["bearing_deg"] - tr["sphere"]["bearing_deg"]) < 0.5):
            continue                                       # the mixed pair on the sphere's own bearing: checked by range below
        r, b, p = peak(tilted, ranges, bearings, g["range_m"], g["bearing_deg"], 0.4, 0.8)
        assert abs(r - g["range_m"]) < 0.12 and abs(b - g["bearing_deg"]) < 0.35, (g, r, b)
        assert 10.0 * np.log10(p / p0) > -15.0, g
    level = copy.deepcopy(cfg)
    level["propagation"]["bottom_slope"] = [0.0, 0.0]
    flat = fan(level)
    lvl = [g for g in tr["ghosts"] if g["via"] == "level_bed"][0]
    bed = [g for g in tr["ghosts"] if g["via"] == "bed" and g["path"] == "both"][0]
    r, b, p = peak(flat, ranges, bearings, lvl["range_m"], lvl["bearing_deg"], 0.4, 0.8)
    assert abs(r - lvl["range_m"]) < 0.12 and abs(b - lvl["bearing_deg"]) < 0.35
    # over the level bed nothing is left where the tilted bed's ghost was (beyond its main lobe)
    assert flat[np.argmin(np.abs(ranges - bed["range_m"])), np.argmin(np.abs(bearings - bed["bearing_deg"]))] < 0.1 * p
    free = copy.deepcopy(cfg)
    free["fidelity"] = "free_field"
    ff = fan(free)
    for g in tr["ghosts"]:
        if g["via"] != "level_bed" and abs(g["bearing_deg"] - tr["sphere"]["bearing_deg"]) > 0.5:
            assert peak(ff, ranges, bearings, g["range_m"], g["bearing_deg"], 0.3, 0.5)[2] < 1e-4 * p0, g


# --- the pool lane ---------------------------------------------------------------------------------------------

def test_pool_lane_ghost_floats():
    cfg = load_scenario_config("pool_fls_p4_2v_walls")
    sonar = np.asarray(cfg["platform"]["position"], dtype=float)
    floats = {i["label"]: i["centre"] for i in cfg["scene"] if i["label"].startswith("float")}
    p = provider_of(cfg)
    for g in cfg["truth"]["ghosts"]:
        rays = p.eigenrays(sonar, np.asarray(floats[g["object"]], dtype=float)[None])
        lengths = rays.sound_speed * rays.delay[0]
        side = [k for k in np.flatnonzero(rays.valid[0]) if abs(np.degrees(np.arcsin(rays.launch[0, k, 1])) - g["bearing_deg"]) < 0.01]
        assert len(side) == 1, g
        rng = lengths[side[0]] if g["path"] == "both" else 0.5 * (lengths[side[0]] + lengths[0])
        assert rng == pytest.approx(g["range_m"], abs=0.01), g
    sim = as_points(cfg, floats)
    ranges, bearings = np.arange(7.2, 8.8, 0.0003), np.arange(-32.0, 32.0, 0.2)
    pw = np.abs(recon.fan_image(simulate(sim).records, ranges, np.radians(bearings), interp="sinc", window="hann",
                                backend=HOST).image) ** 2
    _, _, p0 = peak(pw, ranges, bearings, 7.52, 4.6, 0.05, 1.0)
    free = copy.deepcopy(sim)
    free["fidelity"] = "free_field"
    ff = np.abs(recon.fan_image(simulate(free).records, ranges, np.radians(bearings), interp="sinc", window="hann",
                                backend=HOST).image) ** 2
    for g in cfg["truth"]["ghosts"]:
        r, b, pk = peak(pw, ranges, bearings, g["range_m"], g["bearing_deg"], 0.05, 0.8)
        assert abs(r - g["range_m"]) < 0.01 and abs(b - g["bearing_deg"]) < 0.3, (g, r, b)
        assert 10.0 * np.log10(pk / p0) > -15.0, g
        assert peak(ff, ranges, bearings, g["range_m"], g["bearing_deg"], 0.05, 0.8)[2] < 1e-4 * p0, g


# --- SAS over a flat bed under a surface -----------------------------------------------------------------------

def sas_truth(cfg, centre):
    """Ground ranges on the bed plane of the direct echo and the bed and surface copies, from the provider's paths
    at the closest approach (the sonar abeam of the target: a SAS image puts a point at its range of closest approach)."""
    sonar = np.asarray(cfg["platform"]["position"], dtype=float) + [centre[0], 0.0, 0.0]
    zb = float(cfg["propagation"]["bottom_z"])
    rays = provider_of(cfg).eigenrays(sonar, np.asarray(centre, dtype=float)[None])
    length = {(int(rays.n_surface[0, k]), int(rays.n_bottom[0, k])): rays.sound_speed * rays.delay[0, k]
              for k in np.flatnonzero(rays.valid[0])}

    def ground(rho):
        return float(np.sqrt(rho**2 - (zb - sonar[2]) ** 2))

    d, b, s = length[(0, 0)], length[(0, 1)], length[(1, 0)]
    return {"direct_y_m": ground(d), "mixed_y_m": ground(0.5 * (d + b)), "both_y_m": ground(b), "surface_mixed_y_m": ground(0.5 * (d + s))}


def test_sas_twin_is_the_same_scene_in_free_field():
    a, b = load_scenario_config("sas_auv_multipath"), load_scenario_config("sas_auv_multipath_free")
    assert a["fidelity"] == "image_source" and b["fidelity"] == "free_field"
    for key in ("sonar", "sonar_overrides", "platform", "pings", "scene", "seed", "receiver"):
        assert a[key] == b[key], key


def test_sas_multipath_copies_land_where_the_images_say():
    cfg = load_scenario_config("sas_auv_multipath")
    points = {k: [i for i in cfg["scene"] if i["label"] == k][0]["positions"][0] for k in ("point_1m", "point_3m", "point_8m")}
    for k, centre in points.items():
        mine = sas_truth(cfg, centre)
        for key, value in mine.items():
            assert cfg["truth"][k][key] == pytest.approx(value, abs=0.01), (k, key)
        assert cfg["truth"][k]["x_m"] == centre[0]
    images = {}
    for name in ("sas_auv_multipath", "sas_auv_multipath_free"):
        sim = as_points(load_scenario_config(name), points)
        res = simulate(sim)
        recon.prepare(res.records, backend=HOST)
        images[name] = {}
        for k, t in cfg["truth"].items():
            x = np.arange(t["x_m"] - 0.4, t["x_m"] + 0.4, 0.02)
            y = np.arange(t["direct_y_m"] - 0.3, t["both_y_m"] + 0.3, 0.01)
            X, Y = np.meshgrid(x, y, indexing="ij")
            plane = np.stack([X, Y, np.full_like(X, float(cfg["propagation"]["bottom_z"]))], axis=-1)
            images[name][k] = (X, Y, np.abs(recon.sas_image(res.records, plane, interp="linear", backend=HOST)) ** 2)
    for k, t in cfg["truth"].items():
        X, Y, im = images["sas_auv_multipath"][k]
        _, _, free = images["sas_auv_multipath_free"][k]

        def at(img, y, dx=0.1, dy=0.04):
            m = (np.abs(X - t["x_m"]) <= dx) & (np.abs(Y - y) <= dy)
            i = np.argmax(np.where(m, img, -1.0))
            return X.flat[i], Y.flat[i], img.flat[i]

        x0, y0, p0 = at(im, t["direct_y_m"])
        assert abs(y0 - t["direct_y_m"]) < 0.03 and abs(x0 - t["x_m"]) < 0.05, (k, x0, y0)
        for key in ("mixed_y_m", "both_y_m"):
            if abs(t[key] - t["direct_y_m"]) < 0.3:
                continue                                       # within a few resolution cells of the direct echo
            xg, yg, pg = at(im, t[key])
            assert abs(yg - t[key]) < 0.03 and abs(xg - t["x_m"]) < 0.1, (k, key, xg, yg)
            assert 10.0 * np.log10(pg / p0) > -20.0, (k, key)
            assert at(free, t[key])[2] < 1e-3 * at(free, t["direct_y_m"])[2], (k, key)


# --- both engines (sonar-k9v7) ---------------------------------------------------------------------------------

@pytest.mark.parametrize("name, labels", [("fls_120k_quay_multipath", ("moored_sphere", "block")),
                                          ("pool_fls_p4_2v_walls", ("float_1", "float_3", "ladder_rail_a"))])
def test_a_ping_is_the_same_on_both_engines(name, labels):
    """The scenes with planes compute their eigenrays in C++ on the C++ engine since sonar-k9v7 (in NumPy for both
    before). One ping with the objects as 0 dB points at their centres agrees between the engines to the tolerance of
    the engine-parity test of an image_source scenario (`test_sim_cpp.py` REVERB_TOL, 5e-6 of the largest sample and
    of the rms). The full scenes are compared in the hand-back of sonar-k9v7 (one ping each)."""
    cfg = load_scenario_config(name)
    centres = {}
    for item in cfg["scene"]:
        if item.get("label") in labels:
            ends = ("min", "max") if "min" in item else ("start", "end")
            centres[item["label"]] = item["centre"] if "centre" in item else list(
                0.5 * (np.asarray(item[ends[0]], dtype=float) + np.asarray(item[ends[1]], dtype=float)))
    assert set(centres) == set(labels)
    sim = as_points(cfg, centres)
    sim["pings"]["count"] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = run_scenario(sim, backend=HOST, engine="cpp").records[0].samples.array.astype(np.complex128)
        b = run_scenario(sim, backend=HOST, engine="numpy").records[0].samples.array.astype(np.complex128)
    d = np.abs(a - b)
    assert np.abs(b).max() > 0.0
    assert d.max() / np.abs(b).max() < 5e-6 and np.sqrt(np.mean(d**2) / np.mean(np.abs(b) ** 2)) < 5e-6
