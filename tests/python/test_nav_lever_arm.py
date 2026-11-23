"""sonar-3rv: the surveyed offset between the navigation unit and the array (lever arm and mounting rotation).

A navigation fix is the pose of the navigation unit's reference point and axes; the record's pose is the array's
(the sonar file's body origin and body axes, in which the element lever arms are given). Both receivers (Python
`NavListener`, C++ `Navigation`) move every fix to the array as it arrives, with

    array position = unit position + R(unit rpy) lever_arm          (lever arm in the unit's body axes)
    R(array rpy)   = R(unit rpy) R(mount rpy)                       (R_mount: the array's axes into the unit's)

R the body-to-NED rotation of ASASIN Eq. 2 (Rz(yaw) Ry(pitch) Rx(roll)). Zero offsets leave every fix as it came.
"""

import json

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import live
from gpusonar.live import LiveChain, NavListener
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel

core = gs.core
DEG = np.pi / 180.0
T0 = int(1.7e18)
BASEBAND = {"kind": "baseband", "sample_rate": 8.0e3}


# Written out here, independent of core.rotation_ned and of the code under test.
def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def body_to_ned(roll, pitch, yaw):
    return rz(yaw) @ ry(pitch) @ rx(roll)


def by_hand(fix, lever, mount_deg):
    """The array's (position, rotation matrix) from a unit's fix."""
    r = body_to_ned(fix["roll"], fix["pitch"], fix["yaw"])
    p = np.array([fix["x"], fix["y"], fix["z"]]) + r @ np.asarray(lever, dtype=float)
    return p, r @ body_to_ned(*(np.asarray(mount_deg, dtype=float) * DEG))


def to_array_fix(fix, lever, mount_deg):
    """A fix of the unit as the array's fix (by hand), everything else as it was."""
    p, m = by_hand(fix, lever, mount_deg)
    roll, pitch, yaw = np.arctan2(m[2, 1], m[2, 2]), -np.arcsin(m[2, 0]), np.arctan2(m[1, 0], m[0, 0])
    return dict(fix, x=p[0], y=p[1], z=p[2], roll=roll, pitch=pitch, yaw=yaw)


def blank_record(t):
    sonar = SonarModel.load("popoto_long_range_3row", {"receive": dict(BASEBAND, window={"range": [900.0, 1100.0]})})
    return sonar.make_record(make_pose(), time_ns=T0 + int(t * 1e9))


def receivers(lever=None, mount=None, **extra):
    opts = dict(extra)
    if lever is not None:
        opts["lever_arm_m"] = list(lever)
    if mount is not None:
        opts["mount_rpy_deg"] = list(mount)
    return NavListener(**opts), core.NavigationCpp(opts)


def pose_of(nav_py, nav_cpp, t):
    """(Python (position, rpy, velocity), C++ record) at T0 + t s."""
    rec = blank_record(t)
    assert nav_cpp.apply(rec)
    return nav_py.at(rec.transmit.time_ns), rec


# --- the pose -----------------------------------------------------------------------------------------

def test_a_lever_arm_on_a_rolled_vehicle_heading_east_lands_where_worked_out_on_paper():
    """Heading east (yaw 90), rolled 90 degrees to starboard: the unit's forward axis points east and its down axis
    north (starboard down swings the keel to port, and port is north). The array 1 m ahead of and 1 m below the unit
    is therefore 1 m east and 1 m north of it, at the unit's depth."""
    fix = {"time_ns": T0, "x": 10.0, "y": 20.0, "z": 30.0, "roll": 90 * DEG, "pitch": 0.0, "yaw": 90 * DEG, "vx": 0.0, "vy": 1.5, "vz": 0.0}
    py, cpp = receivers(lever=[1.0, 0.0, 1.0])
    py.push(fix)
    assert cpp.push(json.dumps(fix))
    (pos, rpy, vel), rec = pose_of(py, cpp, 0.0)
    for got_pos, got_rpy, got_vel in ((pos, rpy, vel), (rec.geometry.rx_pose.position, rec.geometry.rx_pose.rpy, rec.geometry.rx_pose.velocity)):
        assert np.allclose(got_pos, [11.0, 21.0, 30.0], atol=1e-12)
        assert np.array_equal(got_rpy, [90 * DEG, 0.0, 90 * DEG])         # no mounting rotation: attitude untouched
        assert np.array_equal(got_vel, [0.0, 1.5, 0.0])                    # velocity is not moved (no rates in a fix)
    assert rec.geometry.tx_pose.position == rec.geometry.rx_pose.position  # the projector is on the array's body


@pytest.mark.parametrize("t", [0.0, 0.5, 1.0])
def test_lever_arm_and_mounting_at_nonzero_roll_pitch_yaw_give_the_hand_computed_pose_in_both_receivers(t):
    """Two fixes with roll, pitch and yaw all non-zero; at each fix's own time and halfway between them."""
    lever, mount = [1.25, -0.40, 0.85], [2.0, -1.5, 3.0]
    fixes = [{"time_ns": T0, "x": 100.0, "y": -50.0, "z": 30.0, "roll": 12 * DEG, "pitch": -7 * DEG, "yaw": 125 * DEG,
              "vx": 1.0, "vy": 0.5, "vz": 0.1},
             {"time_ns": T0 + 1_000_000_000, "x": 101.0, "y": -49.5, "z": 30.1, "roll": 8 * DEG, "pitch": -5 * DEG, "yaw": 127 * DEG,
              "vx": 1.0, "vy": 0.5, "vz": 0.1}]
    py, cpp = receivers(lever, mount)
    for f in fixes:
        py.push(f)
        assert cpp.push(json.dumps(f))
    (pos, rpy, vel), rec = pose_of(py, cpp, t)
    pose = rec.geometry.rx_pose
    # Each fix is moved as it arrives; between fixes the moved poses are interpolated (the same in both receivers).
    arrays = [to_array_fix(f, lever, mount) for f in fixes]
    w = t
    want_pos = (1 - w) * np.array([arrays[0][k] for k in "xyz"]) + w * np.array([arrays[1][k] for k in "xyz"])
    want_rpy = (1 - w) * np.array([arrays[0][k] for k in ("roll", "pitch", "yaw")]) + w * np.array([arrays[1][k] for k in ("roll", "pitch", "yaw")])
    assert np.allclose(pos, want_pos, atol=1e-9) and np.allclose(pose.position, want_pos, atol=1e-9)
    assert np.allclose(rpy, want_rpy, atol=1e-12) and np.allclose(pose.rpy, want_rpy, atol=1e-12)
    assert np.allclose(vel, [1.0, 0.5, 0.1]) and np.allclose(pose.velocity, [1.0, 0.5, 0.1])
    if t in (0.0, 1.0):
        p, m = by_hand(fixes[int(t)], lever, mount)
        assert np.allclose(pos, p, atol=1e-9)
        assert np.allclose(np.asarray(core.rotation_ned(*rpy)), m, atol=1e-12)          # the composed rotation itself
        assert np.allclose(np.asarray(core.rotation_ned(*pose.rpy)), m, atol=1e-12)
    # The engines agree to rounding.
    assert np.allclose(pos, pose.position, atol=1e-12, rtol=0) and np.allclose(rpy, pose.rpy, atol=1e-14, rtol=0)


def test_the_attitude_history_and_depth_samples_are_the_arrays_the_water_depth_and_sound_speed_samples_the_units():
    """The record's navigation.ins and .depth come from the moved fixes; the water depth from altitude (whose
    dvl_offset_m is measured from the unit) and the (depth, sound speed) samples come from the unit's own position."""
    lever, mount = [0.5, 0.2, 2.0], [0.0, 0.0, 4.0]
    offsets = {"lever_arm_m": lever, "mount_rpy_deg": mount}
    plain_py, moved_py = NavListener(water_depth={"source": "altitude"}), NavListener(water_depth={"source": "altitude"}, **offsets)
    plain_cpp, moved_cpp = (core.NavigationCpp({"water_depth": {"from_altitude": True}, **o}) for o in ({}, offsets))
    fixes = []
    for i in range(80):
        t = i / 10.0
        fixes.append({"time_ns": T0 + int(t * 1e9), "x": 1.0 * t, "y": 0.0, "z": 20.0 + 0.5 * t, "roll": 0.05 * np.sin(t),
                      "pitch": 0.03, "yaw": 0.2, "vx": 1.0, "vy": 0.0, "vz": 0.5, "sound_speed": 1500.0 - 0.2 * t,
                      "altitude": 80.0 - 0.5 * t})
    for f in fixes:
        for n in (plain_py, moved_py):
            n.push(f)
        for n in (plain_cpp, moved_cpp):
            assert n.push(json.dumps(f))
    t = 3.0
    recs = {}
    for label, py, cpp in (("plain", plain_py, plain_cpp), ("moved", moved_py, moved_cpp)):
        a, b = blank_record(t), blank_record(t)
        when = a.transmit.time_ns
        live.apply_pose(a, *py.at(when))
        live.apply_environment(a, *py.extras(when))
        live.apply_nav_facts(a, py.facts(when))
        s = a.samples
        live.apply_history(a, py.history(when, when + int((s.start_time + s.n_samples / s.sample_rate) * 1e9)))
        assert cpp.apply(b)
        recs[label] = (a, b)
    for a, b in (recs["plain"], recs["moved"]):                               # the engines agree on each
        assert np.allclose(a.geometry.rx_pose.position, b.geometry.rx_pose.position, atol=1e-9)
        assert np.allclose([q.rpy for q in a.navigation.ins], [q.rpy for q in b.navigation.ins], atol=1e-12)
        assert np.allclose([q.position for q in a.navigation.ins], [q.position for q in b.navigation.ins], atol=1e-9)
        assert np.allclose([q.depth for q in a.navigation.depth], [q.depth for q in b.navigation.depth], atol=1e-9)
        assert np.allclose([q.velocity_body for q in a.navigation.dvl], [q.velocity_body for q in b.navigation.dvl], atol=1e-9)
        assert np.allclose(live.nav_facts(a)["sound_speed_samples"], live.nav_facts(b)["sound_speed_samples"], atol=2e-3)
    (pa, _), (ma, mb) = recs["plain"], recs["moved"]
    # The history is the array's: each sample is its fix, moved.
    for q in ma.navigation.ins:
        f = min(fixes, key=lambda f: abs(f["time_ns"] - (ma.transmit.time_ns + q.t * 1e9)))
        p, m = by_hand(f, lever, mount)
        assert np.allclose(q.position, p, atol=1e-9) and np.allclose(np.asarray(core.rotation_ned(*q.rpy)), m, atol=1e-12)
    assert np.allclose([q.depth for q in ma.navigation.depth], [q.position[2] for q in ma.navigation.ins])
    assert ma.geometry.rx_pose.position[2] - pa.geometry.rx_pose.position[2] == pytest.approx(
        (body_to_ned(0.05 * np.sin(t), 0.03, 0.2) @ lever)[2], abs=1e-9)
    # The water depth, its sigma and the sound-speed samples are those without the offset.
    assert ma.environment.water_depth == pa.environment.water_depth == mb.environment.water_depth
    assert ma.environment.water_depth == pytest.approx(100.0)
    assert live.nav_facts(ma) == live.nav_facts(pa)


# --- zero offsets: nothing moves ----------------------------------------------------------------------

def test_zero_offsets_leave_every_fix_bit_for_bit_as_before():
    """Explicit zeros, or the keys left out: the same pose, history and facts, compared with ==, in both receivers."""
    fixes = []
    for i in range(60):
        t = i / 20.0
        fixes.append({"time_ns": T0 + int(t * 1e9), "x": 1.06 * t, "y": 1.06 * t, "z": 20.0 + 0.25 * t, "roll": 0.03 * np.sin(0.7 * t),
                      "pitch": -0.02, "yaw": np.pi / 4 + 3.5, "vx": 1.06, "vy": 1.06, "vz": 0.25, "sound_speed": 1500.0 - 0.4 * t,
                      "altitude": 70.0})
    kinds = [receivers(), receivers([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), receivers([0, 0, 0], None), receivers(None, [-0.0, 0.0, 0.0])]
    for py, cpp in kinds:
        for f in fixes:
            py.push(f)
            assert cpp.push(json.dumps(f))
    for t in (0.0, 1.0, 1.234, 2.95, 5.0):
        outs = []
        for py, cpp in kinds:
            (pos, rpy, vel), rec = pose_of(py, cpp, t)
            outs.append((tuple(pos), tuple(rpy), tuple(vel), rec.geometry.rx_pose.position, rec.geometry.rx_pose.rpy,
                         [q.rpy for q in rec.navigation.ins], [q.position for q in rec.navigation.ins], rec.metadata.extra_json))
        assert all(o == outs[0] for o in outs[1:]), t
    # Held from the first fix, the pose is that fix as sent (yaw beyond pi included: not re-wrapped).
    f = fixes[0]
    (pos, rpy, vel), rec = pose_of(*kinds[1], (f["time_ns"] - T0) / 1e9)
    assert tuple(rec.geometry.rx_pose.rpy) == (f["roll"], f["pitch"], f["yaw"]) == tuple(float(q) for q in rpy)
    assert tuple(rec.geometry.rx_pose.position) == (f["x"], f["y"], f["z"]) == tuple(float(q) for q in pos)


# --- the keys -----------------------------------------------------------------------------------------

def test_the_keys_reach_the_program_and_bad_values_are_refused():
    deploy = {"nav": {"lever_arm_m": [1.25, -0.4, 0.85], "mount_rpy_deg": [2.0, -1.5, 3.0], "port": 5702}}
    flat = live.program_options({"backend": "host"}, None, deploy)
    assert flat["nav_lever_arm_m"] == "1.25,-0.4,0.85" and flat["nav_mount_rpy_deg"] == "2,-1.5,3"
    got = json.loads(core.live_options_from_text("".join(f"{k}={v}\n" for k, v in flat.items())))
    assert got["nav_lever_arm_m"] == pytest.approx([1.25, -0.4, 0.85]) and got["nav_mount_rpy_deg"] == pytest.approx([2.0, -1.5, 3.0])
    defaults = json.loads(core.live_options_from_text(""))
    assert defaults["nav_lever_arm_m"] == [0, 0, 0] and defaults["nav_mount_rpy_deg"] == [0, 0, 0]
    assert "nav_lever_arm_m" not in live.program_options({"backend": "host"}, None, {"nav": {"port": 5702}})
    for pair in ("nav_lever_arm_m=1,2", "nav_lever_arm_m=1,2,x", "nav_mount_rpy_deg=1,2,3,4", "nav_mount_rpy_deg=nan,0,0"):
        with pytest.raises(ValueError, match=pair.split("=")[0]):
            core.live_options_from_text(pair)
    for bad in ([1.0, 2.0], [1.0, float("nan"), 0.0], "abc"):
        with pytest.raises(ValueError, match="lever_arm_m"):
            NavListener(lever_arm_m=bad)
        with pytest.raises(ValueError, match="mount_rpy_deg"):
            core.NavigationCpp({"mount_rpy_deg": bad}) if not isinstance(bad, str) else NavListener(mount_rpy_deg=bad)
    # From a deployment file's nav section (tools/sonar_live.py), static pose included: it is the unit's pose, moved.
    nav = NavListener.from_config({"static": {"x": 5.0, "yaw": 90 * DEG}, "lever_arm_m": [2.0, 0.0, 0.0]})
    pos, rpy, _ = nav.at(0)
    assert np.allclose(pos, [5.0, 2.0, 0.0], atol=1e-12) and np.array_equal(rpy, [0.0, 0.0, 90 * DEG])


# --- end to end: a target through the chain -----------------------------------------------------------

@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_a_target_seen_through_the_corrected_pose_is_where_the_array_saw_it(engine):
    """The vehicle's unit sends its own pose; the array sits 1.5 m ahead, 0.5 m to port and 2 m below it, turned
    (1, -2, 3) degrees. Pings simulated from the array's true pose, a target 25 m below and 1 km to starboard. The
    chain given the unit's fixes and the offset reports what it reports given the array's own fixes (no offset);
    without the offset the array's position is off by the lever arm and the target's depth by its vertical part."""
    lever, mount = [1.5, -0.5, 2.0], [1.0, -2.0, 3.0]
    unit = []
    for i in range(240):
        t = i / 20.0
        unit.append({"time_ns": T0 + int(round(t * 1e9)), "x": 1.06 * t, "y": 1.06 * t, "z": 20.0 + 0.25 * t,
                     "roll": 0.03 * np.sin(0.7 * t) + 4 * DEG, "pitch": -3 * DEG, "yaw": np.pi / 4, "vx": 1.06, "vy": 1.06, "vz": 0.25,
                     "sound_speed": 1500.0, "water_depth": 100.0, "roll_sigma": 0.0004})
    array = [to_array_fix(f, lever, mount) for f in unit]
    sonar = SonarModel.load("popoto_long_range_3row", {"receive": dict(BASEBAND, window={"range": [900.0, 1100.0]}),
                                                       "transmit": {"waveform": {"duration": 1.0}}})
    chain = {"bearings_deg": [-50.0, 50.0, 0.5], "cfar": {"pfa": 1e-6, "guard_m": 20.0, "train_m": 100.0}, "min_snr_db": 15.0,
             "backend": "host", "engine": engine, "tracker": {"range_sigma": 5.0, "confirm_hits": 2},
             "elevation": {"sector_from_depth": {"bounces": 1}, "roll_sigma_deg": 0.0}}
    truth = NavListener()
    for f in array:
        truth.push(f)
    recs, targets = [], []
    for i, t in enumerate((4.0, 7.5)):
        pos, rpy, vel = truth.at(T0 + int(t * 1e9))
        target = np.asarray(pos) + np.array([-700.0, 700.0, 25.0])          # to starboard of a vehicle heading north-east
        targets.append(target)
        recs.append(simulate_ping(sonar, make_pose(position=tuple(pos), rpy=tuple(rpy)),
                                  Scene().add_points([target], target_strength_db=10.0), environment={"sound_speed": 1500.0},
                                  sequence=i, time_ns=T0 + int(t * 1e9), noise_rms=0.0))
    runs = {}
    for label, stream, opts in (("truth", array, {}), ("corrected", unit, {"lever_arm_m": lever, "mount_rpy_deg": mount}),
                                ("uncorrected", unit, {})):
        nav = NavListener(**opts)
        for f in stream:
            nav.push(f)
        with LiveChain(chain, nav=nav) as c:
            runs[label] = c.run([r.copy() for r in recs])
    for a, b in zip(runs["truth"], runs["corrected"]):
        assert np.allclose(a["ownship_xyz"], b["ownship_xyz"], atol=1e-6)
        assert len(a["detections"]) == len(b["detections"]) >= 1
        for da, db in zip(a["detections"], b["detections"]):
            assert set(da) == set(db)
            for k, v in da.items():
                if isinstance(v, float):
                    assert db[k] == pytest.approx(v, rel=1e-6, abs=1e-6), k
                else:
                    assert db[k] == v, k
    for k, (rep, target) in enumerate(zip(runs["corrected"], targets)):
        best = max(rep["detections"], key=lambda d: d["snr_db"])
        assert best["depth_m"] == pytest.approx(target[2], abs=1.5)
        assert best["range_m"] == pytest.approx(np.linalg.norm(target - np.asarray(truth.at(rep["time_ns"])[0])), abs=3.0)
        # Without the offset the chain places the array at the unit: off by the lever arm in north-east-down.
        off = np.asarray(runs["uncorrected"][k]["ownship_xyz"]) - np.asarray(rep["ownship_xyz"])
        assert np.linalg.norm(off) > 2.0
        # and the 1 degree roll, -2 degree pitch misalignment puts the target about 19 m shallow at 1 km (measured: 28.5 m
        # against 48.0 m; the corrected run 47.6 m).
        wrong = max(runs["uncorrected"][k]["detections"], key=lambda d: d["snr_db"])
        assert abs(wrong["depth_m"] - target[2]) > 10.0
