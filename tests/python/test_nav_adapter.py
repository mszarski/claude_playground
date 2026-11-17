"""Navigation from an inertial unit with a Doppler log: the adapter that turns a vendor's stream
into our datagrams (frames, units, signs, angle order, time base), what the datagrams carry
into a record, water depth from altitude + depth and what happens when bottom lock is lost,
the two navigation receivers (Python and C++) against each other, and a chain run fed through
the adapter against the same run fed directly.
"""

import json
import pathlib

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import live, navadapter
from gpusonar.live import LiveChain, NavListener
from gpusonar.navadapter import Mapping
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel

core = gs.core
ROOT = pathlib.Path(__file__).resolve().parents[2]
DEG = np.pi / 180.0
BASEBAND = {"kind": "baseband", "sample_rate": 8.0e3}


# --- the adapter ----------------------------------------------------------------------------------

def test_our_convention_is_yaw_pitch_roll_about_north_east_down():
    """The adapter's matrix for (roll, pitch, yaw) is the record's own (core.rotation_ned)."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        roll, pitch, yaw = rng.uniform(-1.2, 1.2, 3)
        m = navadapter.rotation_from_euler((roll, pitch, yaw), "zyx")
        assert np.allclose(m, np.asarray(core.rotation_ned(roll, pitch, yaw)), atol=1e-12)
        assert np.allclose(navadapter.euler_zyx(m), (roll, pitch, yaw), atol=1e-12)


def test_frames_signs_units_and_angle_order_come_out_as_ours():
    # A vehicle heading east, nose up 10 degrees, rolled 5 degrees to starboard, 30 m down, 100 m east of the origin.
    want = {"roll": 5 * DEG, "pitch": 10 * DEG, "yaw": 90 * DEG, "x": 0.0, "y": 100.0, "z": 30.0}

    def check(mapping, rec):
        got = Mapping(mapping).convert(dict(rec, t=1.0))
        for k, v in want.items():
            assert got[k] == pytest.approx(v, abs=1e-9), (k, mapping)

    # Ours, in degrees and centimetres.
    check({"time": {"field": "t"}, "attitude": {"fields": ["r", "p", "h"], "unit": "deg"},
           "position": {"fields": ["n", "e", "d"], "unit": "cm"}},
          {"r": 5, "p": 10, "h": 90, "n": 0, "e": 10000, "d": 3000})
    # Pitch positive nose down; a pressure depth beside an east-north-up position.
    check({"time": {"field": "t"}, "attitude": {"fields": ["r", "p", "h"], "unit": "deg", "signs": [1, -1, 1]},
           "position": {"fields": ["e", "n", "u"], "frame": "enu", "depth": {"field": "dep"}}},
          {"r": 5, "p": -10, "h": 90, "e": 100, "n": 0, "u": 999.0, "dep": 30})
    # East-north-up world with forward-left-up body axes (yaw counter-clockwise from east; a positive
    # pitch about the left-pointing axis is nose down; roll about the forward axis is the same roll).
    check({"time": {"field": "t"}, "attitude": {"fields": ["r", "p", "y"], "unit": "deg", "world": "enu", "body": "flu"},
           "position": {"fields": ["e", "n", "u"], "frame": "enu"}},
          {"r": 5, "p": -10, "y": 0, "e": 100, "n": 0, "u": -30})
    # The same attitude as a quaternion (scalar first), body forward-right-down to north-east-down.
    m = np.asarray(core.rotation_ned(want["roll"], want["pitch"], want["yaw"]))
    w = 0.5 * np.sqrt(1.0 + np.trace(m))
    q = [w, (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w)]
    check({"time": {"field": "t"}, "attitude": {"kind": "quaternion", "fields": ["qw", "qx", "qy", "qz"]},
           "position": {"fields": ["n", "e", "d"]}},
          dict(zip(("qw", "qx", "qy", "qz"), q), n=0, e=100, d=30))
    # Angles given in the other order of rotation (roll first, then pitch, then yaw about the turned axes) are another attitude:
    # the adapter must not read them as ours.
    got = Mapping({"time": {"field": "t"}, "attitude": {"fields": ["r", "p", "h"], "unit": "deg", "order": "xyz"}}).convert(
        {"t": 1.0, "r": 5, "p": 10, "h": 90})
    assert np.allclose(np.asarray(core.rotation_ned(got["roll"], got["pitch"], got["yaw"])),
                       navadapter.rotation_from_euler((5 * DEG, 10 * DEG, 90 * DEG), "xyz"), atol=1e-12)
    assert abs(got["roll"] - 5 * DEG) > 1 * DEG


def test_velocity_over_ground_from_the_vehicles_axes_and_from_east_north_up():
    att = {"fields": ["r", "p", "h"], "unit": "deg"}
    rec = {"t": 0.0, "r": 0, "p": 0, "h": 90, "a": 1500, "b": 200, "c": 50}
    got = Mapping({"time": {"field": "t"}, "attitude": att, "velocity": {"fields": ["a", "b", "c"], "frame": "body_frd", "unit": "mm/s"}}).convert(rec)
    assert (got["vx"], got["vy"], got["vz"]) == pytest.approx((-0.2, 1.5, 0.05), abs=1e-12)       # heading east: forward is east, starboard is south
    got = Mapping({"time": {"field": "t"}, "attitude": att, "velocity": {"fields": ["a", "b", "c"], "frame": "body_flu", "unit": "mm/s"}}).convert(rec)
    assert (got["vx"], got["vy"], got["vz"]) == pytest.approx((0.2, 1.5, -0.05), abs=1e-12)
    got = Mapping({"time": {"field": "t"}, "velocity": {"fields": ["a", "b", "c"], "frame": "enu", "unit": "mm/s"}}).convert(rec)
    assert (got["vx"], got["vy"], got["vz"]) == pytest.approx((0.2, 1.5, -0.05), abs=1e-12)
    # A velocity in the vehicle's axes cannot be turned without an attitude: it is left out, not guessed.
    got = Mapping({"time": {"field": "t"}, "velocity": {"fields": ["a", "b", "c"], "frame": "body_frd"}}).convert(rec)
    assert "vx" not in got
    # With pitch and roll the Doppler log's forward velocity has a vertical part.
    got = Mapping({"time": {"field": "t"}, "attitude": att, "velocity": {"fields": ["a", "b", "c"], "frame": "body_frd"}}).convert(
        {"t": 0.0, "r": 0, "p": 10, "h": 0, "a": 2.0, "b": 0.0, "c": 0.0})
    assert (got["vx"], got["vz"]) == pytest.approx((2.0 * np.cos(10 * DEG), -2.0 * np.sin(10 * DEG)), abs=1e-12)


def test_time_base_offset_rate_and_optional_fields():
    m = Mapping({"time": {"field": "t", "unit": "ms", "offset_s": 1.7e9 - 0.012}, "sound_speed": {"field": "sv"},
                 "altitude": {"field": "alt", "unit": "cm", "lock": {"field": "st", "equals": 3}, "invalid": [0, -100]},
                 "sigma": {"roll": {"field": "rs", "unit": "deg"}, "altitude": {"field": "as"}}, "output": {"rate_hz": 10.0}})
    a = m.convert({"t": 5000, "sv": 1487.5, "alt": 8010, "st": 3, "rs": 0.02, "as": 0.2})
    assert a["time_ns"] == 5_000_000_000 + 1_700_000_000_000_000_000 - 12_000_000
    assert a["sound_speed"] == 1487.5 and a["altitude"] == pytest.approx(80.1) and a["bottom_lock"] == 1
    assert a["roll_sigma"] == pytest.approx(0.02 * DEG) and a["altitude_sigma"] == 0.2
    assert m.convert({"t": 5050, "alt": 8000, "st": 3}) is None                 # faster than the output rate: dropped
    b = m.convert({"t": 5100, "alt": 8000, "st": 2})                            # the lock field says no
    assert b["bottom_lock"] == 0 and "altitude" not in b and "sound_speed" not in b
    assert m.convert({"t": 5200, "alt": -100, "st": 3})["bottom_lock"] == 0     # the unit's own "no value"
    assert m.convert({"alt": 1}) is None                                        # no time: no datagram
    with pytest.raises(navadapter.MappingError):
        Mapping({"position": {"fields": ["a", "b", "c"]}})                      # no time field
    with pytest.raises(navadapter.MappingError):
        Mapping({"time": {"field": "t", "unit": "fortnights"}}).convert({"t": 1})


def test_csv_and_json_lines_and_the_shipped_example(tmp_path):
    mapping = Mapping.load(ROOT / "configs" / "nav" / "generic_csv.yaml")
    lines = (ROOT / "configs" / "nav" / "generic_csv_example.csv").read_text().splitlines()
    fixes = list(mapping.convert_lines(lines))
    assert len(fixes) == 3
    a, c = fixes[0], fixes[2]
    assert a["time_ns"] == 1_000_000_000 and a["z"] == 20.0 and (a["x"], a["y"]) == (0.0, 0.0)
    assert a["roll"] == pytest.approx(1.5 * DEG) and a["pitch"] == pytest.approx(2.0 * DEG) and a["yaw"] == pytest.approx(90 * DEG)
    assert a["vy"] == pytest.approx(1.5, abs=0.01) and a["altitude"] == 80.1 and a["bottom_lock"] == 1
    assert c["bottom_lock"] == 0 and "altitude" not in c and c["y"] == pytest.approx(0.3)
    # The same content as JSON lines, and as CSV without a header line.
    names = lines[0].split(",")
    as_json = [json.dumps(dict(zip(names, map(float, row.split(","))))) for row in lines[1:]]
    cfg = dict(mapping.cfg, format="json")
    cfg.pop("csv")
    assert list(Mapping(cfg).convert_lines(as_json)) == fixes
    assert list(Mapping(dict(mapping.cfg, csv={"header": False, "columns": names})).convert_lines(lines[1:])) == fixes
    # Replay from a file through the command-line tool gives the same datagrams.
    import subprocess
    import sys
    out = tmp_path / "nav.jsonl"
    subprocess.run([sys.executable, str(ROOT / "tools" / "nav_adapter.py"), str(ROOT / "configs" / "nav" / "generic_csv.yaml"),
                    "--replay", str(ROOT / "configs" / "nav" / "generic_csv_example.csv"), "--out-file", str(out)], check=True)
    assert [json.loads(row) for row in out.read_text().splitlines()] == fixes
    # Paced replay waits by the stream's own time stamps.
    waits = []
    n = navadapter.replay(Mapping(mapping.cfg), lines, lambda fix: None, speed=2.0, sleep=waits.append)
    assert n == 3 and len(waits) == 2 and waits[-1] == pytest.approx(0.1, abs=0.02)


def test_the_unit_template_is_marked_unconfirmed_and_claims_nothing():
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "nav" / "sprintnav_template.yaml").read_text())
    assert set(cfg["unconfirmed"]) >= {"format", "time", "position", "attitude", "velocity", "altitude"}
    assert "FILL_IN" in json.dumps(cfg)


# --- water depth from altitude + depth ------------------------------------------------------------

def test_water_depth_from_altitude_and_depth_and_the_loss_of_bottom_lock():
    w = core.WaterDepthTracker({"from_altitude": True, "dvl_offset_m": 0.4, "altitude_sigma_m": 0.3, "depth_sigma_m": 0.4,
                                "hold_s": 30.0, "hold_sigma_m_per_s": 0.1, "hold_sigma_m_per_m": 0.05})
    s = int(1e9)
    depth, sigma, held, _ = w.update(0, 0.0, 0.0, 20.0, 79.6)
    assert depth == pytest.approx(100.0) and sigma == pytest.approx(0.5) and not held          # hypot(0.3, 0.4)
    depth, sigma, held, _ = w.update(1 * s, 2.0, 0.0, 20.0, 80.6, True, 1.2, 0.5)             # the fix's own sigmas
    assert depth == pytest.approx(101.0) and sigma == pytest.approx(1.3) and not held
    # Lock lost 10 s and 20 m later: the last value, with sigma grown by time and by distance.
    depth, sigma, held, held_s = w.update(11 * s, 22.0, 0.0, 25.0, float("nan"))
    assert held and held_s == pytest.approx(10.0) and depth == pytest.approx(101.0)
    assert sigma == pytest.approx(np.sqrt(1.3**2 + (0.1 * 10.0) ** 2 + (0.05 * 20.0) ** 2))
    later = w.update(21 * s, 42.0, 0.0, 25.0, 55.0, False)                                      # an altitude the unit does not vouch for
    assert later[2] and later[1] > sigma
    assert np.isnan(w.update(40 * s, 80.0, 0.0, 25.0, 0.0)[0])                                  # held too long: unknown
    back = w.update(41 * s, 82.0, 0.0, 25.0, 70.0)
    assert back[0] == pytest.approx(95.4) and not back[2]
    # An implausible altitude is no lock; and nothing comes out when the source is not the altitude.
    limited = core.WaterDepthTracker({"from_altitude": True, "max_altitude_m": 200.0})
    assert np.isnan(limited.update(0, 0, 0, 10.0, 500.0)[0])
    assert np.isnan(core.WaterDepthTracker({}).update(0, 0, 0, 10.0, 50.0)[0])


# --- datagrams into a record: both receivers ------------------------------------------------------

def fixes_of_a_dive(n=400, rate=20.0, t0=int(1.7e18)):
    """A vehicle heading north-east at 1.5 m/s, diving 0.25 m/s, rolling; bottom lock lost for a while."""
    out = []
    for i in range(n):
        t = i / rate
        lock = not (6.0 <= t < 9.0)
        fix = {"time_ns": t0 + int(round(t * 1e9)), "x": 1.06 * t, "y": 1.06 * t, "z": 20.0 + 0.25 * t,
               "roll": 0.03 * np.sin(0.7 * t), "pitch": -0.02, "yaw": np.pi / 4, "vx": 1.06, "vy": 1.06, "vz": 0.25,
               "sound_speed": 1500.0 - 0.4 * t, "bottom_lock": int(lock), "roll_sigma": 0.0004, "depth_sigma": 0.05,
               "altitude_sigma": 0.2}
        if lock:
            fix["altitude"] = 100.0 - (20.0 + 0.25 * t)
        out.append(fix)
    return out, t0


NAV = {"water_depth": {"source": "altitude", "hold_s": 20.0, "hold_sigma_m_per_s": 0.05}, "profile_step_m": 0.5}


def blank_record(t0, t=7.5):
    sonar = SonarModel.load("popoto_long_range_3row", {"receive": dict(BASEBAND, window={"range": [900.0, 1100.0]})})
    rec = sonar.make_record(make_pose(), time_ns=t0 + int(t * 1e9))
    return rec


def test_python_and_cpp_receivers_put_the_same_navigation_into_a_record():
    fixes, t0 = fixes_of_a_dive()
    py_nav = NavListener(**NAV)
    cpp_nav = core.NavigationCpp({"water_depth": dict(NAV["water_depth"], from_altitude=True), "profile_step_m": 0.5})
    for fix in fixes:
        py_nav.push(fix)
        assert cpp_nav.push(json.dumps(fix))
    a, b = blank_record(t0), blank_record(t0)
    t = a.transmit.time_ns
    live.apply_pose(a, *py_nav.at(t))
    live.apply_environment(a, *py_nav.extras(t))
    live.apply_nav_facts(a, py_nav.facts(t))
    s = a.samples
    live.apply_history(a, py_nav.history(t, t + int((s.start_time + s.n_samples / s.sample_rate) * 1e9)))
    assert cpp_nav.apply(b)
    for name in ("position", "rpy", "velocity"):
        assert np.allclose(getattr(a.geometry.rx_pose, name), getattr(b.geometry.rx_pose, name), atol=1e-9)
    # The ping falls inside the stretch without bottom lock: the water depth is the held one, and says so.
    assert a.environment.sound_speed == pytest.approx(1500.0 - 0.4 * 7.5) and b.environment.sound_speed == a.environment.sound_speed
    assert a.environment.water_depth == pytest.approx(100.0) and b.environment.water_depth == pytest.approx(100.0)
    fa, fb = live.nav_facts(a), live.nav_facts(b)
    assert fa["sound_speed_measured"] == 1 and fb["sound_speed_measured"] == 1
    assert fa["water_depth_held_s"] == pytest.approx(1.55, abs=0.06) and fb["water_depth_held_s"] == pytest.approx(fa["water_depth_held_s"])
    assert fa["water_depth_sigma_m"] > np.hypot(0.2, 0.05) and fb["water_depth_sigma_m"] == pytest.approx(fa["water_depth_sigma_m"])
    assert fa["roll_sigma"] == fb["roll_sigma"] == 0.0004
    # Depth-binned sound-speed samples since the start, the same from both (a sample every half metre of depth).
    sa, sb = np.array(fa["sound_speed_samples"]), np.array(fb["sound_speed_samples"])
    assert sa.shape == sb.shape and sa.shape[0] >= 8 and np.allclose(sa, sb, atol=2e-3)
    assert np.all(np.diff(sa[:, 3]) >= 0.5 - 1e-9)
    assert py_nav.facts(t).get("sound_speed_samples") is None                    # handed out once
    # Attitude history, depth and Doppler-log samples: in the record's own fields (no schema change).
    for name in ("ins", "depth", "dvl"):
        assert len(getattr(a.navigation, name)) == len(getattr(b.navigation, name)) > 20
    assert np.allclose([q.rpy for q in a.navigation.ins], [q.rpy for q in b.navigation.ins], atol=1e-12)
    assert np.allclose([q.depth for q in a.navigation.depth], [q.depth for q in b.navigation.depth])
    alt_a, alt_b = (np.array([q.altitude for q in r.navigation.dvl]) for r in (a, b))
    assert np.array_equal(np.isnan(alt_a), np.isnan(alt_b)) and np.isnan(alt_a).any() and np.isfinite(alt_a).any()
    assert np.allclose(alt_a[np.isfinite(alt_a)], alt_b[np.isfinite(alt_b)])
    vb_a, vb_b = (np.array([q.velocity_body for q in r.navigation.dvl]) for r in (a, b))
    assert np.allclose(vb_a, vb_b, atol=1e-9)
    assert np.allclose(np.linalg.norm(vb_a, axis=1), np.linalg.norm([1.06, 1.06, 0.25])) and vb_a[0, 0] == pytest.approx(1.5, abs=0.02)
    # It survives the container file.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        w = gs.RecordWriter(f"{tmp}/a.h5")
        w.write(a)
        w.close()
        back = gs.RecordReader(f"{tmp}/a.h5").read(0)
    assert live.nav_facts(back) == fa and len(back.navigation.dvl) == len(a.navigation.dvl)


def test_fixes_are_kept_long_enough_for_a_ping_processed_after_its_window_closed():
    """A hundred fixes a second: the old 600-fix buffer held six seconds, less than one ping's window."""
    nav = NavListener()
    cpp = core.NavigationCpp({})
    t0 = int(1.7e18)
    for i in range(3000):                                                   # 30 s at 100 Hz
        fix = {"time_ns": t0 + i * 10_000_000, "x": 0.0, "y": 0.0, "z": 10.0, "roll": 0.001 * i, "pitch": 0.0, "yaw": 0.0}
        nav.push(fix)
        cpp.push(json.dumps(fix))
    rec = blank_record(t0, t=8.0)                                           # transmitted 22 s before the newest fix
    assert nav.at(rec.transmit.time_ns)[1][0] == pytest.approx(0.8, abs=1e-6)
    assert cpp.apply(rec) and rec.geometry.rx_pose.rpy[0] == pytest.approx(0.8, abs=1e-6)
    assert len(rec.navigation.ins) > 300                                    # 2 s either side of a 1.5 s window, at 100 Hz
    short = NavListener(keep_s=5.0)
    for i in range(3000):
        short.push({"time_ns": t0 + i * 10_000_000, "roll": 0.001 * i})
    assert len(short.history(0, 2**62)) == pytest.approx(500, abs=2)


# --- the chain fed through the adapter ------------------------------------------------------------

def vendor_line(fix, t0):
    """One of our fixes as a line of the generic example stream (east-north-up, degrees, mm/s in the vehicle's axes,
    ms on the unit's own clock, which started at t0)."""
    rot = np.asarray(core.rotation_ned(fix["roll"], fix["pitch"], fix["yaw"]))
    body = rot.T @ np.array([fix["vx"], fix["vy"], fix["vz"]])
    lock = fix.get("bottom_lock", 1)
    cols = [(fix["time_ns"] - t0) / 1e6, fix["y"], fix["x"], -fix["z"], fix["z"], fix["roll"] / DEG, -fix["pitch"] / DEG, fix["yaw"] / DEG,
            *(body * 1e3), fix["sound_speed"], fix.get("altitude", -1), lock, fix["roll_sigma"] / DEG, fix["depth_sigma"], fix["altitude_sigma"]]
    return ",".join(f"{v:.12f}" for v in cols)


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_chain_fed_through_the_adapter_equals_the_chain_fed_directly(engine):
    fixes, t0 = fixes_of_a_dive(n=240)
    header = (ROOT / "configs" / "nav" / "generic_csv_example.csv").read_text().splitlines()[0]
    cfg = dict(Mapping.load(ROOT / "configs" / "nav" / "generic_csv.yaml").cfg)
    cfg.pop("output")                                                       # every fix, as the direct run gets
    cfg["time"] = dict(cfg["time"], offset_s=t0 / 1e9)                      # the unit's clock to the Unix epoch
    adapted = list(Mapping(cfg).convert_lines([header] + [vendor_line(f, t0) for f in fixes]))
    assert len(adapted) == len(fixes)
    worst = max(abs(a[k] - f[k]) for a, f in zip(adapted, fixes) for k in ("x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz"))
    assert worst < 1e-9
    assert all(a["time_ns"] == f["time_ns"] for a, f in zip(adapted, fixes))

    sonar = SonarModel.load("popoto_long_range_3row", {"receive": dict(BASEBAND, window={"range": [900.0, 1100.0]}),
                                                       "transmit": {"waveform": {"duration": 1.0}}})
    chain = {"bearings_deg": [-50.0, 50.0, 0.5], "cfar": {"pfa": 1e-6, "guard_m": 20.0, "train_m": 100.0}, "min_snr_db": 15.0,
             "backend": "host", "engine": engine, "tracker": {"range_sigma": 5.0, "confirm_hits": 2},
             "elevation": {"sector_from_depth": {"bounces": 1}, "roll_sigma_deg": 0.0}}      # (the simulated ping holds its attitude)
    reports = {}
    for label, stream in (("direct", fixes), ("adapter", adapted)):
        nav = NavListener(**NAV)
        for fix in stream:
            nav.push(fix)
        recs = []
        for i, t in enumerate((4.0, 7.5)):                                  # the second ping while bottom lock is lost
            pos, rpy, vel = nav.at(t0 + int(t * 1e9))
            pose = make_pose(position=tuple(pos), rpy=tuple(rpy))
            target = np.asarray(pos) + np.array([-700.0, 700.0, 25.0])     # to starboard of a vehicle heading north-east
            recs.append(simulate_ping(sonar, pose, Scene().add_points([target], target_strength_db=10.0), environment={"sound_speed": 1500.0},
                                      sequence=i, time_ns=t0 + int(t * 1e9), noise_rms=0.0))
        with LiveChain(chain, nav=nav) as live_chain:
            reports[label] = live_chain.run(recs)
    for a, b in zip(reports["direct"], reports["adapter"]):
        assert len(a["detections"]) == len(b["detections"]) >= 1
        for da, db in zip(a["detections"], b["detections"]):
            assert set(da) == set(db)
            for k, v in da.items():
                if isinstance(v, float):
                    assert db[k] == pytest.approx(v, rel=1e-6, abs=1e-6), k
                else:
                    assert db[k] == v, k
        assert np.allclose(a["ownship_xyz"], b["ownship_xyz"], atol=1e-6)
    best = max(reports["adapter"][0]["detections"], key=lambda d: d["snr_db"])
    # What the datagrams carried reached the estimate: the unit's own roll sigma, the water depth from the Doppler log
    # (the depth of the target is its own depth; the bounded sector needs the water depth to be there at all).
    assert best["elevation_sigma_deg"] >= np.degrees(0.0004) and best["depth_m"] == pytest.approx(20.0 + 0.25 * 4.0 + 25.0, abs=1.5)
