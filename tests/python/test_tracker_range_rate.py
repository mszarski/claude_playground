"""D5: the tracker and the classifier use a detection's measured range rate (`chain.tracker.range_rate`, opt-in; both
engines: python/gpusonar/detect.py Tracker, classify.py, libs/live tracker.cpp, classify.cpp; docs/tasks/DOPPLER.md "D5").

What is held here (ours, every number derived in the test or stated):
- the measurement: rate relative to the platform, opening positive, h(x) = u . (v - v_o), u from the platform to the
  track (detect.rate_model); with the rate the velocity's covariance along the line of sight is the closed-form linear
  Kalman (information-form) result, and across it nothing changes;
- with noise, the velocity error shrinks as the covariance says;
- a still target with a zero rate stays static; a mover with the rate is classed a mover in fewer pings;
- a rate flagged not trusted (range_rate_sigma NaN: the sweep bank's on an extended target, D-232) changes nothing;
- a still target seen from a moving platform has a nonzero measured rate and stays static (the rate is taken relative to
  the platform; a naive form that reads it as over ground calls it a mover: shown);
- update_and_gate: a rate far from the track's predicted one argues against the pair;
- the two engines equal (tracks, covariances, classes, taps), and off is the tracker as before, bit for bit.
"""

import copy

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import classify, detect

core = gs.core
NAN = float("nan")


def det(x, y, snr=20.0, rate=NAN, sigma=NAN, platform=(0.0, 0.0, 0.0, 0.0), time=None):
    """A tracker detection at (across, forward) = (x, y) about the frame origin, with a range rate."""
    d = detect.Detection(float(np.hypot(x, y)), float(np.arctan2(x, y)), snr, 2, time=time)
    d.range_rate, d.range_rate_sigma = rate, sigma
    d.platform_x, d.platform_y, d.platform_vx, d.platform_vy = platform
    return d


def as_cpp(dets):
    return ([(d.range, d.bearing, d.snr_db, d.cells) for d in dets],
            [None if d.time is None else d.time for d in dets],
            [None if not np.isfinite(d.range_rate) else [d.range_rate, d.range_rate_sigma, d.platform_x, d.platform_y,
                                                         d.platform_vx, d.platform_vy] for d in dets])


def true_rate(p, v, platform):
    """Range rate relative to the platform, opening positive, of a target at p with velocity v."""
    o, vo = np.array(platform[:2]), np.array(platform[2:])
    u = (np.asarray(p) - o) / np.linalg.norm(np.asarray(p) - o)
    return float(u @ (np.asarray(v) - vo))


# --- the measurement and its covariance -------------------------------------------------------------------------------

def test_the_rate_model_is_the_line_of_sight_component_relative_to_the_platform_and_its_jacobian():
    rng = np.random.default_rng(3)
    for _ in range(20):
        x = np.concatenate([rng.uniform(-3000, 3000, 2), rng.uniform(-3, 3, 2)])
        d = det(1.0, 1.0, platform=tuple(np.concatenate([rng.uniform(-100, 100, 2), rng.uniform(-2, 2, 2)])))
        h, hh = detect.rate_model(x, d)
        assert h == pytest.approx(true_rate(x[:2], x[2:], (d.platform_x, d.platform_y, d.platform_vx, d.platform_vy)), rel=1e-12)
        num = np.array([(detect.rate_model(x + e, d)[0] - detect.rate_model(x - e, d)[0]) / 2e-4 for e in np.eye(4) * 1e-4])
        assert np.allclose(hh, num, rtol=1e-6, atol=1e-10)
    assert detect.rate_model(np.array([5.0, 5.0, 0.0, 0.0]), det(1.0, 1.0, platform=(5.0, 5.0, 0.0, 0.0))) is None


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_with_the_rate_the_velocity_covariance_along_the_line_of_sight_is_the_closed_form_kalman_result(engine):
    """A still target straight ahead (forward axis) at R, noise-free detections (so the state stays at the truth and the
    filter is linear: H_rate = [0, 0, 0, 1] exactly), no process noise: the filter is least squares with the birth's
    velocity prior. Per axis, with the state at the first ping (p0, v) and pings at t_k:
    J = sum_k [1, t_k]^T [1, t_k] / sigma^2 + diag(0, 1 / sigma0^2) (sigma: the axis's position noise: R sigma_b across,
    sigma_r forward), and the rate adds N / sigma_rate^2 to J_vv on the forward axis. The velocity variance is (J^-1)_vv
    (the constant-velocity transition does not change it). Across: unchanged by the rate."""
    R, sr, sb, s0, srate, T, N = 3000.0, 2.0, np.radians(1.0), 8.0, 0.06, 12.0, 6
    t = T * np.arange(N)

    def closed(sigma, rate_sigma=None):
        a = np.stack([np.ones(N), t], axis=1)
        j = a.T @ a / sigma**2 + np.diag([0.0, 1.0 / s0**2])
        if rate_sigma is not None:
            j[1, 1] += N / rate_sigma**2
        return float(np.linalg.inv(j)[1, 1])

    got = {}
    for mode in ("off", "update"):
        kw = dict(process_noise=0.0, range_sigma=sr, bearing_sigma=sb, initial_speed_sigma=s0, confirm_hits=1, max_misses=3,
                  range_rate=mode)
        trk = detect.Tracker(**kw) if engine == "python" else core.LiveTracker(**kw)
        for k in range(N):
            dets = [det(0.0, R, rate=0.0, sigma=srate)]
            if engine == "python":
                out = trk.update(t[k], dets)
                cov = out[0].cov if out else None
            else:
                a, _, r = as_cpp(dets)
                out = trk.update(t[k], a, None, r)
                cov = np.array(out[0]["cov"]).reshape(4, 4) if out else None
        got[mode] = cov
    assert got["off"][2, 2] == pytest.approx(closed(R * sb), rel=1e-9)
    assert got["off"][3, 3] == pytest.approx(closed(sr), rel=1e-9)
    assert got["update"][2, 2] == pytest.approx(closed(R * sb), rel=1e-9)            # across: nothing from the rate
    assert got["update"][3, 3] == pytest.approx(closed(sr, srate), rel=1e-9)        # along: the closed form with the rate
    # Six pings, 12 s apart, range sigma 2 m, rate sigma 0.06 m/s: the along velocity's sigma 0.0398 -> 0.0209 m/s; across
    # (52 m at 3 km) 1.034 m/s either way (ours, this geometry).
    assert np.sqrt(got["off"][3, 3]) == pytest.approx(0.0398, abs=5e-5)
    assert np.sqrt(got["update"][3, 3]) == pytest.approx(0.0209, abs=5e-5)


def _run_mover(range_rate, seed, n_pings=8, p0=(400.0, 3000.0), v=(1.6, -1.2), platform_v=(0.0, 0.0), range_sigma=20.0,
               rate_sigma=0.04, T=12.0, engine="python", naive=False):
    """A constant-velocity target from a platform moving at platform_v from the origin, detections in range and bearing
    (noise range_sigma, 1 degree) and, with the rate on, its measured rate relative to the platform (noise rate_sigma).
    naive: the platform's velocity withheld from the tracker (the rate read as over ground). Returns the tracks per ping."""
    rng = np.random.default_rng(seed)
    kw = dict(process_noise=0.05, range_sigma=range_sigma, bearing_sigma=np.radians(1.0), initial_speed_sigma=8.0,
              confirm_hits=3, max_misses=2, range_rate=range_rate)
    trk = detect.Tracker(**kw) if engine == "python" else core.LiveTracker(**kw)
    out = []
    for k in range(n_pings):
        tk = T * k
        own = np.array(platform_v) * tk
        p = np.array(p0) + np.array(v) * tk
        rel = p - own
        r = np.hypot(*rel) + range_sigma * rng.standard_normal()
        b = np.arctan2(rel[0], rel[1]) + np.radians(1.0) * rng.standard_normal()
        q = own + r * np.array([np.sin(b), np.cos(b)])
        plat = (own[0], own[1], platform_v[0], platform_v[1])
        z = true_rate(p, v, plat) + rate_sigma * rng.standard_normal()
        d = det(q[0], q[1], rate=z, sigma=rate_sigma, platform=(own[0], own[1], 0.0, 0.0) if naive else plat)
        if engine == "python":
            out.append(copy.deepcopy(trk.update(tk, [d])))        # the tracks as they were at this ping
        else:
            a, _, rr = as_cpp([d])
            out.append(trk.update(tk, a, None, rr))
    return out


def test_with_noise_the_velocity_error_shrinks_as_the_covariance_says():
    """Forty seeds of a mover at 3 km (1.6 m/s across, -1.2 m/s along the line of sight; range sigma 20 m, as a tone's
    coarse range; bearing 1 degree; rate sigma 0.04 m/s), eight pings: the RMS velocity error along the line of sight falls
    by more than three times with the rate, and stays within 1.5 times what the filter's covariance says."""
    v = np.array([1.6, -1.2])
    err = {"off": [], "update": []}
    pred = {"off": [], "update": []}
    for mode in err:
        for seed in range(40):
            tr = _run_mover(mode, seed, v=tuple(v))[-1][0]
            u = tr.state[:2] / np.linalg.norm(tr.state[:2])
            err[mode].append(float(u @ (tr.state[2:] - v)))
            pred[mode].append(float(u @ tr.cov[2:, 2:] @ u))
    rms = {m: float(np.sqrt(np.mean(np.square(e)))) for m, e in err.items()}
    want = {m: float(np.sqrt(np.mean(p))) for m, p in pred.items()}
    assert rms["update"] < rms["off"] / 3.0, rms
    for m in rms:
        assert rms[m] < 1.5 * want[m], (m, rms, want)


# --- the classes ---------------------------------------------------------------------------------------------------------

LIVE_CLASSIFY = dict(min_speed=1.0, static_speed=1.0)        # live.py DEFAULTS classify


def _first_mover(per_ping, range_rate):
    for k, tracks in enumerate(per_ping):
        kinds = [c.kind for c in classify.classify_tracks(tracks, range_rate=range_rate, **LIVE_CLASSIFY)]
        if "mover" in kinds:
            return k + 1
    return None


def test_a_mover_with_the_rate_is_classed_a_mover_in_fewer_pings():
    """The mover of the test above (2 m/s, 1.6 across and -1.2 along the line of sight, at 3 km, 12 s pings), twenty
    seeds: the ping on which it is first a mover. Without the rate (the line fit alone) 9 to 14 pings, median 12; with it
    4 (the classifier's min_points) on 18 seeds, 5 and 6 on the others (ours, measured here; a seed that never makes it
    within 16 counts 17)."""
    off, on = [], []
    for seed in range(20):
        off.append(_first_mover(_run_mover("off", seed, n_pings=16), False) or 17)
        on.append(_first_mover(_run_mover("update", seed, n_pings=16), True) or 17)
    assert np.median(on) == 4 and max(on) <= 6, on
    assert np.median(off) >= 10 and min(off) >= 8, off


def test_a_still_target_with_a_zero_rate_stays_static():
    """A still target at 3 km, twenty seeds, sixteen pings, the rate on: never a mover; on every ping the class the line
    fit gives without the rate (static on 251 of 260 classed pings, uncertain on 9: the fit's own scatter rule; ours,
    measured), and static at the last ping on every seed."""
    for seed in range(20):
        per = _run_mover("update", seed, n_pings=16, v=(0.0, 0.0))
        for tracks in per[3:]:
            kinds = [c.kind for c in classify.classify_tracks(tracks, range_rate=True, **LIVE_CLASSIFY)]
            assert "mover" not in kinds and kinds == [c.kind for c in classify.classify_tracks(tracks, **LIVE_CLASSIFY)], (seed, kinds)
        assert [c.kind for c in classify.classify_tracks(per[-1], range_rate=True, **LIVE_CLASSIFY)] == ["static"], seed


def test_a_still_target_from_a_moving_platform_stays_static_and_a_naive_rate_would_not():
    """The platform under way at 1.5 m/s along x, a still target ahead and to the side: its measured rate relative to the
    platform is -u . v_o (-1.43 m/s at the first ping, falling as the platform passes), not zero. Taken relative to the platform the track stays static;
    read as over ground (the platform's velocity withheld) the track is a mover or broken on almost every seed: the case
    that breaks a naive form.
    (With the rate taken correctly: static on 174 of 180 classed pings, uncertain on 6, as without the rate.)"""
    naive_wrong = 0
    for seed in range(20):
        per = _run_mover("update", seed, n_pings=12, p0=(2500.0, 800.0), v=(0.0, 0.0), platform_v=(1.5, 0.0))
        rate0 = true_rate((2500.0, 800.0), (0.0, 0.0), (0.0, 0.0, 1.5, 0.0))
        assert rate0 == pytest.approx(-1.5 * 2500.0 / np.hypot(2500.0, 800.0))
        for tracks in per[3:]:
            kinds = [c.kind for c in classify.classify_tracks(tracks, range_rate=True, **LIVE_CLASSIFY)]
            assert "mover" not in kinds and kinds == [c.kind for c in classify.classify_tracks(tracks, **LIVE_CLASSIFY)], (seed, kinds)
        assert [c.kind for c in classify.classify_tracks(per[-1], range_rate=True, **LIVE_CLASSIFY)] == ["static"], seed
        assert per[-1][0].hits == 12, seed                              # one track, every detection taken
        naive = _run_mover("update", seed, n_pings=12, p0=(2500.0, 800.0), v=(0.0, 0.0), platform_v=(1.5, 0.0), naive=True)
        kinds = [c.kind for c in classify.classify_tracks(naive[-1], range_rate=True, **LIVE_CLASSIFY)]
        naive_wrong += kinds != ["static"] or naive[-1][0].hits < 12
    # Measured (ours): the naive form gives a mover on 5 seeds, an uncertain or a broken track on 6, and a static track that
    # lost detections (the rate pulls it off the positions) on 9.
    assert naive_wrong >= 18, naive_wrong


@pytest.mark.parametrize("association", ["likelihood", "in_order"])
def test_an_untrusted_rate_changes_nothing(association):
    """Every detection carries a scattered rate (the sweep bank's on an extended target: +-3 m/s, D-232) flagged not
    trusted (range_rate_sigma NaN, as the chain sets it for doppler_path sweep): with the rate on, the tracks, their
    covariances and the classes are those with it off, bit for bit; and with no rate at all the same."""
    runs = {}
    for mode in ("off", "update", "update_and_gate"):
        trk = detect.Tracker(confirm_hits=2, max_misses=2, association=association, range_rate=mode)
        rng, rr = np.random.default_rng(5), np.random.default_rng(9)
        for k in range(8):
            dets = []
            for p in ((300.0 + 2.0 * k, 2000.0), (-500.0, 2500.0 - 1.5 * k)):
                q = np.array(p) + 3.0 * rng.standard_normal(2)
                dets.append(det(*q, rate=float(rr.uniform(-3, 3)), sigma=NAN, platform=(0.0, 0.0, 0.0, 0.0)))
            out = trk.update(12.0 * k, dets)
        runs[mode] = [(t.id, t.state.tobytes(), t.cov.tobytes(), t.rate_updates) for t in out]
        runs[mode + "_cls"] = [(c.kind, c.velocity) for c in classify.classify_tracks(out, range_rate=mode != "off")]
    assert runs["off"] == runs["update"] == runs["update_and_gate"]
    assert runs["off_cls"] == runs["update_cls"] == runs["update_and_gate_cls"]
    assert all(r[3] == 0 for r in runs["update"])


def test_update_and_gate_refuses_a_pair_whose_rate_is_far_from_the_prediction():
    """A confirmed track on a target closing at 1.2 m/s with good rates; on the last ping its detection comes with a rate of
    +2.0 m/s (a wrong merge): with `update` the track takes it (and its velocity is pulled), with `update_and_gate` the pair
    is outside the gate (the tap says rate_d2 >= gate^2), the track misses and a new track is born on it."""
    def run(mode):
        trk = detect.Tracker(confirm_hits=2, max_misses=2, range_rate=mode, range_sigma=5.0)
        trk.tap = True
        for k in range(6):
            y = 2000.0 - 1.2 * 12.0 * k
            trk.update(12.0 * k, [det(0.0, y, rate=-1.2, sigma=0.05)])
        trk.update(72.0, [det(0.0, 2000.0 - 1.2 * 72.0, rate=2.0, sigma=0.05)])
        return trk
    upd, gate = run("update"), run("update_and_gate")
    assert [t.id for t in upd.tracks] == [1] and upd.tracks[0].misses == 0
    assert sorted(t.id for t in gate.tracks) == [1, 2] and gate.tracks[0].misses == 1
    pair = [p for p in gate.last_tap["pairs"] if p["track"] == 1][0]
    assert pair["rate_d2"] >= gate.gate**2 and not pair["gated"]
    assert "rate_d2" not in upd.last_tap["pairs"][0]


def test_the_tap_shows_the_rate_update_only_with_the_rate_on():
    for mode in ("off", "update"):
        trk = detect.Tracker(confirm_hits=1, range_rate=mode)
        trk.tap = True
        trk.update(0.0, [det(0.0, 1000.0, rate=-1.0, sigma=0.05)])
        trk.update(12.0, [det(0.0, 988.0, rate=-1.0, sigma=0.05)])
        tap = trk.last_tap
        if mode == "off":
            assert set(tap) == set(detect.TRACKER_TAP)
            continue
        assert set(tap) == set(detect.TRACKER_TAP) | {"range_rate", "rate"} and tap["range_rate"] == "update"
        (r,) = tap["rate"]
        assert r["track"] == 1 and r["detection"] == 0 and r["z"] == -1.0
        assert r["innovation"] == pytest.approx(r["z"] - r["predicted"]) and r["s"] > r["sigma"] ** 2


# --- the two engines ----------------------------------------------------------------------------------------------------

def _same_tap(a, b, tol=1e-8):
    assert set(a) == set(b)
    assert [(p["track"], p["detection"], p["gated"], p["nearest"]) for p in a["pairs"]] == \
           [(p["track"], p["detection"], p["gated"], p["nearest"]) for p in b["pairs"]]
    for p, q in zip(a["pairs"], b["pairs"]):
        assert p["d2"] == pytest.approx(q["d2"], rel=tol, abs=tol)
        if "rate_d2" in p:
            assert (p["rate_d2"] is None) == (q["rate_d2"] is None)
            if p["rate_d2"] is not None:
                assert p["rate_d2"] == pytest.approx(q["rate_d2"], rel=tol, abs=tol)
    if "rate" in a:
        assert a["range_rate"] == b["range_rate"]
        assert [(r["track"], r["detection"]) for r in a["rate"]] == [(r["track"], r["detection"]) for r in b["rate"]]
        for r, s in zip(a["rate"], b["rate"]):
            for k in ("z", "predicted", "innovation", "s", "sigma"):
                assert r[k] == pytest.approx(s[k], rel=tol, abs=tol), k


@pytest.mark.parametrize("mode", ["off", "update", "update_and_gate"])
@pytest.mark.parametrize("association", ["likelihood", "in_order"])
def test_the_two_trackers_and_classifiers_are_equal_with_the_rate(mode, association):
    """Two movers, a still target, a platform under way, clutter that comes and goes, some rates trusted and some not, a
    few wrong: the Python and C++ trackers' tracks, covariances, rate counts, taps and classes equal (1e-8)."""
    rng = np.random.default_rng(12)
    kw = dict(process_noise=0.1, range_sigma=8.0, bearing_sigma=np.radians(0.8), confirm_hits=3, max_misses=2,
              initial_speed_sigma=6.0, association=association, range_rate=mode)
    py, cpp = detect.Tracker(**kw), core.LiveTracker(**kw)
    py.tap = True
    cpp.set_tap(True)
    vo = np.array([1.2, 0.4])
    truth = [((300.0, 1500.0), (1.5, -0.8)), ((-400.0, 1800.0), (0.0, 0.0)), ((100.0, 1200.0), (-0.6, 1.4))]
    for k in range(10):
        t = 12.0 * k
        own = vo * t
        plat = (own[0], own[1], vo[0], vo[1])
        dets = []
        for p0, v in truth:
            p = np.array(p0) + np.array(v) * t
            q = p + 4.0 * rng.standard_normal(2)
            z = true_rate(p, v, plat) + 0.04 * rng.standard_normal()
            trusted = rng.random() < 0.7
            if rng.random() < 0.1:
                z += 2.5                                        # a wrong merge now and then
            dets.append(det(*q, rate=z, sigma=0.05 if trusted else NAN, platform=plat, time=t + np.hypot(*(q - own)) / 1500.0))
        if rng.random() < 0.6:
            q = rng.uniform(-600, 600, 2) + np.array([0.0, 1500.0])
            dets.append(det(*q, snr=13.0, rate=float(rng.uniform(-3, 3)), sigma=0.05, platform=plat))
        a = py.update(t, dets)
        cd, ct, cr = as_cpp(dets)
        b = cpp.update(t, cd, ct, cr)
        assert [tr.id for tr in a] == [tr["id"] for tr in b]
        for tr, cr_ in zip(a, b):
            assert np.allclose(tr.state, cr_["state"], rtol=1e-9, atol=1e-8)
            assert np.allclose(tr.cov.reshape(-1), cr_["cov"], rtol=1e-9, atol=1e-9)
            assert tr.rate_updates == cr_["rate_updates"] and tr.hits == cr_["hits"]
            assert np.allclose(np.array(tr.measured), np.array(cr_["measured"]), atol=1e-8, equal_nan=True)
        _same_tap(py.last_tap, cpp.last_tap())
        if mode == "off":
            assert all(tr.rate_updates == 0 for tr in a)
    if mode != "off":
        assert any(tr.rate_updates > 0 for tr in a)
    origin = tuple(vo * 108.0)
    for rr in (False, True):
        want = classify.classify_tracks(a, range_rate=rr, origin=origin, time=t, **LIVE_CLASSIFY)
        got = cpp.classify(range_rate=rr, origin_x=origin[0], origin_y=origin[1], time=t, **LIVE_CLASSIFY)
        assert [c.kind for c in want] == [c["kind"] for c in got]
        for w, g in zip(want, got):
            assert np.allclose(w.velocity, g["velocity"], atol=1e-8) and w.speed == pytest.approx(g["speed"], rel=1e-9, abs=1e-12)
            assert w.speed_sigma == pytest.approx(g["speed_sigma"], rel=1e-8, abs=1e-12)
    if mode != "off":
        assert "mover" in [c.kind for c in want]


def test_the_two_classifiers_taps_carry_the_rate_evidence_only_with_the_rate_on():
    tracks = _run_mover("update", 1, n_pings=6)[-1]
    for rr in (False, True):
        ev = []
        classify.classify_tracks(tracks, range_rate=rr, evidence=ev, **LIVE_CLASSIFY)
        assert ("filter_chi2" in ev[0]) == rr
        if rr:
            assert ev[0]["rule"] == "mover_range_rate" and ev[0]["rate_updates"] == 6 and ev[0]["filter_chi2"] >= 25.0


@pytest.mark.parametrize("association", ["likelihood", "in_order"])
def test_with_the_rate_off_the_tracker_is_as_before_bit_for_bit(association):
    """Off (the default) and the detections' rate fields set (trusted, finite platform): both engines' tracks are those of
    a tracker given no rate at all, every member, bit for bit."""
    rng = np.random.default_rng(4)
    py_a, py_b = detect.Tracker(association=association), detect.Tracker(association=association, range_rate="off")
    cpp_a, cpp_b = core.LiveTracker(association=association), core.LiveTracker(association=association, range_rate="off")
    for k in range(8):
        pts = [(200.0 + 3.0 * k, 900.0), (-150.0, 700.0 - 2.0 * k)]
        bare = [det(*(np.array(p) + rng.standard_normal(2))) for p in pts]
        rated = [det(d.range * np.sin(d.bearing), d.range * np.cos(d.bearing), rate=-1.0, sigma=0.05, platform=(1.0, 2.0, 0.5, 0.5))
                 for d in bare]
        for d, e in zip(bare, rated):
            e.range, e.bearing = d.range, d.bearing
        a, b = py_a.update(12.0 * k, bare), py_b.update(12.0 * k, rated)
        assert [(t.id, t.state.tobytes(), t.cov.tobytes()) for t in a] == [(t.id, t.state.tobytes(), t.cov.tobytes()) for t in b]
        ca, _, _ = as_cpp(bare)
        cb, _, rb = as_cpp(rated)
        assert repr(cpp_a.update(12.0 * k, ca)) == repr(cpp_b.update(12.0 * k, cb, None, rb))


# --- the live chain (both engines; the tone path's records of test_doppler_tone.py) ------------------------------------

def _alternating(own=(0.0, 0.0, 0.0), n=8, v=(1.2, -1.0, 0.0), p0=(-150.0, 600.0, 0.0)):
    """flank_32 (array along x, broadside +y), pings 2 s apart, tone (CW) then sweep in turn; one mover over ground at v
    (1.56 m/s; its range rate over ground -1.25 m/s at the first ping). `own`: the platform's velocity (its position held
    at the origin by the record helper: the rate's own-ship part is what is tested, not the motion)."""
    import test_doppler_tone as tone
    recs = []
    for i in range(n):
        p = np.array(p0) + np.array(v) * 2.0 * i
        recs.append(tone.ping(tone.CW if i % 2 == 0 else tone.LFM, targets=((tuple(p), v),), seed=30 + i, sequence=i, own=own))
    return recs


def _chain(engine, mode, recs, taps=None):
    import test_doppler_tone as tone
    cfg = dict(tone.CHAIN, engine=engine, doppler=tone.TONE)
    if mode is not None:
        cfg["tracker"] = {"range_rate": mode}
    with gs.live.LiveChain(cfg, taps=taps) as chain:
        out = []
        for r in recs:
            rep = chain.process(r.copy())
            out.append((rep, chain.chain_taps))
    return out


def _target_kinds(reps, v=(1.2, -1.0)):
    return [[t["kind"] for t in rep["tracks"]] for rep, _ in reps]


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_the_chain_calls_the_tone_mover_static_without_the_rate_and_a_mover_with_it(engine):
    """The defect of D-241 and D-243 on a small case (ours, measured here): off, the one track on the 1.56 m/s mover is
    static from its fifth ping on (the line fit cannot tell it from still); with `range_rate: update` a mover from the fifth
    ping (the first with a class: min_points 4 and confirm_hits 3), at a speed within 0.3 m/s of the truth."""
    recs = _alternating()
    off, on = _chain(engine, None, recs), _chain(engine, "update", recs)
    assert _target_kinds(off) == [[], [], [], ["uncertain"]] + [["static"]] * 4
    assert _target_kinds(on) == [[], [], [], ["uncertain"]] + [["mover"]] * 4
    assert on[-1][0]["tracks"][0]["speed_m_s"] == pytest.approx(np.hypot(1.2, -1.0), abs=0.3)
    # The detections are the same: the option touches the tracker and the classifier only.
    assert [r["detections"] for r, _ in off] == [r["detections"] for r, _ in on]


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_the_chain_takes_the_rate_relative_to_the_moving_platform(engine):
    """The platform at 1.5 m/s along the array (x): every measured rate carries -u . v_o. The track's speed over ground is
    the target's (1.56 m/s, within 0.3), not that of the rate read as over ground."""
    recs = _alternating(own=(1.5, 0.0, 0.0))
    on = _chain(engine, "update", recs)
    assert _target_kinds(on)[-1] == ["mover"]
    t = on[-1][0]["tracks"][0]
    # The report's vx, vy are the record frame's x, y: the scene's velocity (1.2, -1.0).
    assert t["speed_m_s"] == pytest.approx(np.hypot(1.2, -1.0), abs=0.3)
    assert t["vx"] == pytest.approx(1.2, abs=0.4) and t["vy"] == pytest.approx(-1.0, abs=0.4)


@pytest.mark.parametrize("mode", ["update", "update_and_gate"])
def test_the_two_chains_report_lines_and_taps_are_equal_with_the_rate(mode):
    """With the rate on, the Python and C++ chains' report lines (detections within test_doppler_tone.py's tolerances,
    the tracks' ids and kinds exact, positions and velocities within 0.05) and the tracker taps' rate updates (1e-6)."""
    recs = _alternating(own=(1.5, 0.0, 0.0))
    taps = {"chain": ["tracker", "classifier"]}
    want, got = _chain("python", mode, recs, taps), _chain("cpp", mode, recs, taps)
    saw_rate = 0
    for (w, wt), (g, gt) in zip(want, got):
        assert len(w["detections"]) == len(g["detections"])
        for a, b in zip(w["detections"], g["detections"]):
            for key in ("range_m", "bearing_deg", "snr_db"):
                assert a[key] == pytest.approx(b[key], abs=0.05), key
            assert a["doppler_path"] == b["doppler_path"]
        assert [(t["id"], t["kind"]) for t in w["tracks"]] == [(t["id"], t["kind"]) for t in g["tracks"]]
        for a, b in zip(w["tracks"], g["tracks"]):
            for key in ("x", "y", "vx", "vy", "speed_m_s"):
                assert a[key] == pytest.approx(b[key], abs=0.05), key
        if wt["tracker"] is not None:
            assert wt["tracker"]["range_rate"] == gt["tracker"]["range_rate"] == mode
            _same_tap(wt["tracker"], gt["tracker"], tol=1e-6)
            saw_rate += len(wt["tracker"]["rate"])
        assert [(e["rule"], e["rate_updates"]) for e in wt["classifier"]] == [(e["rule"], e["rate_updates"]) for e in gt["classifier"]]
    assert saw_rate == 3              # tone pings 2, 4, 6 (ping 0, a tone ping before the first sweep, is not tracked: D3)
    assert "mover_range_rate" in {e["rule"] for _, t in want for e in t["classifier"]}


def test_which_rates_the_chain_trusts():
    """live.range_rate_sigma (chain.cpp the same): the tone path's rate (doppler_path tone or both) with one step and the
    navigation's error in quadrature; the sweep's (path sweep: the sweep bank's, D-232) only with range_rate_from any;
    nothing with the option off, or without a rate or a step."""
    from gpusonar.live import range_rate_sigma
    mk = lambda path, rate=0.5, step=0.04: detect.Detection(1000.0, 0.0, 20.0, 1, range_rate=rate, range_rate_step=step, doppler_path=path)  # noqa: E731
    assert range_rate_sigma(mk(1), "update", "tone", 0.03) == pytest.approx(0.05)
    assert range_rate_sigma(mk(2), "update_and_gate", "tone", 0.03) == pytest.approx(0.05)
    assert np.isnan(range_rate_sigma(mk(0), "update", "tone", 0.03))
    assert range_rate_sigma(mk(0), "update", "any", 0.03) == pytest.approx(0.05)
    assert np.isnan(range_rate_sigma(mk(1), "off", "tone", 0.03))
    assert np.isnan(range_rate_sigma(mk(1, rate=NAN), "update", "tone", 0.03))
    assert np.isnan(range_rate_sigma(mk(1, step=NAN), "update", "tone", 0.03))


def test_the_option_is_off_by_default_said_only_when_on_and_bad_words_refused():
    from gpusonar import live
    assert live.DEFAULTS["tracker"]["range_rate"] == "off"
    assert not any(k.startswith("tracker_range_rate") for k in live.cpp_chain_options({}))
    on = live.cpp_chain_options({"tracker": {"range_rate": "update_and_gate", "range_rate_from": "any", "range_rate_nav_sigma_m_s": 0.02}})
    assert on["tracker_range_rate"] == "update_and_gate" and on["tracker_range_rate_from"] == "any"
    assert on["tracker_range_rate_nav_sigma_m_s"] == 0.02
    for bad in ({"range_rate": "on"}, {"range_rate": "update", "range_rate_from": "bank"},
                {"range_rate": "update", "range_rate_nav_sigma_m_s": -0.1}):
        with pytest.raises(ValueError):
            live.cpp_chain_options({"tracker": bad})
    with pytest.raises(ValueError):
        detect.Tracker(range_rate="yes")
    with pytest.raises(ValueError):
        core.LiveTracker(range_rate="yes")
    for bad in ({"tracker_range_rate": "on"}, {"tracker_range_rate_from": "bank"}, {"tracker_range_rate_nav_sigma_m_s": -1.0}):
        with pytest.raises(ValueError):
            core.LiveChainCpp(dict(live.cpp_chain_options({}), **bad))


def test_the_program_and_the_binding_read_the_option_alike_and_dump_it_only_when_on():
    """sonar_live's key=value options (libs/live options.cpp) and the binding (LiveChainCpp) take the same keys; the
    options' dump says them only with the rate on, so every dump before D5 is the one it was."""
    import json

    from gpusonar import live
    defaults = json.loads(core.live_options_from_text(""))
    assert not any(k.startswith("tracker_range_rate") for k in defaults)
    chain = {"tracker": {"range_rate": "update_and_gate", "range_rate_from": "any", "range_rate_nav_sigma_m_s": 0.02}}
    flat = live.program_options(chain)
    via_text = json.loads(core.live_options_from_text("".join(f"{k}={v}\n" for k, v in flat.items())))
    via_binding = json.loads(core.LiveChainCpp(live.cpp_chain_options(chain)).options())
    for got in (via_text, via_binding):
        assert got["tracker_range_rate"] == "update_and_gate" and got["tracker_range_rate_from"] == "any"
        assert got["tracker_range_rate_nav_sigma_m_s"] == pytest.approx(0.02)
    for pair in ("tracker_range_rate=on", "tracker_range_rate_from=bank", "tracker_range_rate_nav_sigma_m_s=-1"):
        with pytest.raises(ValueError, match=pair.split("=")[0]):
            core.live_options_from_text(pair)
