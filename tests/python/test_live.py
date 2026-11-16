"""The live chain: the same object processes simulated pings, a replayed recording and
blocks arriving over UDP, and reports detections and classified tracks per ping."""

import json

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import taps
from gpusonar.live import LiveChain
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel
from gpusonar.sources import FlankBoardEmulator, make_source

OVERRIDES = {"receive": {"window": {"range": [100.0, 400.0]}}}
INTERVAL = 2.0
CHAIN = {"bearings_deg": [-40.0, 40.0, 1.0], "range_cell_m": 1.0, "cfar": {"pfa": 1e-6, "guard_m": 15.0, "train_m": 50.0},
         "min_snr_db": 12.0, "backend": "host",
         "tracker": {"range_sigma": 3.0, "bearing_sigma_deg": 1.0, "confirm_hits": 3, "max_misses": 2},
         "classify": {"min_speed": 1.0, "static_speed": 0.6, "max_fit_rms": 10.0, "multipath": None}}
N = 6


@pytest.fixture(scope="module")
def pings():
    """Ownship making 1.5 m/s along its array; one target crossing at 3 m/s, one standing still."""
    sonar = SonarModel.load("flank_32", OVERRIDES)
    rng = np.random.default_rng(5)
    out = []
    for i in range(N):
        t = i * INTERVAL
        scene = Scene().add_points([[40.0 - 3.0 * t, 250.0, 0.0], [-60.0, 320.0, 0.0]], target_strength_db=10.0)
        out.append(simulate_ping(sonar, make_pose(position=(1.5 * t, 0.0, 0.0), velocity=(1.5, 0.0, 0.0)), scene,
                                 environment={"sound_speed": 1500.0}, sequence=i, time_ns=int(t * 1e9),
                                 noise_rms=1e-5, rng=rng))
    return out


def copies(pings):
    return [p.copy() for p in pings]


def nearest(report, x, y):
    return min(report["tracks"], key=lambda t: np.hypot(t["x"] - x, t["y"] - y))


def test_chain_tracks_and_classifies_from_a_moving_ownship(pings, tmp_path):
    with LiveChain(CHAIN, recording=str(tmp_path / "rec.h5"), report_path=str(tmp_path / "rep.jsonl"),
                   taps={"dir": str(tmp_path / "taps"), "pings": 1}) as chain:
        reports = chain.run(copies(pings))
    last = reports[-1]
    t_end = (N - 1) * INTERVAL
    mover = nearest(last, 40.0 - 3.0 * t_end, 250.0)
    assert mover["kind"] == "mover" and np.hypot(mover["x"] - (40.0 - 3.0 * t_end), mover["y"] - 250.0) < 6.0
    assert mover["vx"] == pytest.approx(-3.0, abs=0.6) and abs(mover["vy"]) < 0.6
    still = nearest(last, -60.0, 320.0)
    assert still["kind"] == "static" and still["speed_m_s"] < 0.6      # ownship's 1.5 m/s is not put on it
    assert all(len(r["detections"]) >= 2 for r in reports)
    # The report file has one line per ping; the taps hold every stage of the first ping.
    lines = (tmp_path / "rep.jsonl").read_text().splitlines()
    assert len(lines) == N and json.loads(lines[-1])["tracks"] == last["tracks"]
    assert [e["stage"] for e in taps.manifest(tmp_path / "taps" / "ping0000")] == ["raw", "prepare", "beamform", "report"]
    beam = taps.read_arrays(tmp_path / "taps" / "ping0000", "beamform")
    assert beam["power"].shape == (len(beam["bearings"]), len(beam["ranges"]))


def test_replaying_the_recording_gives_the_same_reports(pings, tmp_path):
    with LiveChain(CHAIN, recording=str(tmp_path / "rec.h5")) as chain:
        first = chain.run(copies(pings))
    raw = list(make_source({"type": "container", "path": str(tmp_path / "rec.h5")}))
    assert np.array_equal(raw[0].samples.array, pings[0].samples.array)   # recorded before processing
    with LiveChain(CHAIN) as chain:
        again = chain.run(raw)
    for a, b in zip(first, again):
        assert [(t["id"], t["kind"]) for t in a["tracks"]] == [(t["id"], t["kind"]) for t in b["tracks"]]
        assert np.allclose([t["x"] for t in a["tracks"]], [t["x"] for t in b["tracks"]], atol=1e-6)


def test_beamformers_agree_on_where_the_targets_are(pings):
    out = {}
    for name in ("backprojection", "farfield"):
        with LiveChain(dict(CHAIN, beamformer=name)) as chain:
            out[name] = chain.run(copies(pings))[-1]
    for x, y in ((40.0 - 3.0 * (N - 1) * INTERVAL, 250.0), (-60.0, 320.0)):
        a, b = nearest(out["backprojection"], x, y), nearest(out["farfield"], x, y)
        assert a["kind"] == b["kind"] and np.hypot(a["x"] - b["x"], a["y"] - b["y"]) < 4.0


@pytest.mark.skipif(not gs.cuda_available(), reason="CUDA backend not built or no GPU")
def test_gpu_chain_matches_the_host_chain(pings):
    out = {}
    for backend in ("host", "cuda"):
        with LiveChain(dict(CHAIN, backend=backend)) as chain:
            out[backend] = chain.run(copies(pings))[-1]
    assert [(t["id"], t["kind"]) for t in out["host"]["tracks"]] == [(t["id"], t["kind"]) for t in out["cuda"]["tracks"]]
    assert np.allclose([t["x"] for t in out["host"]["tracks"]], [t["x"] for t in out["cuda"]["tracks"]], atol=0.5)


def test_blocks_over_udp_through_the_chain(pings):
    source = make_source({"type": "flank_udp", "sonar": "flank_32", "sonar_overrides": OVERRIDES, "max_pings": N, "timeout": 5.0})
    try:
        sender = FlankBoardEmulator(pings, INTERVAL, block_samples=128).send(source.port, pace=0.25)
        with LiveChain(CHAIN) as chain:
            reports = chain.run(source)
        sender.join(timeout=60.0)
    finally:
        source.close()
    if source.assembler.lost_samples or len(reports) < N:
        pytest.skip(f"the loopback dropped {source.assembler.lost_samples} samples")
    # The stream carries samples and timing, not navigation: the assembler's records sit at the origin,
    # so only what does not need ownship's track is checked here.
    assert all(len(r["detections"]) >= 2 for r in reports)
    assert any(abs(d["range_m"] - 250.0) < 15.0 for d in reports[0]["detections"])


def test_navigation_datagrams_put_the_pose_into_the_records(pings):
    import socket
    import time
    from gpusonar.live import NavListener
    nav = NavListener(port=0, host="127.0.0.1")
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for i in range(N):                                             # a fix at every ping time: 1.5 m/s north
        t = i * INTERVAL
        tx.sendto(json.dumps({"time_ns": int(t * 1e9), "x": 1.5 * t, "y": 0.0, "z": 0.0, "yaw": 0.0, "vx": 1.5}).encode(),
                  ("127.0.0.1", nav.port))
    time.sleep(0.5)
    pos, rpy, vel = nav.at(int(1.0e9))                             # between the first two fixes
    assert pos == pytest.approx([1.5, 0.0, 0.0]) and vel[0] == pytest.approx(1.5)
    # Records as the board stream delivers them: no pose. The chain takes it from navigation.
    stripped = copies(pings)
    from gpusonar.live import apply_pose
    for r in stripped:
        apply_pose(r, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    with LiveChain(CHAIN, nav=nav) as chain:
        last = chain.run(stripped)[-1]
    assert last["pings_without_nav"] == 0 and last["ownship_xyz"][0] == pytest.approx(1.5 * (N - 1) * INTERVAL, abs=0.01)
    still = nearest(last, -60.0, 320.0)
    assert still["kind"] == "static" and still["speed_m_s"] < 0.6
    # Without navigation the same records put ownship's speed on the standing target.
    again = copies(pings)
    for r in again:
        apply_pose(r, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    with LiveChain(CHAIN) as chain:
        wrong = nearest(chain.run(again)[-1], -60.0 - 1.5 * (N - 1) * INTERVAL, 320.0)
    assert wrong["speed_m_s"] == pytest.approx(1.5, abs=0.5)
