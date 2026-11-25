"""The C++ Holoscan application (apps/sonar_live_holoscan.cpp, libs/holoscan_ops) against the
standalone program on the same input. Runs where the build had -DSONAR_BUILD_HOLOSCAN=ON."""
import json
import pathlib
import socket
import subprocess
import time

import numpy as np
import pytest

import gpusonar as gs
from gpusonar.sources import FlankBoardEmulator, make_source
from test_live import INTERVAL, N, copies, pings  # noqa: F401  (pings is a fixture)
from test_live_cpp import EXE, EXE_OPTIONS

ROOT = pathlib.Path(__file__).resolve().parents[2]
HOLO = ROOT / "build" / "bin" / "sonar_live_holoscan"
pytestmark = pytest.mark.skipif(not (HOLO.exists() and EXE.exists()), reason="built without the Holoscan C++ operators")


def lines(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.startswith("{")]


def strip(report):
    return {k: v for k, v in report.items() if k != "processing_seconds"}


def test_replay_through_holoscan_gives_the_programs_reports(pings, tmp_path):
    w = gs.RecordWriter(str(tmp_path / "pings.h5"))
    for r in copies(pings):
        w.write(r)
    w.close()
    plain = subprocess.run([str(EXE), "--replay", str(tmp_path / "pings.h5"), "--report", str(tmp_path / "a.jsonl"), *EXE_OPTIONS],
                           capture_output=True, text=True, timeout=600)
    assert plain.returncode == 0, plain.stderr
    holo = subprocess.run([str(HOLO), "--replay", str(tmp_path / "pings.h5"), "--report", str(tmp_path / "b.jsonl"),
                           "--recording", str(tmp_path / "copy.h5"), *EXE_OPTIONS], capture_output=True, text=True, timeout=600)
    assert holo.returncode == 0, holo.stderr[-2000:]
    a, b = lines(tmp_path / "a.jsonl"), lines(tmp_path / "b.jsonl")
    assert len(a) == len(b) == N
    assert [strip(x) for x in a] == [strip(x) for x in b]             # the same library, so the same numbers
    assert {t["kind"] for t in b[-1]["tracks"]} >= {"mover", "static"}
    raw = list(make_source({"type": "container", "path": str(tmp_path / "copy.h5")}))
    assert len(raw) == N and np.array_equal(raw[0].samples.array, pings[0].samples.array)   # recorded before conditioning


def test_holoscan_takes_blocks_over_udp_with_navigation(pings, tmp_path):
    w = gs.RecordWriter(str(tmp_path / "template.h5"))
    w.write(pings[0].copy())
    w.close()
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    proc = subprocess.Popen([str(HOLO), "--udp", str(port), "--template", str(tmp_path / "template.h5"), "--nav-port", str(port + 1),
                             "--report", str(tmp_path / "rep.jsonl"), "--pings", str(N), *EXE_OPTIONS],
                            stderr=subprocess.PIPE, text=True)
    try:
        time.sleep(4.0)                                                # the runtime takes a moment to start
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for i in range(N + 1):
            t = i * INTERVAL
            tx.sendto(json.dumps({"time_ns": int(t * 1e9), "x": 1.5 * t, "y": 0.0, "z": 0.0, "yaw": 0.0, "vx": 1.5}).encode(),
                      ("127.0.0.1", port + 1))
        FlankBoardEmulator(pings, INTERVAL, block_samples=128).send(port, pace=0.25).join(timeout=120.0)
        _, err = proc.communicate(timeout=180.0)
    finally:
        proc.kill()
    assert proc.returncode == 0, f"the program ended abnormally ({proc.returncode}): {err.strip()[-400:]}"   # a crash is not lost data
    reports = lines(tmp_path / "rep.jsonl")
    if "0 dropped, 0 samples lost" not in err or len(reports) < N:
        pytest.skip(f"the loopback dropped data: {[ln for ln in err.splitlines() if 'UdpSourceOp' in ln]}")
    assert "0 without navigation" in err
    assert reports[-1]["ownship_xyz"][0] == pytest.approx(1.5 * (N - 1) * INTERVAL, abs=0.6)
    assert {t["kind"] for t in reports[-1]["tracks"]} >= {"mover", "static"}


def test_an_operator_that_throws_ends_the_application_with_its_reason_not_an_abort(tmp_path):
    """sonar-ogg: the chain operator, downstream of UdpSourceOp, throws on its first ping. The application must end
    with the reason and an error status. Before, UdpSourceOp joined its receiver and navigation threads in stop() only;
    destroyed with them still running (the application unwinding from the operator's exception), std::thread's
    destructor called std::terminate: "terminate called without an active exception", SIGABRT, the reason lost - the
    fault D-183 fixed in build/bin/sonar_live. The throw is D-183's: the one-row trial's stream (water at 1496.8 m/s)
    received with the window laid out at the nominal 1500 m/s, so the reference bursts are not where sync looks."""
    import shlex
    import sys
    from gpusonar.config import load_scenario_config, load_yaml
    from gpusonar.sim import run_scenario
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import sonar_live
    finally:
        sys.path.pop(0)
    sc = load_scenario_config("popoto_testauv_trial_1row")
    records = run_scenario(sc).records
    cfg = load_yaml(str(ROOT / "configs" / "deploy" / "popoto_testauv_1row.yaml"))
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    cfg["source"] = dict(cfg["source"], sonar=sc["sonar"], sonar_overrides=sc.get("sonar_overrides"), timeout=15.0, host="127.0.0.1",
                         port=port)
    cfg.pop("nav")                                                     # the navigation thread runs too: give it a port
    cfg["output"] = {"report": str(tmp_path / "rep.jsonl")}
    cmd = shlex.split(sonar_live.export_template(cfg, str(tmp_path / "template.h5"), str(HOLO)))
    proc = subprocess.Popen([*cmd, "--nav-port", str(port + 1), "--pings", str(len(records))], stderr=subprocess.PIPE, text=True)
    try:
        time.sleep(4.0)                                                # the runtime takes a moment to start
        FlankBoardEmulator(records, float(sc["pings"]["interval"])).send(port, pace=0.25).join(timeout=120.0)
        _, err = proc.communicate(timeout=180.0)
    finally:
        proc.kill()
    print(err[-3000:])
    assert "terminate called" not in err, err.strip()[-600:]
    assert proc.returncode == 1, f"ended with {proc.returncode} (negative: a signal): {err.strip()[-600:]}"
    assert "reference burst not found" in err                          # the reason reaches the operator's log or main's
    assert not (tmp_path / "rep.jsonl").exists() or not lines(tmp_path / "rep.jsonl")
