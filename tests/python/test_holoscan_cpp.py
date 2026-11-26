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
import test_board_streams as tbs
from test_board_streams import trial  # noqa: F401  (a fixture)
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
    """sonar-ogg: the chain operator, downstream of UdpSourceOp, throws on its first ping. The application ends with
    the reason and an error status, not std::terminate from UdpSourceOp's threads (the fault D-183 fixed in
    build/bin/sonar_live). This passed before the fix too: Holoscan 4.6 deactivates the graph after an operator's
    exception, which calls UdpSourceOp::stop(), and run() rethrows (A10G job 6b063a21d0582cfbbefd60a3; an operator
    whose start() throws ended with status 1 too). The destructor that now also joins the threads guards the case no
    run here produced: the operator destroyed without stop(). The throw is D-183's: the
    one-row trial's stream (water at 1496.8 m/s) received with the window laid out at the nominal 1500 m/s, so the
    reference bursts are not where sync looks."""
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


@pytest.mark.parametrize("mode", ["pings", "listen"])
@pytest.mark.parametrize("case", ["clean", "lost_zero"])
def test_holoscan_merges_per_board_streams_as_the_relay_in_front_does(trial, tmp_path, mode, case):
    """sonar-bwu: the Holoscan source operator merges per-board streams (stream_boards=1; it refused them before and
    needed tools/board_merge.py in front). On the four boards' streams of the one-row trial it reports what it reports
    on the whole-array stream the relay sends on for the same blocks (pings: the trial's chain; --listen: passive
    frames), and the merge's statistics per board are the relay merger's (test_board_streams.py holds the relay tool to
    that merger and build/bin/sonar_live to the same comparison)."""
    import shlex
    import sys
    from gpusonar.config import load_yaml
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import sonar_live
    finally:
        sys.path.pop(0)
    sc, sonar, records = trial
    name, options, blocks = next(c for c in tbs._listen_cases(sc, records, sonar) if c[0] == case)
    relayed, relay_stats = tbs._relayed(sonar, options, blocks)
    cfg = load_yaml(str(ROOT / "configs" / "deploy" / "popoto_testauv_1row.yaml"))
    cfg["source"] = dict(cfg["source"], sonar=sc["sonar"], sonar_overrides=sc.get("sonar_overrides"), host="127.0.0.1", port=0,
                         sound_speed=float(records[0].environment.sound_speed), boards=options)
    cfg.pop("nav")
    cfg["output"] = {}
    template = tmp_path / "template.h5"
    cmd = shlex.split(sonar_live.export_template(cfg, str(template), str(HOLO)))
    keys = [a for a in cmd if "=" in a and not a.startswith("-")]                    # the chain's and the merge's keys
    merge = [k for k in keys if k.startswith("stream_board")]
    assert "stream_boards=1" in merge
    if mode == "pings":
        args = ["--template", str(template), *[k for k in keys if k not in merge], "idle_seconds=6"]
    else:
        args = ["--template", str(template), "--listen", *tbs.LISTEN, "idle_seconds=6"]   # the runtime takes a moment to start
    got = {}
    for run, extra, sent in (("boards", merge, [b for _, b in blocks]), ("relayed", [], relayed)):
        code, out, err = tbs._run_program(HOLO, [*args, *extra], sent, tmp_path / f"{run}.jsonl", start_s=4.0)
        assert code == 0, f"{run}: ended with {code}: {err.strip()[-600:]}"
        summary = [ln for ln in err.splitlines() if ln.startswith("UdpSourceOp: ") and " emitted, " in ln]
        stats = tbs._boards_line(err, "UdpSourceOp: boards ")
        if run == "boards" and (stats is None or sum(b["blocks"] + b["bad_blocks"] for b in stats["boards"].values()) < len(blocks)):
            pytest.skip(f"the loopback dropped data: {stats}")
        if run == "relayed" and (not summary or " 0 blocks dropped at the full queue" not in summary[-1]):
            pytest.skip(f"the relayed stream was not all taken: {summary}")
        got[run] = ([strip(json.loads(x)) for x in out if x.startswith("{")], stats, summary[-1].split(" pings discarded")[0])
    (a, sa, ua), (b, sb, ub) = got["boards"], got["relayed"]
    assert sa == relay_stats and sb is None
    assert len(a) >= 2 and a == b and ua == ub
