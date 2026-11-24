"""tools/sonar_live.py --emulate (sonar-8zj): a scenario's pings sent as board blocks over UDP arrive as the pings the
simulator made, whatever the scenario's sound speed, and `source.sound_speed` of a `flank_udp` source lays the receive
window out (Python source; the C++ program takes the same from its template)."""

import json
import pathlib
import sys

import numpy as np
import pytest

import gpusonar as gs
from gpusonar.config import load_scenario_config
from gpusonar.sim import run_scenario
from gpusonar.sonar import SonarModel
from gpusonar.sources import FlankBoardEmulator, PingAssembler, make_source
from gpusonar.sources.flank_stream import window_sound_speed

core = gs.core
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

TRIAL = "popoto_testauv_trial_1row"             # four boards on their own clocks, water at 1496.8 m/s (D-140)
DEPLOY = ROOT / "configs" / "deploy" / "popoto_testauv_1row.yaml"
EXE = ROOT / "build" / "bin" / "sonar_live"
WINDOW = {"receive": {"window": {"range": [600.0, 1400.0]}}}
# One board (flank_32), free field, a target at 1000 m abeam in water that is not the sonar file's nominal 1500 m/s.
ONE_BOARD = {"name": "one_board_1km", "sonar": "flank_32", "fidelity": "free_field", "sonar_overrides": WINDOW,
             "environment": {"sound_speed_m_s": 1496.8, "temperature_c": 12.0, "salinity_psu": 35.0,
                             "overrides": {"sound_speed_m_s": "the test's water"}},
             "platform": {"position": [0, 0, 30.0], "rpy": [0, 0, 0], "velocity": [0, 0, 0]},
             "pings": {"count": 1, "interval": 15.0},
             "scene": [{"type": "points", "label": "target", "positions": [[0.0, 1000.0, 30.0]], "target_strength_db": 10.0}],
             "receiver": {"adc": False, "ambient_noise": {"level_db": 45.0}}, "seed": 1}


def run_tool(tmp_path, argv):
    import sonar_live
    out = tmp_path / "out"
    assert sonar_live.main([*argv, "--port", "0", "--out", str(out), "--quiet"]) == 0
    return [json.loads(line) for line in (out / "report.jsonl").read_text().splitlines()]


def nearest(report, range_m):
    """The detection nearest a range, or None."""
    return min(report["detections"], key=lambda d: abs(d["range_m"] - range_m), default=None)


# --- the key ------------------------------------------------------------------------------------------------

def test_flank_udp_lays_its_window_out_at_source_sound_speed():
    """`source.sound_speed` is the speed at which `receive.window.range` becomes samples, and what the records carry;
    absent, None or 0 it is the sonar file's nominal speed, as before."""
    sonar = SonarModel.load("flank_32", WINDOW)
    fs = sonar.sample_rate
    made = {}
    for name, extra in (("absent", {}), ("none", {"sound_speed": None}), ("zero", {"sound_speed": 0}),
                        ("stated", {"sound_speed": 1496.8})):
        source = make_source({"type": "flank_udp", "sonar": "flank_32", "sonar_overrides": WINDOW, "timeout": 0.2, **extra})
        made[name] = (source.assembler.window_start, source.assembler.window_length, source.assembler.sound_speed)
        source.close()
    start, n = sonar.window(sonar.nominal_sound_speed)
    assert made["absent"] == made["none"] == made["zero"] == (int(round(start * fs)), n, sonar.nominal_sound_speed)
    start, n = sonar.window(1496.8)
    assert made["stated"] == (int(round(start * fs)), n, 1496.8)
    assert made["stated"][:2] != made["absent"][:2]


@pytest.mark.parametrize("bad", [-1500.0, float("nan"), float("inf"), "1500", True, [1500.0]])
def test_a_sound_speed_that_is_not_a_positive_number_is_refused(bad):
    with pytest.raises(ValueError, match="source.sound_speed"):
        make_source({"type": "flank_udp", "sonar": "flank_32", "sonar_overrides": WINDOW, "sound_speed": bad})
    with pytest.raises(ValueError, match="source.sound_speed"):
        window_sound_speed(bad)


def test_a_misspelt_source_key_is_still_refused():
    with pytest.raises(TypeError, match="soundspeed"):
        make_source({"type": "flank_udp", "sonar": "flank_32", "sonar_overrides": WINDOW, "soundspeed": 1496.8})


def test_the_exported_template_and_the_python_source_lay_the_window_out_alike(tmp_path):
    """One key, both receivers: the template `--export-template` writes for build/bin/sonar_live (its PingAssembler
    takes the window from it) and the Python `flank_udp` source, with and without `source.sound_speed`; and the
    C++ assembler built from that template returns the pings a board at that speed sent, sample for sample."""
    import sonar_live
    records = run_scenario(dict(ONE_BOARD)).records
    blocks = list(FlankBoardEmulator(records, 15.0).blocks())
    for c in (None, 1496.8):
        src = {"type": "flank_udp", "sonar": "flank_32", "sonar_overrides": WINDOW, "port": 5700, "timeout": 0.2}
        if c:
            src["sound_speed"] = c
        path = str(tmp_path / f"template_{c}.h5")
        sonar_live.export_template({"source": src, "chain": {}}, path)
        template = list(make_source({"type": "container", "path": path}))[0]
        source = make_source(dict(src, port=0))
        a = source.assembler
        source.close()
        assert int(round(template.samples.start_time * template.samples.sample_rate)) == a.window_start
        assert template.samples.n_samples == a.window_length
        assert template.environment.sound_speed == a.sound_speed == (c or 1500.0)
        cpp = core.PingAssemblerCpp(template)
        got = [r for b in blocks for r in cpp.feed(b)]
        want = records[0].samples
        same = len(got) == 1 and got[0].samples.n_samples == want.n_samples and \
            got[0].samples.start_time == pytest.approx(want.start_time, abs=0.5 / want.sample_rate)    # the stream is whole samples
        assert same == bool(c)                     # at the nominal speed the window is another one (the default, unchanged)
        if c:
            lsb = records[0].metadata.adc_full_scale / 2**23
            assert np.abs(got[0].samples.array - want.array).max() <= 0.51 * lsb + 1e-12
            assert got[0].environment.sound_speed == records[0].environment.sound_speed


# --- --emulate ----------------------------------------------------------------------------------------------

def test_emulated_source_receives_the_pings_the_simulator_made(tmp_path):
    """The receiver of --emulate has the scenario's window and sound speed: what it returns is the simulated ping
    (window, sound speed, samples to half a count), not a window cut at the sonar file's nominal 1500 m/s."""
    import sonar_live
    import yaml
    path = tmp_path / "one_board_1km.yaml"
    path.write_text(yaml.safe_dump(ONE_BOARD))
    want = run_scenario(load_scenario_config(str(path))).records
    source, sender = sonar_live.emulated_source({"type": "flank_udp", "sonar": "popoto_testauv_1row_clocks", "port": 0}, str(path))
    try:
        got = list(source)
        sender.join(timeout=60.0)
    finally:
        source.close()
    if source.assembler.lost_samples:
        pytest.skip(f"the loopback dropped {source.assembler.lost_samples} samples")
    assert len(got) == len(want) == 1
    g, w = got[0], want[0]
    assert w.environment.sound_speed == pytest.approx(1496.8)
    assert g.environment.sound_speed == w.environment.sound_speed
    assert g.samples.n_samples == w.samples.n_samples
    assert g.samples.start_time == pytest.approx(w.samples.start_time, abs=0.5 / w.samples.sample_rate)
    assert np.abs(g.samples.array - w.samples.array).max() <= 0.51 * w.metadata.adc_full_scale / 2**23 + 1e-12


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_emulate_one_board_reports_the_target_at_its_range(tmp_path, engine):
    """Before: 1500 / 1496.8 too far (+0.2 %, 1002.1 m for 1000 m), the received record carrying the nominal speed.
    Now the range of the same pings given straight to the chain (--scenario)."""
    import yaml
    scenario = tmp_path / "one_board_1km.yaml"
    scenario.write_text(yaml.safe_dump(ONE_BOARD))
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text(yaml.safe_dump({"source": {"type": "flank_udp", "sonar": "flank_32", "port": 5700},
                                      "chain": {"bearings_deg": [-40.0, 40.0, 1.0], "range_cell_m": 0.5, "engine": engine}}))
    direct = run_tool(tmp_path / "direct", [str(deploy), "--scenario", str(scenario)])
    over_udp = run_tool(tmp_path / "udp", [str(deploy), "--emulate", str(scenario)])
    assert len(direct) == len(over_udp) == 1
    want, got = nearest(direct[0], 1000.0), nearest(over_udp[0], 1000.0)
    assert want is not None and got is not None
    assert abs(want["range_m"] - 1000.0) < 1.0                               # 999.5: the leading edge, in 0.5 m cells
    assert got["range_m"] == pytest.approx(want["range_m"], abs=0.05)        # 2 m apart before; 5 mm: the stream is whole samples
    assert abs(got["range_m"] - 1000.0) < 1.0


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_emulate_a_trial_with_four_boards_synchronises_them(tmp_path, engine):
    """The shipped stage-1 command of docs/handoff/EXAMPLES.md. Before: RuntimeError on the first ping, `sync: reference
    burst not found on a board`, the window cut at 1500 m/s putting the tail burst 2.5 ms from where it is looked for.
    Now every ping is synchronised and the target (at 497 m) is where the same pings, given straight to the chain, put
    it: to one 2 m range cell (the window opens on a whole sample of the stream, up to half a sample from the simulated
    record's, which moves a leading edge across a cell boundary on one ping of four) and 0.13 degrees."""
    direct = run_tool(tmp_path / "direct", [str(DEPLOY), "--scenario", TRIAL, "--engine", engine])
    over_udp = run_tool(tmp_path / "udp", [str(DEPLOY), "--emulate", TRIAL, "--engine", engine])
    if len(over_udp) < len(direct):
        pytest.skip(f"the loopback dropped data: {len(over_udp)} of {len(direct)} pings")
    assert len(over_udp) == len(direct) == 4
    for got, want in zip(over_udp, direct):
        assert got["sync"]["ok"] and want["sync"]["ok"]
        assert got["sync"]["worst_residual_us"] == pytest.approx(want["sync"]["worst_residual_us"], abs=0.01) and got["sync"]["worst_residual_us"] < 1.0
        assert got["sync"]["offsets_us"] == pytest.approx(want["sync"]["offsets_us"], abs=0.001)
        a, b = nearest(got, 497.0), nearest(want, 497.0)
        assert a is not None and b is not None and abs(b["range_m"] - 497.0) < 3.0
        assert a["range_m"] == pytest.approx(b["range_m"], abs=2.1)
        assert abs(a["bearing_deg"] - b["bearing_deg"]) < 0.25


@pytest.mark.skipif(not EXE.exists(), reason="build/bin/sonar_live is not built")
def test_the_program_synchronises_a_trial_stream_with_the_template_of_its_sound_speed(tmp_path):
    """build/bin/sonar_live has no sonar file: its receive window is the template's. With `source.sound_speed` the
    scenario's, the command `--export-template` prints synchronises the four boards of the emulated trial stream;
    without it (the nominal 1500 m/s) the same stream is not synchronised, as in the Python source before."""
    import shlex
    import socket
    import subprocess
    import time
    import sonar_live
    from gpusonar.config import load_yaml
    sc = load_scenario_config(TRIAL)
    records = run_scenario(sc).records
    outcome = {}
    for name, c in (("stated", float(records[0].environment.sound_speed)), ("nominal", None)):
        cfg = load_yaml(str(DEPLOY))
        cfg["source"] = dict(cfg["source"], sonar=sc["sonar"], sonar_overrides=sc.get("sonar_overrides"), timeout=15.0, host="127.0.0.1")
        if c:
            cfg["source"]["sound_speed"] = c
        cfg.pop("nav")
        cfg["output"] = {"report": str(tmp_path / f"{name}.jsonl")}
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("127.0.0.1", 0))
        cfg["source"]["port"] = probe.getsockname()[1]
        probe.close()
        cmd = shlex.split(sonar_live.export_template(cfg, str(tmp_path / f"{name}.h5"), str(EXE)))
        proc = subprocess.Popen([*cmd, "--pings", str(len(records))], stderr=subprocess.PIPE, text=True)
        try:
            time.sleep(1.0)
            FlankBoardEmulator(records, float(sc["pings"]["interval"])).send(cfg["source"]["port"], pace=0.25).join(timeout=120.0)
            _, err = proc.communicate(timeout=180.0)
        finally:
            proc.kill()
        path = tmp_path / f"{name}.jsonl"
        reports = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        outcome[name] = (proc.returncode, reports, err)
    code, reports, err = outcome["stated"]
    assert code == 0, err.strip()[-400:]
    if "0 dropped, 0 samples lost" not in err or len(reports) < len(records):
        pytest.skip(f"the loopback dropped data: {err.strip().splitlines()[-1] if err.strip() else 'no summary'}")
    assert len(reports) == 4 and all(r["sync"]["ok"] and r["sync"]["worst_residual_us"] < 1.0 for r in reports)
    assert all(abs(nearest(r, 497.0)["range_m"] - 497.0) < 3.0 for r in reports)
    code, reports, err = outcome["nominal"]
    # The default is unchanged: at 1500 m/s this stream's bursts are not where the window puts them. (The program then
    # ends abnormally; its message is not pinned here.)
    assert code != 0 or not reports or not all(r["sync"]["ok"] for r in reports), "the default window moved"
