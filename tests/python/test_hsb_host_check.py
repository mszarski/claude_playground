"""tools/hsb_host_check.py (sonar-02c): the read-only check of a host for Sensor Bridge's socket receiver at a Popoto
build's rate. The logic is tested on injected readings (a container like the ones used so far, a host set up as Sensor
Bridge's guide asks); one test runs the tool on this machine and checks only its exit status and JSON."""
import json
import pathlib
import subprocess
import sys

import pytest

from gpusonar.sources import hsb
from gpusonar.sources.flank_stream import HEADER_BYTES

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "hsb_host_check.py"
sys.path.insert(0, str(ROOT / "tools"))
try:
    import hsb_host_check as hc
finally:
    sys.path.pop(0)

BUILDS = json.loads((ROOT / "docs" / "handoff" / "hardware_builds.json").read_text())["builds"]
SOFTWARE_OK = {"holoscan": None, "hololink": None, "roce_receiver": "AttributeError: RoceReceiverOp", "gpusonar": None,
               "cuda_devices": 1}


def rates():
    return hc.stream_rates(BUILDS, hsb.frame_bytes, hsb.FRAME_SAMPLES)


def container():
    """A Hugging Face a10g-large job as D-186 describes it: rmem_max 212,992 B (so 425,984 granted for the guide's
    31,326,208), no isolated CPU, a 12-CPU quota with 48 CPUs listed, no CAP_SYS_NICE, raw sockets refused, the
    patched emulator switched on."""
    return {"rmem_max": 212992, "rmem_default": 212992, "socket_buffer_granted": 2 * 212992, "isolated": "",
            "online": "0-47", "affinity": list(range(48)), "cpu_max": "1200000 100000", "cpuset": "0-47",
            "cmdline": "console=ttyS0 root=/dev/vda", "rtprio_limit": [0, 0], "cap_eff": 0, "euid": 0,
            "raw_socket": False, "udp_patch_env": True, "software": dict(SOFTWARE_OK), "rdma_devices": []}


def set_up_host(core=2):
    """A host set up as the guide asks: rmem_max 31,326,208, core 2 isolated (so outside the default affinity)."""
    g = hc.GUIDE_RMEM_MAX
    return {"rmem_max": g, "rmem_default": 212992, "socket_buffer_granted": 2 * g, "isolated": str(core),
            "online": "0-7", "affinity": [c for c in range(8) if c != core], "cpu_max": "max 100000", "cpuset": "0-7",
            "cmdline": f"BOOT_IMAGE=/vmlinuz isolcpus={core}", "rtprio_limit": [0, 0], "cap_eff": 1 << hc.CAP_NET_RAW,
            "euid": 0, "raw_socket": True, "udp_patch_env": False, "software": dict(SOFTWARE_OK), "rdma_devices": []}


def by_name(report):
    return {i["item"]: i for i in report["items"]}


def test_the_guides_number_is_the_one_the_docs_cite():
    """The one number typed into the tool is Sensor Bridge's rmem_max, as docs/HOLOSCAN_NOTES.md cites it."""
    notes = (ROOT / "docs" / "HOLOSCAN_NOTES.md").read_text()
    assert f"net.core.rmem_max = {hc.GUIDE_RMEM_MAX}" in notes and "setup.mdx:784-793" in notes
    assert all(str(hc.GUIDE_RMEM_MAX) in line for line in hc.GUIDE_RMEM_COMMANDS[:1])
    assert "isolcpus=2" in notes and hc.GUIDE_RECEIVER_CORE == 2


def test_a_container_like_the_ones_used_is_not_ready_and_names_the_socket_buffer_and_the_isolated_core():
    report = hc.assess(container(), rates(), hc.GUIDE_RMEM_MAX)
    items = by_name(report)
    assert not report["ready"] and hc.exit_status(report) == 1
    assert {"net.core.rmem_max", "isolated CPUs", "socket buffer granted", "receiver core"} == set(report["missing"])
    assert items["net.core.rmem_max"]["verdict"] == hc.NOT_AS_ASKED
    assert items["net.core.rmem_max"]["fix"] == list(hc.GUIDE_RMEM_COMMANDS)
    assert items["isolated CPUs"]["verdict"] == hc.NOT_AS_ASKED and "isolcpus=2" in items["isolated CPUs"]["fix"][0]
    assert items["socket buffer granted"]["found"] == 425984
    # The CPU quota caps the chain: 12 usable of the 48 listed, so the study's OpenMP is 6 threads (D-173).
    assert items["CPUs for the chain"]["found"] == {"affinity": 48, "cgroup_cpu_max": "1200000 100000", "usable": 12}
    assert report["chain_omp_threads"] == 6 and not items["CPUs for the chain"]["blocking"]
    assert items["raw socket (emulator)"]["verdict"] == hc.OK                      # the patched emulator is asked for
    assert items["real-time priority"]["found"]["possible"] is False and not items["real-time priority"]["blocking"]
    text = hc.render(report)
    assert "has not measured the receiver" in text and "Not ready. Missing: " in text
    assert "echo 'net.core.rmem_max = 31326208' | sudo tee /etc/sysctl.d/52-hololink-rmem_max.conf" in text


def test_a_host_set_up_as_the_guide_asks_is_ready_and_gets_the_studys_command():
    report = hc.assess(set_up_host(), rates(), hc.GUIDE_RMEM_MAX)
    assert report["ready"] and report["missing"] == [] and hc.exit_status(report) == 0
    core = by_name(report)["receiver core"]
    assert core["verdict"] == hc.OK and report["receiver_core"] == 2          # the first isolated CPU, none given
    assert core["found"]["in_process_affinity"] is False and core["found"]["in_cgroup_cpuset"] is True
    study = next(c for c in report["commands"] if "GPUSONAR_HSB_STUDY=1" in c)
    assert 'GPUSONAR_HSB_STUDY_CONFIGS="400:def:4 400:31326208:4"' in study and "GPUSONAR_HSB_STUDY_RECEIVER_CORE=2" in study
    assert "-k test_popoto_rate_loopback_study" in study
    assert not any("HSB_EMULATOR_UDP_SOCKET" in c for c in report["commands"])   # raw sockets open here
    assert "Ready to measure." in hc.render(report)
    # Every test the commands name exists.
    src = (ROOT / "tests" / "python" / "test_hsb.py").read_text()
    assert "def test_popoto_rate_loopback_study" in src and "GPUSONAR_HSB_STUDY_RECEIVER_CORE" in src
    assert "def test_loopback_at_a_popoto_builds_channel_count_and_rate_gives_the_direct_chains_reports" in src


def test_a_receiver_core_that_is_not_isolated_is_not_as_asked():
    report = hc.assess(set_up_host(core=2), rates(), hc.GUIDE_RMEM_MAX, receiver_core=5)
    item = by_name(report)["receiver core"]
    assert not report["ready"] and report["missing"] == ["isolated CPUs", "receiver core"]
    assert item["verdict"] == hc.NOT_AS_ASKED and item["found"]["isolated"] is False and item["found"]["in_process_affinity"]
    assert "isolcpus=5" in item["fix"][0]
    # A core the cgroup's cpuset does not allow, or one that is not online, is refused too.
    r = dict(set_up_host(core=2), cpuset="0-1,3-7")
    item = by_name(hc.assess(r, rates(), hc.GUIDE_RMEM_MAX, receiver_core=2))["receiver core"]
    assert item["verdict"] == hc.NOT_AS_ASKED and any("cpuset" in f for f in item["fix"])
    item = by_name(hc.assess(set_up_host(), rates(), hc.GUIDE_RMEM_MAX, receiver_core=9))["receiver core"]
    assert item["verdict"] == hc.NOT_AS_ASKED and item["found"]["online"] is False


def test_unread_readings_are_unknown_and_not_ready():
    r = {k: None for k in set_up_host()}
    r["affinity"] = None
    report = hc.assess(r, rates(), hc.GUIDE_RMEM_MAX)
    items = by_name(report)
    assert not report["ready"]
    for name in ("net.core.rmem_max", "socket buffer granted", "isolated CPUs", "receiver core", "gpusonar", "CUDA device"):
        assert items[name]["verdict"] == hc.UNKNOWN, name


def test_seconds_of_stream_follow_the_builds_files():
    """The stream a Popoto build sends through the loopback, from hardware_builds.json and the frame format: samples
    plus one frame header a frame (the identity test_hsb.py holds for the trials), and what a buffer holds of it."""
    got = {r["channels"]: r for r in rates()}
    assert sorted(got) == sorted({b["channels"] for b in BUILDS}) == [32, 64, 96]
    for b in BUILDS:
        row = got[b["channels"]]
        assert b["build"] in row["builds"]
        fps = b["sample_rate_hz"] / hsb.FRAME_SAMPLES
        assert row["frames_per_s"] == pytest.approx(fps)
        assert row["bytes_per_s"] == pytest.approx(b["stream"]["payload_bytes_per_s"] + HEADER_BYTES * fps)
        assert row["frame_bytes"] == HEADER_BYTES + b["stream"]["sample_bytes"] * hsb.FRAME_SAMPLES * b["channels"]
    granted = 2 * hc.GUIDE_RMEM_MAX
    for s in hc.seconds_of_stream(granted, rates()):
        b = next(b for b in BUILDS if b["channels"] == s["channels"])
        fps = b["sample_rate_hz"] / hsb.FRAME_SAMPLES
        assert s["seconds"] == pytest.approx(granted / (b["stream"]["payload_bytes_per_s"] + HEADER_BYTES * fps))
    # Without gpusonar's frame format: the samples alone, and said so.
    bare = {r["channels"]: r for r in hc.stream_rates(BUILDS)}
    assert bare[96]["bytes_per_s"] == next(b for b in BUILDS if b["channels"] == 96)["stream"]["payload_bytes_per_s"]
    assert "headers not counted" in bare[96]["counted"]
    assert hc.seconds_of_stream(None, rates())[0]["seconds"] is None


def test_cpu_lists_and_quotas():
    assert hc.parse_cpu_list("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]
    assert hc.parse_cpu_list("") == [] and hc.parse_cpu_list(None) is None
    assert hc.quota_cpus(48, "1200000 100000") == 12
    assert hc.quota_cpus(48, "150000 100000") == 2                    # 1.5 rounded up
    assert hc.quota_cpus(4, "1200000 100000") == 4                    # never more than the affinity
    assert hc.quota_cpus(8, "max 100000") == 8 and hc.quota_cpus(8) == 8
    assert hc.quota_cpus(8, None, "-1", "100000") == 8 and hc.quota_cpus(8, None, "300000", "100000") == 3


def test_json_shape():
    report = hc.assess(container(), rates(), hc.GUIDE_RMEM_MAX)
    back = json.loads(json.dumps(report))
    assert set(back) >= {"what_this_is", "ready", "missing", "items", "socket_buffer", "streams", "receiver_core",
                         "chain_omp_threads", "commands"}
    for i in back["items"]:
        assert set(i) == {"item", "found", "wanted", "verdict", "blocking", "fix", "note"}
        assert i["verdict"] in (hc.OK, hc.NOT_AS_ASKED, hc.UNKNOWN, hc.INFO)
    assert back["socket_buffer"]["asked"] == hc.GUIDE_RMEM_MAX and back["socket_buffer"]["granted"] == 425984
    assert [s["channels"] for s in back["socket_buffer"]["seconds_of_stream"]] == [32, 64, 96]


@pytest.mark.parametrize("argv", [["--receiver-core", "-1"], ["--socket-buffer", "0"], ["--no-such-option"]])
def test_usage_errors_exit_2(argv):
    p = subprocess.run([sys.executable, str(TOOL), *argv], capture_output=True, text=True, timeout=60)
    assert p.returncode == 2 and "usage:" in p.stderr


def test_the_tool_runs_on_this_machine(tmp_path):
    """Whatever this machine is, the check exits 0 (ready) or 1 (not) with a well-formed report, and changes nothing."""
    out = tmp_path / "check.json"
    p = subprocess.run([sys.executable, str(TOOL), "--json", "--json-out", str(out)], capture_output=True, text=True,
                       timeout=300, cwd=tmp_path)
    assert p.returncode in (0, 1), p.stderr[-2000:]
    report = json.loads(p.stdout)
    assert report == json.loads(out.read_text())
    assert report["ready"] is (p.returncode == 0) and isinstance(report["missing"], list)
    assert report["readings"]["socket_buffer_granted"] is None or report["readings"]["socket_buffer_granted"] > 0
    assert sorted(f.name for f in tmp_path.iterdir()) == ["check.json"]
