#!/usr/bin/env python3
"""Is this host set up for Sensor Bridge's socket receiver at a Popoto build's rate? A read-only check (issue sonar-02c).

    tools/hsb_host_check.py                                  # the table, then the commands that make the measurement
    tools/hsb_host_check.py --receiver-core 2 --json-out hsb_host_check.json
    tools/hsb_host_check.py --json                           # the same as JSON only

It changes nothing on the host: it reads /proc and /sys, opens and closes a throwaway UDP socket (to see what receive
buffer the kernel grants) and a raw socket (to see whether the stock emulator could open one), and imports Sensor
Bridge, Holoscan and gpusonar in a child process. It does not set a scheduler policy: whether a real-time priority
could be set is read from RLIMIT_RTPRIO and the process's capabilities, not tried.

What it checks against, and where each comes from:
- net.core.rmem_max >= 31,326,208 and an isolated core for the receiver (isolcpus), the two settings Sensor Bridge
  2.7.0's own setup guide gives for its socket receiver (docs/user_guide/setup.mdx:784-793 and 262-296, as
  docs/HOLOSCAN_NOTES.md section 7.5 and docs/DEPLOYMENT.md cite them). The receiver's default core is 2
  (linux_receiver_op.cpp:35,66-74, HOLOSCAN_NOTES section 7.5); `receive --receiver-core` moves it.
- The Popoto builds' streams: channels and sample rate from docs/handoff/hardware_builds.json, framed as the loopback
  frames them (gpusonar.sources.hsb: FRAME_SAMPLES samples a frame, frame_bytes()).
- The chain's OpenMP threads: half the CPUs the process may use, its affinity capped by a cgroup CPU quota (D-173);
  that rule lives here (quota_cpus) and tests/python/test_hsb.py uses it.

Exit status: 0 ready to measure, 1 not ready (what is missing is listed with the command that sets it), 2 usage
error. Nothing it prints is a measurement of the receiver: that is the study it prints the command for
(docs/DEPLOYMENT.md, "Checking a host and making the measurement").
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDS_FILE = ROOT / "docs" / "handoff" / "hardware_builds.json"

# Sensor Bridge 2.7.0 docs/user_guide/setup.mdx:784-793 ("for all configurations"), cited in docs/HOLOSCAN_NOTES.md
# section 7.5 and table 7.1 ("Socket receivers") and docs/DEPLOYMENT.md.
GUIDE_RMEM_MAX = 31326208
GUIDE_RMEM_COMMANDS = ("echo 'net.core.rmem_max = 31326208' | sudo tee /etc/sysctl.d/52-hololink-rmem_max.conf",
                       "sudo sysctl -p /etc/sysctl.d/52-hololink-rmem_max.conf")
# Sensor Bridge 2.7.0 runs its receiver thread on core 2 by default (linux_receiver_op.cpp:35,66-74) and its guide
# isolates that core with `isolcpus=2` (setup.mdx:262-296, the Jetson section; docs/HOLOSCAN_NOTES.md section 7.5).
GUIDE_RECEIVER_CORE = 2
CAP_NET_RAW, CAP_SYS_NICE = 13, 23                          # linux/capability.h
OK, NOT_AS_ASKED, UNKNOWN, INFO = "ok", "not as asked", "unknown", "info"


# --- pure logic over readings --------------------------------------------------------------------------------------

def parse_cpu_list(text: str | None) -> list[int] | None:
    """A kernel CPU list ("0-3,8,10-11", "" for none) as numbers; None when it was not read."""
    if text is None:
        return None
    out = set()
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def quota_cpus(n_affinity: int, cpu_max: str | None = None, cfs_quota: str | None = None, cfs_period: str | None = None) -> int:
    """CPUs a process may use: its affinity's count, capped by a cgroup CPU quota rounded up (cgroup v2 `cpu.max`
    "QUOTA PERIOD", or v1 cpu.cfs_quota_us and cpu.cfs_period_us). A container's vCPUs are often a quota: a Hugging
    Face a10g-small job (4 vCPUs) lists the host's 96 CPUs in its affinity (job 6b04b897c97b8ed4fcd689e1, sonar-8rj;
    D-173)."""
    pair = None
    if cpu_max is not None and len(cpu_max.split()) >= 2:
        pair = cpu_max.split()[:2]
    elif cfs_quota is not None and cfs_period is not None:
        pair = cfs_quota.strip(), cfs_period.strip()
    if pair is None:
        return n_affinity
    quota, period = pair
    try:
        if quota in ("max", "-1") or float(period) <= 0:
            return n_affinity
        return max(1, min(n_affinity, math.ceil(float(quota) / float(period))))
    except ValueError:
        return n_affinity


def usable_cpus(cgroup_root="/sys/fs/cgroup", n_affinity: int | None = None) -> int:
    """quota_cpus over this process's affinity and the cgroup files under `cgroup_root` (v2, else v1's cpu/)."""
    n = len(os.sched_getaffinity(0)) if n_affinity is None else n_affinity
    root = Path(cgroup_root)
    return quota_cpus(n, read_text(root / "cpu.max"), read_text(root / "cpu" / "cpu.cfs_quota_us"),
                      read_text(root / "cpu" / "cpu.cfs_period_us"))


def stream_rates(builds: list[dict], frame_bytes=None, frame_samples: int | None = None) -> list[dict]:
    """The distinct streams of the builds (channels, sample rate), each with the builds that share it. With
    `frame_bytes(channels, frame_samples)` and `frame_samples` (gpusonar.sources.hsb), the stream as the loopback frames
    it: bytes a second = frame bytes x frames a second. Without them, the samples alone (`stream.payload_bytes_per_s`
    of hardware_builds.json; the frames' headers not counted)."""
    out = {}
    for b in builds:
        key = (int(b["channels"]), float(b["sample_rate_hz"]))
        row = out.setdefault(key, {"builds": [], "channels": key[0], "sample_rate_hz": key[1]})
        row["builds"].append(b["build"])
        if frame_bytes is not None and frame_samples:
            fps = key[1] / frame_samples
            row.update(frame_samples=frame_samples, frame_bytes=int(frame_bytes(key[0], frame_samples)), frames_per_s=fps,
                       bytes_per_s=int(frame_bytes(key[0], frame_samples)) * fps, counted="frames (samples and frame headers)")
        else:
            row.update(frame_samples=None, frame_bytes=None, frames_per_s=None,
                       bytes_per_s=float(b["stream"]["payload_bytes_per_s"]), counted="samples only (frame headers not counted)")
    return [out[k] for k in sorted(out)]


def seconds_of_stream(buffer_bytes: int | None, rates: list[dict]) -> list[dict]:
    """How long each stream takes to fill `buffer_bytes` of payload. An upper bound: the kernel charges each datagram
    more than its payload (on loopback 2,304 B for a 1,408-B datagram, measured; gpusonar/sources/hsb.py)."""
    return [{"channels": r["channels"], "builds": r["builds"],
             "seconds": None if buffer_bytes is None else buffer_bytes / r["bytes_per_s"]} for r in rates]


def _caps(cap_eff: int | None, bit: int) -> bool | None:
    return None if cap_eff is None else bool(cap_eff >> bit & 1)


def choose_receiver_core(asked: int | None, isolated: list[int] | None) -> tuple[int | None, str]:
    if asked is not None:
        return asked, "given with --receiver-core"
    if isolated:
        return isolated[0], "the first isolated CPU (none given)"
    return None, "none given and no CPU isolated"


def assess(r: dict, rates: list[dict], asked_buffer: int, receiver_core: int | None = None) -> dict:
    """The readiness report from the readings `r` (see read_host for its keys; any may be None when not read)."""
    items = []

    def item(name, found, wanted, verdict, blocking, fix=None, note=None):
        items.append({"item": name, "found": found, "wanted": wanted, "verdict": verdict, "blocking": blocking,
                      "fix": list(fix) if fix else [], "note": note})

    rmem = r.get("rmem_max")
    item("net.core.rmem_max", rmem, f">= {GUIDE_RMEM_MAX} (Sensor Bridge 2.7.0 setup.mdx:784-793)",
         UNKNOWN if rmem is None else OK if rmem >= GUIDE_RMEM_MAX else NOT_AS_ASKED, True,
         None if rmem is not None and rmem >= GUIDE_RMEM_MAX else GUIDE_RMEM_COMMANDS,
         f"rmem_default {r.get('rmem_default')}: what Sensor Bridge's receiver leaves its socket at unless asked "
         "(receive --socket-buffer)")

    granted = r.get("socket_buffer_granted")
    expected = None if rmem is None else 2 * min(asked_buffer, rmem)
    secs = seconds_of_stream(granted, rates)
    item("socket buffer granted", granted,
         f"2 x {asked_buffer} = {2 * asked_buffer} (asked {asked_buffer}; the kernel doubles, up to 2 x rmem_max"
         + (f" = {expected} here" if expected is not None else "") + ")",
         UNKNOWN if granted is None else OK if granted >= 2 * asked_buffer else NOT_AS_ASKED, True,
         None if granted is not None and granted >= 2 * asked_buffer else GUIDE_RMEM_COMMANDS if asked_buffer <= GUIDE_RMEM_MAX
         else [f"net.core.rmem_max of at least {asked_buffer} (the guide's two lines with that number), "
               "or a smaller --socket-buffer"],
         "; ".join(f"{s['channels']} ch: {s['seconds'] * 1e3:.1f} ms" for s in secs if s["seconds"] is not None)
         + (" of stream (payload; an upper bound)" if granted is not None else "")
         + ("; " + rates[0]["counted"] if granted is not None and rates and rates[0]["frame_bytes"] is None else "")
         + (f"; probe error: {r['socket_buffer_error']}" if r.get("socket_buffer_error") else ""))

    isolated = parse_cpu_list(r.get("isolated"))
    online = parse_cpu_list(r.get("online"))
    core, why = choose_receiver_core(receiver_core, isolated)
    suggest = core if core is not None else (GUIDE_RECEIVER_CORE if online is None or GUIDE_RECEIVER_CORE in online else None)
    isol_fix = [f"add isolcpus={suggest if suggest is not None else 'K'} to the kernel command line and reboot "
                "(Sensor Bridge 2.7.0's guide shows it on a Jetson: the end of the APPEND line of "
                "/boot/extlinux/extlinux.conf, setup.mdx:262-296; elsewhere, the same argument on the boot loader's "
                "kernel command line)"]
    if isolated is None:
        v = UNKNOWN
    elif not isolated:
        v = NOT_AS_ASKED
    else:
        v = OK if core is None or core in isolated else NOT_AS_ASKED
    item("isolated CPUs", "none" if r.get("isolated") == "" else r.get("isolated"), "at least the receiver's core (isolcpus; setup.mdx:262-296)", v, True,
         None if v == OK else isol_fix,
         ("kernel command line has isolcpus" if "isolcpus=" in (r.get("cmdline") or "") else "no isolcpus= on the kernel "
          "command line") if r.get("cmdline") is not None else None)

    affinity = r.get("affinity")
    cpuset = parse_cpu_list(r.get("cpuset"))
    if core is not None:
        facts = {"core": core, "why": why, "isolated": None if isolated is None else core in isolated,
                 "online": None if online is None else core in online,
                 "in_cgroup_cpuset": None if cpuset is None else core in cpuset,
                 "in_process_affinity": None if affinity is None else core in affinity}
        if facts["online"] is False or facts["in_cgroup_cpuset"] is False or facts["isolated"] is False:
            v = NOT_AS_ASKED
        elif facts["isolated"] is None or facts["online"] is None:
            v = UNKNOWN
        else:
            v = OK
        fix = []
        if facts["isolated"] is False:
            fix += isol_fix
        if facts["online"] is False:
            fix.append(f"core {core} is not online here (online: {r.get('online')}): choose another with --receiver-core")
        if facts["in_cgroup_cpuset"] is False:
            fix.append(f"allow core {core} in this cgroup's cpuset (cpuset.cpus; effective here: {r.get('cpuset')})")
        item("receiver core", facts, "isolated, online and allowed by the cgroup's cpuset", v, True, fix,
             "an isolated core is normally outside a process's default affinity; the receiver thread is moved onto it "
             "(receive --receiver-core, or HOLOLINK_AFFINITY), which the cpuset must allow"
             + (f"; HOLOLINK_AFFINITY={r['hololink_affinity']!r} is set here" if r.get("hololink_affinity") is not None else ""))
    else:
        item("receiver core", None, "a core for --receiver-core", NOT_AS_ASKED if isolated == [] else UNKNOWN, True,
             isol_fix, why)

    n_aff = None if affinity is None else len(affinity)
    usable = None if n_aff is None else quota_cpus(n_aff, r.get("cpu_max"), r.get("cfs_quota"), r.get("cfs_period"))
    omp = None if usable is None else max(1, usable // 2)
    item("CPUs for the chain", {"affinity": n_aff, "cgroup_cpu_max": r.get("cpu_max"), "usable": usable},
         "the chain's OpenMP below the core count (receive --chain-omp-threads; D-162, D-173)", INFO, False, None,
         None if omp is None else f"the study runs --chain-omp-threads {omp} (half of {usable} usable CPUs)")

    cap = r.get("cap_eff")
    rt = r.get("rtprio_limit")
    rt_possible = None
    if _caps(cap, CAP_SYS_NICE) or (rt is not None and rt[0] != 0):   # either lets sched_setscheduler ask; RLIM_INFINITY is -1
        rt_possible = True
    elif cap is not None and rt is not None:
        rt_possible = False
    item("real-time priority", {"possible": rt_possible, "rlimit_rtprio": rt, "cap_sys_nice": _caps(cap, CAP_SYS_NICE),
                                "euid": r.get("euid")},
         "not asked by Sensor Bridge's guide; receive has no option for it", INFO, False, None,
         "read from RLIMIT_RTPRIO and the capabilities, not tried (this check changes no scheduler); never tried in "
         "the containers used (DEPLOYMENT.md)")

    raw = r.get("raw_socket")
    udp = r.get("udp_patch_env")
    v = OK if raw or udp else UNKNOWN if raw is None else NOT_AS_ASKED
    item("raw socket (emulator)", {"opened": raw, "cap_net_raw": _caps(cap, CAP_NET_RAW),
                                             "HSB_EMULATOR_UDP_SOCKET": "1" if udp else None},
         "CAP_NET_RAW (root), or the emulator built with HSB_UDP_PATCH=1 and HSB_EMULATOR_UDP_SOCKET=1", v, True,
         None if v == OK else ["run the check and the study as root (CAP_NET_RAW), or:",
                               "HSB_UDP_PATCH=1 tools/build_hsb.sh /opt/hsb && export HSB_EMULATOR_UDP_SOCKET=1"],
         "needed by the loopback's emulated board only, not by the receiver (third_party/patches/"
         "hsb-2.7.0-emulator-udp-socket.patch, D-79)")

    sw = r.get("software") or {}
    for key, name, fix in (
            ("hololink", "Sensor Bridge (hololink)",
             ['eval "$(tools/build_hsb.sh /opt/hsb | tail -1)"   # builds 2.7.0; needs CUDA 12 and Holoscan 4.x (its header)']),
            ("holoscan", "Holoscan", ["the Holoscan 4.x Python wheel and C++ package (tools/build_hsb.sh's header; "
                                                "tools/gpu_job.sh GPUSONAR_HOLOSCAN_CPP=1 installs both in a container)"]),
            ("gpusonar", "gpusonar", ["build the suite with -DSONAR_BUILD_PYTHON=ON and CUDA "
                                                                    "(docs/HANDOFF.md section 3); PYTHONPATH=python"])):
        err = sw.get(key, "not checked")
        item(name, "imports" if err is None else err, "imports (hololink: its emulator and LinuxReceiverOp too)" if key == "hololink" else "imports", OK if err is None else
             UNKNOWN if err == "not checked" else NOT_AS_ASKED, True, None if err is None else fix,
             sw.get("probe_error"))
    cuda = sw.get("cuda_devices")
    item("CUDA device", cuda, ">= 1 (the receiver puts frames in GPU memory)",
         UNKNOWN if cuda is None else OK if cuda > 0 else NOT_AS_ASKED, True,
         None if cuda else ["a CUDA device and its driver (libcuda.so.1)"], sw.get("cuda_error"))

    rdma = r.get("rdma_devices")
    item("RDMA devices (RoCE)", {"devices": rdma, "RoceReceiverOp": "imports" if sw.get("roce_receiver") is None
                                              else sw.get("roce_receiver", "not checked")},
         "not needed for the socket receiver", INFO, False, None,
         "the RoCE receiver is untested here: it needs ConnectX hardware and Sensor Bridge built with RoCE "
         "(tools/build_hsb.sh builds it without, -DHOLOLINK_BUILD_ROCE=OFF); /sys/class/infiniband lists the devices "
         "(setup.mdx:98-126, HOLOSCAN_NOTES table 7.1)")

    missing = [i for i in items if i["blocking"] and i["verdict"] != OK]
    return {"what_this_is": "a readiness check of this host; it does not measure the receiver",
            "ready": not missing, "missing": [i["item"] for i in missing], "items": items,
            "socket_buffer": {"asked": asked_buffer, "granted": granted, "rmem_max": rmem, "rmem_default": r.get("rmem_default"),
                              "seconds_of_stream": secs},
            "streams": rates, "receiver_core": core, "chain_omp_threads": omp,
            "commands": measurement_commands(asked_buffer, core, udp_patch=bool(udp) or raw is False)}


def measurement_commands(asked_buffer: int, receiver_core: int | None, udp_patch: bool = False) -> list[str]:
    """The commands that make the measurement on this host, as shell lines (comments start with #)."""
    core = "K" if receiver_core is None else str(receiver_core)
    env = "PYTHONPATH=python${PYTHONPATH:+:$PYTHONPATH}"
    pre = "HSB_UDP_PATCH=1 " if udp_patch else ""
    lines = ["# 1. Sensor Bridge 2.7.0 built and on the path (skip if done; run from the repository root):",
             f'eval "$({pre}tools/build_hsb.sh /opt/hsb | tail -1)"']
    if udp_patch:
        lines.append("export HSB_EMULATOR_UDP_SOCKET=1")
    lines += ["# 2. The study: each Popoto build's trial (32, 64, 96 channels) at real time, 10 runs each, with Sensor",
              f"#    Bridge's default socket buffer and with --socket-buffer {asked_buffer}, the receiver on core {core}:",
              f"{env} GPUSONAR_HSB_STUDY=1 GPUSONAR_HSB_STUDY_RUNS=10 GPUSONAR_HSB_STUDY_ROWS=1,2,3 "
              f"GPUSONAR_HSB_STUDY_ENGINES=cpp GPUSONAR_HSB_STUDY_CONFIGS=\"400:def:4 400:{asked_buffer}:4\" "
              f"GPUSONAR_HSB_STUDY_RECEIVER_CORE={core} python3 -m pytest -s -q -p no:cacheprovider tests/python/test_hsb.py "
              "-k test_popoto_rate_loopback_study 2>&1 | tee hsb_study.log",
              "# 3. The suite's own Popoto-rate test as it runs by default (default buffer, no receiver core):",
              f"{env} python3 -m pytest -q -rxs -p no:cacheprovider tests/python/test_hsb.py "
              "-k test_loopback_at_a_popoto_builds_channel_count_and_rate 2>&1 | tee hsb_popoto.log",
              "# Send back: this check's JSON (--json-out), the lines of hsb_study.log that start with cpus:, POPOTO_STUDY",
              "# and POPOTO_LOSS, and the last 20 lines of hsb_popoto.log:",
              "grep -E '^(cpus:|POPOTO_STUDY|POPOTO_LOSS)' hsb_study.log > hsb_study_lines.txt; tail -20 hsb_popoto.log"]
    return lines


def _shown(value) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{k} {json.dumps(v)}" for k, v in value.items() if v is not None and k != "why") or "-"
    return "-" if value in (None, "") else str(value)


def render(report: dict) -> str:
    """The report as a table: one line per item (its verdict and what was found), what is wanted under it."""
    w = max(len(i["item"]) for i in report["items"])
    out = ["This checks the host only; it has not measured the receiver.", ""]
    for i in report["items"]:
        verdict = i["verdict"] + ("" if i["blocking"] else " (not required)")
        out.append(f"{i['item'].ljust(w)}  {verdict.ljust(21)}  {_shown(i['found'])}")
        out.append(f"{''.ljust(w)}  {'wanted:'.ljust(21)}  {i['wanted']}")
    out.append("")
    for i in report["items"]:
        if i["note"]:
            out.append(f"- {i['item']}: {i['note']}")
    out.append("")
    if report["ready"]:
        out.append("Ready to measure.")
    else:
        out.append("Not ready. Missing: " + ", ".join(report["missing"]) + ". What an administrator would run:")
        seen = set()
        for i in report["items"]:
            if i["blocking"] and i["verdict"] != OK:
                for line in i["fix"]:
                    if line not in seen:
                        seen.add(line)
                        out.append(f"  {line}")
    out += ["", "Then, to make the measurement:"] + [f"  {line}" for line in report["commands"]]
    return "\n".join(out)


def exit_status(report: dict) -> int:
    return 0 if report["ready"] else 1


# --- thin readers of this host -------------------------------------------------------------------------------------

def read_text(path) -> str | None:
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def read_int(path) -> int | None:
    text = read_text(path)
    try:
        return None if text is None else int(text.split()[0])
    except (ValueError, IndexError):
        return None


def probe_socket_buffer(asked: int) -> tuple[int | None, str | None]:
    """The receive buffer the kernel grants a fresh UDP socket asked for `asked` bytes with SO_RCVBUF (as `receive
    --socket-buffer` asks, gpusonar/sources/hsb.py DataSocketWatch). The socket is closed at once."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(asked))
            return int(s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)), None
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def probe_raw_socket() -> bool:
    """Whether a raw IP socket opens, as the stock emulator's data socket does (hsb.raw_sockets_permitted, without
    its HSB_EMULATOR_UDP_SOCKET shortcut). Closed at once."""
    try:
        socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_RAW).close()
        return True
    except OSError:
        return False


PROBE = r"""
import ctypes, json, sys
out = {}
def tryit(key, fn):
    try:
        fn(); out[key] = None
    except BaseException as exc:
        out[key] = f"{type(exc).__name__}: {exc}"[:300]
def hololink():
    import hololink, hololink.emulation
    hololink.operators.LinuxReceiverOp
def roce():
    import hololink
    hololink.operators.RoceReceiverOp
tryit("holoscan", lambda: __import__("holoscan"))
tryit("hololink", hololink)
tryit("roce_receiver", roce)
sys.path.insert(0, sys.argv[1])
tryit("gpusonar", lambda: __import__("gpusonar.sources.hsb"))
try:
    lib = ctypes.CDLL("libcuda.so.1")
    n = ctypes.c_int()
    rc = lib.cuInit(0)
    out["cuda_devices"] = n.value if rc == 0 and lib.cuDeviceGetCount(ctypes.byref(n)) == 0 else 0
    if rc != 0:
        out["cuda_error"] = f"cuInit returned {rc}"
except OSError as exc:
    out["cuda_devices"] = 0
    out["cuda_error"] = f"no libcuda.so.1 ({exc})"[:300]
print("PROBE " + json.dumps(out))
"""


def probe_software(timeout: float = 180.0) -> dict:
    """Imports and the CUDA device count, in a child process (an import that crashes cannot take this one down)."""
    try:
        p = subprocess.run([sys.executable, "-c", PROBE, str(ROOT / "python")], capture_output=True, text=True, timeout=timeout)
        for line in p.stdout.splitlines():
            if line.startswith("PROBE "):
                return json.loads(line[6:])
        return {"probe_error": f"exit {p.returncode}: {p.stderr[-300:]}"}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"probe_error": f"{type(exc).__name__}: {exc}"}


def cap_eff() -> int | None:
    for line in (read_text("/proc/self/status") or "").splitlines():
        if line.startswith("CapEff:"):
            return int(line.split()[1], 16)
    return None


def read_host(asked_buffer: int, software: bool = True) -> dict:
    """Every reading assess() takes, from this host."""
    import resource
    granted, err = probe_socket_buffer(asked_buffer)
    try:
        rt = list(resource.getrlimit(resource.RLIMIT_RTPRIO))
    except (AttributeError, OSError, ValueError):
        rt = None
    ib = Path("/sys/class/infiniband")
    return {"rmem_max": read_int("/proc/sys/net/core/rmem_max"), "rmem_default": read_int("/proc/sys/net/core/rmem_default"),
            "socket_buffer_granted": granted, "socket_buffer_error": err,
            "isolated": read_text("/sys/devices/system/cpu/isolated"), "online": read_text("/sys/devices/system/cpu/online"),
            "affinity": sorted(os.sched_getaffinity(0)), "cpu_max": read_text("/sys/fs/cgroup/cpu.max"),
            "cfs_quota": read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), "cfs_period": read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
            "cpuset": read_text("/sys/fs/cgroup/cpuset.cpus.effective") or read_text("/sys/fs/cgroup/cpuset/cpuset.effective_cpus"),
            "cmdline": read_text("/proc/cmdline"), "hololink_affinity": os.environ.get("HOLOLINK_AFFINITY"),
            "rtprio_limit": rt, "cap_eff": cap_eff(), "euid": os.geteuid(),
            "raw_socket": probe_raw_socket(), "udp_patch_env": os.environ.get("HSB_EMULATOR_UDP_SOCKET") == "1",
            "software": probe_software() if software else {},
            "rdma_devices": sorted(os.listdir(ib)) if ib.is_dir() else []}


def load_rates(builds_file=BUILDS_FILE) -> list[dict]:
    builds = json.loads(Path(builds_file).read_text())["builds"]
    try:
        sys.path.insert(0, str(ROOT / "python"))
        from gpusonar.sources import hsb
        return stream_rates(builds, hsb.frame_bytes, hsb.FRAME_SAMPLES)
    except Exception:  # noqa: BLE001  (gpusonar not built here: the samples alone, said in the report)
        return stream_rates(builds)
    finally:
        sys.path.pop(0)


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text}: at least 1")
    return value


def core_number(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"{text}: a CPU number, 0 or more")
    return value


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--socket-buffer", type=positive_int, default=GUIDE_RMEM_MAX, metavar="BYTES",
                    help=f"what receive --socket-buffer will ask for (default {GUIDE_RMEM_MAX}, the guide's rmem_max)")
    ap.add_argument("--receiver-core", type=core_number, metavar="K",
                    help="the core receive --receiver-core will name (default: the first isolated CPU)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON only")
    ap.add_argument("--json-out", metavar="FILE", help="also write the report as JSON to FILE")
    ap.add_argument("--builds", default=str(BUILDS_FILE), help="the builds' file (default docs/handoff/hardware_builds.json)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    readings = read_host(args.socket_buffer)
    report = assess(readings, load_rates(args.builds), args.socket_buffer, args.receiver_core)
    report["readings"] = readings
    text = json.dumps(report, indent=1)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    print(text if args.json else render(report))
    return exit_status(report)


if __name__ == "__main__":
    sys.exit(main())
