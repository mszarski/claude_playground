#!/usr/bin/env python3
"""Time every stage that has a CUDA backend on the host and on the device, in one table.

Each case is run on the host backend and, when the build has CUDA and a device is present,
on the CUDA backend: one warm-up run (not counted), then the best of N runs. The table gives
the sizes, both times, the speed-up, and the relative rms difference between the host and
the CUDA result (record samples, image, beam map or spectrum), so a fast but wrong device
path shows. On a host-only build the CUDA column reads "unavailable". A case that raises is
recorded with its error text and the run goes on.

Groups (`--only a,b`):

  synthesis     the echo synthesis kernel (`core.synthesize_into`): the calls a scenario makes
                are captured once and replayed on each backend, so path geometry is not timed
  reverb        the reverberation generator (`sim/reverb.py: add_reverberation`), replayed the
                same way; the whole generator, and inside it the kernel (transforms, Doppler
                shifts, steering and summation: `core.sim_reverb_accumulate`) and the
                interpolation to the record's rate (`core.sim_reverb_finish`)
  scenarios     whole scenarios through `run_scenario(..., backend=...)`, with the time spent
                in each stage (which of them the device runs, which stay on the host)
  imaging       backprojection: the datasets of tools/benchmark.py (sas, fls, flank)
  conditioning  baseband conversion and pulse compression, incl. a sampled-pulse replica and
                a bank of Doppler hypotheses
  detect        CFAR; sparse Bayesian bearings (`core.doa_gate`, `core.doa_passive`)
  live          the long-range chain per ping: Python and C++ engines, bearings from the map
                and by SBL, backprojection / far-field / MVDR beamformers, one row and (when
                the stacked-row sonar configs and the elevation chain exist) two and three;
                and the far-field and MVDR beamformers alone on one ping
  hostonly      stages with no CUDA build, with their host time

All calls into the library are synchronous (results come back as host arrays), so there is no
device work outstanding when the clock stops; `core.cuda_synchronize` is called as well if
a build has it.

Usage: tools/gpu_benchmark_all.py [--quick] [--only groups] [--scale 1.0] [--repeats 3] [--budget 180]
                                  [--out table.md] [--json result.json] [--backend auto|host|cuda]
"""

import argparse
import copy
import importlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import gpusonar as gs  # noqa: E402
from gpusonar import recon  # noqa: E402
from gpusonar.config import deep_merge, list_configs, load_scenario_config, load_yaml  # noqa: E402
from gpusonar.sim import Scene, find_bellhop, make_pose, run_scenario, simulate_ping  # noqa: E402
from gpusonar.sonar import SonarModel  # noqa: E402

core = gs.core
simulator = importlib.import_module("gpusonar.sim.simulator")
reverb = importlib.import_module("gpusonar.sim.reverb")
reverb_bistatic = importlib.import_module("gpusonar.sim.reverb_bistatic")
reverb_relief = importlib.import_module("gpusonar.sim.reverb_relief")
propagation = importlib.import_module("gpusonar.sim.propagation")
timespread = importlib.import_module("gpusonar.sim.timespread")
bellhop = importlib.import_module("gpusonar.sim.bellhop")
noise_models = importlib.import_module("gpusonar.sim.noise")
receiver = importlib.import_module("gpusonar.sim.receiver")
scene_mod = importlib.import_module("gpusonar.sim.scene")

GROUPS = ("synthesis", "reverb", "scenarios", "imaging", "conditioning", "detect", "live", "hostonly")
HOST, CUDA = core.Backend.Host, core.Backend.Cuda


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# --- timing ---------------------------------------------------------------------------------

def device_sync():
    fn = getattr(core, "cuda_synchronize", None)
    if fn is not None:
        fn()


class Timed:
    """A result whose time the case measured itself (e.g. seconds per ping)."""

    def __init__(self, seconds, out):
        self.seconds, self.out = float(seconds), out


class StageTimer:
    """Wall time spent inside chosen functions, exclusive of the chosen functions they call."""

    def __init__(self):
        self.seconds, self.counts, self._stack, self._patched = {}, {}, [], []

    def patch(self, owner, attr, label, count=None):
        if attr not in vars(owner):
            return
        orig = vars(owner)[attr]
        timer = self

        def wrapper(*a, **k):
            timer._stack.append(0.0)
            t0 = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                dt = time.perf_counter() - t0
                inner = timer._stack.pop()
                timer.seconds[label] = timer.seconds.get(label, 0.0) + dt - inner
                if timer._stack:
                    timer._stack[-1] += dt
                if count is not None:
                    try:
                        for key, value in count(a, k).items():
                            timer.counts[key] = timer.counts.get(key, 0) + value
                    except Exception:
                        pass

        # The library looks at the kernels' docstrings to learn what a build can do (reverb.cpp_has_components).
        wrapper.__doc__ = getattr(orig, "__doc__", None)
        wrapper.__name__ = getattr(orig, "__name__", attr)
        setattr(owner, attr, wrapper)
        self._patched.append((owner, attr, orig))

    def reset(self):
        self.seconds, self.counts, self._stack = {}, {}, []

    def snapshot(self):
        return dict(self.seconds), dict(self.counts)

    def restore(self):
        for owner, attr, orig in reversed(self._patched):
            setattr(owner, attr, orig)
        self._patched = []


# Labels of the stages timed inside a scenario; the first two are the ones with a CUDA build.
S_SYNTH = "echo synthesis kernel"
S_RKERNEL = "reverberation steering/summation kernel"
S_PAIR = "path pairing"
S_EIGEN = "eigenrays (free field, image source)"
S_BELLHOP = "eigenrays (Bellhop, external program)"
S_SPREAD = "time-spread taps"
S_SCENE = "scene modifiers (aspect, glints, seabed models)"
S_BUILD = "scene construction"
S_RPOWER = "reverberation power (strength, propagation, Doppler bins)"
S_RDRAW = "reverberation random draws and texture"
S_RFINISH = "reverberation interpolation to the record's rate"
S_RSETUP = "reverberation set-up (grids, pulse, patterns)"
S_NOISE = "ambient noise with a spectrum (real passband)"
S_RX = "receiver (reference, gain, ADC)"
S_OTHER = "other (records, baseband noise draws, bookkeeping)"
CUDA_STAGES = (S_SYNTH, S_RKERNEL, S_RFINISH)
HOST_STAGES = (S_PAIR, S_EIGEN, S_BELLHOP, S_SPREAD, S_SCENE, S_BUILD, S_RPOWER, S_RDRAW, S_RSETUP,
               S_NOISE, S_RX, S_OTHER)


def simulator_timer():
    """A StageTimer on every simulator stage that can be reached without changing the code."""
    t = StageTimer()
    t.patch(core, "synthesize_into", S_SYNTH, lambda a, k: {"paths": len(a[1]), "synth_calls": 1})
    t.patch(core, "sim_reverb_accumulate", S_RKERNEL,
            lambda a, k: {"reverb_rows": int(np.shape(a[4])[0]), "reverb_kernel_calls": 1})
    t.patch(core, "sim_reverb_finish", S_RFINISH)
    t.patch(core, "sim_two_way_paths", S_PAIR)
    t.patch(simulator, "two_way_paths", S_PAIR)
    t.patch(simulator, "build_scene", S_BUILD)
    t.patch(scene_mod.Scene, "evaluate", S_SCENE)
    t.patch(scene_mod.Scene, "visible", S_SCENE)
    t.patch(propagation.FreeFieldProvider, "eigenrays", S_EIGEN)
    t.patch(propagation.ImageSourceProvider, "eigenrays", S_EIGEN)
    if hasattr(propagation, "SedimentProvider"):
        t.patch(propagation.SedimentProvider, "eigenrays", S_EIGEN)
    t.patch(bellhop.BellhopProvider, "eigenrays", S_BELLHOP)
    t.patch(timespread.TimeSpreadProvider, "eigenrays", S_SPREAD)
    t.patch(reverb, "add_reverberation", S_RSETUP)
    t.patch(reverb, "_series_cpp", S_RDRAW)
    t.patch(reverb, "_series_numpy", S_RDRAW)
    t.patch(reverb, "reverberation_power", S_RPOWER)
    t.patch(reverb_bistatic, "directional_power", S_RPOWER)
    t.patch(reverb_relief, "directional_power", S_RPOWER)
    t.patch(noise_models, "gaussian_noise", S_NOISE)
    for name in ("apply_adc", "apply_hardware_tvg", "add_reference_signal", "apply_channel_errors", "add_noise"):
        t.patch(receiver, name, S_RX)
    return t


def measure(run, prep, repeats, budget, timer=None, warmup=True):
    """Warm-up (not counted), then the best of `repeats` runs. A case whose warm-up shows that
    `repeats` runs would take longer than `budget` seconds is run fewer times; one that alone
    takes longer than the budget is not repeated and its only run is the result. With
    `warmup=False` the case leaves its own warm-up out of the time it reports (the live chain:
    the first ping of every run), so every run counts."""
    def once():
        args = prep() if prep is not None else None
        if timer is not None:
            timer.reset()
        device_sync()
        t0 = time.perf_counter()
        out = run(args)
        device_sync()
        dt = time.perf_counter() - t0
        if isinstance(out, Timed):
            dt, out = out.seconds, out.out
        return dt, out, (timer.snapshot() if timer is not None else ({}, {}))

    wall = time.perf_counter()
    first = once()
    first_wall = max(time.perf_counter() - wall, 1e-9)
    affordable = repeats if first_wall * repeats <= budget else int(budget // first_wall)
    if warmup and affordable > 0:
        results, more, counted = [], affordable, False
    else:
        results, more, counted = [first], max(affordable - 1, 0), warmup
    for _ in range(more):
        results.append(once())
    best = min(results, key=lambda r: r[0])
    return {"seconds": best[0], "runs": [r[0] for r in results], "warmup_seconds": first[0] if warmup and not counted else None,
            "warmup_counted": counted, "stages": best[2][0], "counts": best[2][1]}, best[1]


def arrays_of(out):
    """The numbers of a result as a list of arrays: records give their samples."""
    if isinstance(out, np.ndarray):
        return [out]
    if hasattr(out, "samples"):
        return [np.asarray(out.samples.array)]
    if isinstance(out, (list, tuple)):
        return [a for item in out for a in arrays_of(item)]
    return [np.asarray(out)]


def relative_rms(a, b):
    """sqrt(sum |a - b|^2 / sum |a|^2) over two lists of arrays."""
    a, b = arrays_of(a), arrays_of(b)
    if len(a) != len(b) or any(x.shape != y.shape for x, y in zip(a, b)):
        raise ValueError(f"host and CUDA results differ in shape: {[x.shape for x in a][:3]} against {[y.shape for y in b][:3]}")
    num = sum(float(np.sum(np.abs(np.asarray(x, dtype=complex) - np.asarray(y, dtype=complex)) ** 2)) for x, y in zip(a, b))
    den = sum(float(np.sum(np.abs(np.asarray(x, dtype=complex)) ** 2)) for x in a)
    if not np.isfinite(num):
        return float("nan")
    return float(np.sqrt(num / den)) if den > 0.0 else (0.0 if num == 0.0 else float("inf"))


class Bench:
    def __init__(self, args):
        self.args = args
        self.quick = args.quick
        self.scale = args.scale
        self.repeats = args.repeats
        self.budget = args.budget
        self.cuda = args.backend == "cuda" or (args.backend == "auto" and bool(core.cuda_available()))
        self.cases, self.host_only, self.breakdown, self.skipped = [], [], [], []
        self.group = ""

    # -- one case --------------------------------------------------------------------------

    def case(self, name, what, sizes, run, prep=None, compare=None, timer=None, note="", keep=False, warmup=True):
        """Time `run(prep(backend), backend)` on the host and on the device. `sizes` is a dict
        (or a function of the host result giving one). Returns the row; with `keep` the host
        result is in row["_host_out"]."""
        row = {"group": self.group, "case": name, "what": what, "sizes": {}, "host_s": None, "cuda_s": None,
               "speedup": None, "agreement": None, "note": note, "host_error": None,
               "cuda_error": None if self.cuda else "unavailable", "host_runs": [], "cuda_runs": []}
        self.cases.append(row)
        log(f"[{self.group}] {name} ...")
        outs = {}
        for label, backend in (("host", HOST), ("cuda", CUDA)):
            if label == "cuda" and not self.cuda:
                continue
            try:
                m, out = measure(lambda a, be=backend: run(a, be), (lambda be=backend: prep(be)) if prep else None,
                                 self.repeats, self.budget, timer, warmup)
                row[f"{label}_s"], row[f"{label}_runs"] = m["seconds"], m["runs"]
                row[f"{label}_warmup_s"], row[f"{label}_warmup_counted"] = m["warmup_seconds"], m["warmup_counted"]
                row[f"{label}_stages"], row[f"{label}_counts"] = m["stages"], m["counts"]
                outs[label] = out
            except Exception as e:  # noqa: BLE001 - one failing case must not end the run
                row[f"{label}_error"] = f"{type(e).__name__}: {e}".replace("\n", " ")[:400]
                log("    " + label + " FAILED: " + row[f"{label}_error"])
                if self.args.traceback:
                    traceback.print_exc()
        try:
            if "host" in outs:
                row["sizes"] = sizes(outs["host"], row) if callable(sizes) else dict(sizes)
            elif not callable(sizes):
                row["sizes"] = dict(sizes)
            if "host" in outs and "cuda" in outs:
                if compare is not None:
                    row["agreement"], extra = compare(outs["host"], outs["cuda"])
                    if extra:
                        row["note"] = (row["note"] + "; " if row["note"] else "") + extra
                else:
                    row["agreement"] = relative_rms(outs["host"], outs["cuda"])
                if row["host_s"] and row["cuda_s"]:
                    row["speedup"] = row["host_s"] / row["cuda_s"]
        except Exception as e:  # noqa: BLE001
            row["cuda_error"] = f"comparison failed: {type(e).__name__}: {e}"[:400]
        if keep and "host" in outs:
            row["_host_out"] = outs["host"]
        runs = len(row["host_runs"])
        log(f"    host {fmt_time(row['host_s'])} s ({runs} run{'s' if runs != 1 else ''}"
            f"{', warm-up counted' if row.get('host_warmup_counted') else ''})   "
            f"cuda {cuda_cell(row)}   speed-up {fmt_ratio(row['speedup'])}   rel. rms {fmt_agreement(row['agreement'])}"
            f"   {fmt_sizes(row['sizes'])}")
        return row

    def host_case(self, name, what, sizes, run, prep=None, note=""):
        """A stage with no CUDA build: host time only."""
        row = {"group": self.group, "case": name, "what": what, "sizes": dict(sizes), "host_s": None, "note": note,
               "host_error": None, "host_runs": []}
        self.host_only.append(row)
        log(f"[{self.group}] {name} (host only) ...")
        try:
            m, _ = measure(lambda a: run(a), prep, self.repeats, self.budget)
            row["host_s"], row["host_runs"] = m["seconds"], m["runs"]
        except Exception as e:  # noqa: BLE001
            row["host_error"] = f"{type(e).__name__}: {e}".replace("\n", " ")[:400]
            if self.args.traceback:
                traceback.print_exc()
        log(f"    host {fmt_time(row['host_s'])} s" + (f"   FAILED: {row['host_error']}" if row["host_error"] else ""))
        return row

    def host_row(self, name, what, sizes, seconds, note=""):
        self.host_only.append({"group": self.group, "case": name, "what": what, "sizes": dict(sizes), "host_s": seconds,
                               "note": note, "host_error": None, "host_runs": [seconds]})

    def skip(self, name, why):
        self.skipped.append({"group": self.group, "case": name, "reason": why})
        log(f"[{self.group}] {name}: skipped ({why})")

    def guard(self, name, fn):
        """Run a case builder; a failure while it sets up is recorded like a failing case."""
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}".replace("\n", " ")[:400]
            self.cases.append({"group": self.group, "case": name, "what": "", "sizes": {}, "host_s": None, "cuda_s": None,
                               "speedup": None, "agreement": None, "note": "set-up failed", "host_error": msg,
                               "cuda_error": msg if self.cuda else "unavailable", "host_runs": [], "cuda_runs": []})
            log(f"[{self.group}] {name}: set-up FAILED: {msg}")
            if self.args.traceback:
                traceback.print_exc()


# --- formatting -----------------------------------------------------------------------------

def fmt_time(t):
    if t is None:
        return "-"
    if t >= 100.0:
        return f"{t:.0f}"
    if t >= 1.0:
        return f"{t:.2f}"
    if t >= 1e-3:
        return f"{t:.4f}"
    return f"{t:.2e}"


def fmt_ratio(r):
    return "-" if r is None else (f"{r:.1f}x" if r >= 10 else f"{r:.2f}x")


def fmt_agreement(a):
    return "-" if a is None else ("0" if a == 0.0 else f"{a:.1e}")


def fmt_sizes(sizes):
    def one(v):
        if isinstance(v, float):
            return f"{v:.3g}"
        return f"{v:,}".replace(",", " ") if isinstance(v, int) else str(v)
    return ", ".join(f"{k} {one(v)}" for k, v in sizes.items())


def cuda_cell(row):
    if row.get("cuda_s") is not None and not row.get("cuda_error"):
        return fmt_time(row["cuda_s"])
    err = row.get("cuda_error") or "-"
    return err if err == "unavailable" else "error: " + err


def md_escape(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


# --- scenario helpers -----------------------------------------------------------------------

# The quick run: scenarios whose every path costs a second of pulse on 32 channels get a
# shorter pulse and fewer hydrophones (sizes are in the table either way).
QUICK_OVERRIDES = {
    "flank_sst_shallow": {"sonar_overrides": {"array": {"count": 8}, "transmit": {"waveform": {"duration": 0.1}}}},
    "flank_sst_spread": {"sonar_overrides": {"array": {"count": 8}, "transmit": {"waveform": {"duration": 0.1}}}},
    # Bellhop scenarios keep fewer arrivals per leg (paths per scatterer go as its square).
    "popoto_12km": {"propagation": {"max_arrivals": 8}},
    "popoto_far_mover": {"propagation": {"max_arrivals": 8}},
}


def scenario_config(name, pings=None, density=1.0, overrides=None, quick=False):
    """A shipped scenario as a dict, with the number of pings and the scatterer density changed."""
    if quick and name in QUICK_OVERRIDES:
        overrides = deep_merge(copy.deepcopy(QUICK_OVERRIDES[name]), overrides or {})
    cfg = copy.deepcopy(load_scenario_config(name, overrides))
    if pings is not None:
        plat = cfg.get("platform", {})
        if plat.get("legs"):
            plat["legs"] = [dict(plat["legs"][0], pings=int(pings))]
            plat["legs"][0].pop("length", None)
        else:
            cfg.setdefault("pings", {})["count"] = int(pings)
    if density != 1.0:
        for item in cfg.get("scene") or []:
            if "density" in item:
                item["density"] = float(item["density"]) * density
            if "count" in item:
                item["count"] = max(int(round(int(item["count"]) * density)), 1)
    return cfg


def needs_bellhop(cfg):
    return cfg.get("fidelity") == "eigenray"


def without_bellhop(cfg):
    """A Bellhop scenario with the image-source tier in its place: the same water depth and
    bottom, straight rays, two bounces. A stand-in so that the case runs anywhere."""
    cfg = copy.deepcopy(cfg)
    env = (cfg.get("propagation") or {}).get("environment") or {}
    cfg["fidelity"] = "image_source"
    cfg["propagation"] = {"surface_z": 0.0, "bottom_z": float(env.get("depth", 100.0)), "max_order": 2,
                          "bottom": {"speed": float(env.get("bottom_speed", 1650.0)), "density": float(env.get("bottom_density", 1.9)),
                                     "attenuation": float(env.get("bottom_attenuation", 0.8))}}
    return cfg


def echoes_only(cfg):
    cfg = copy.deepcopy(cfg)
    cfg.pop("reverberation", None)
    rx = dict(cfg.get("receiver") or {})
    rx.pop("ambient_noise", None)
    rx.update(noise_rms=0.0, adc=False)
    cfg["receiver"] = rx
    return cfg


def reverberation_only(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["scene"] = []
    rx = dict(cfg.get("receiver") or {})
    rx.pop("ambient_noise", None)
    rx.update(noise_rms=0.0, adc=False)
    cfg["receiver"] = rx
    return cfg


def sampled_waveform(cfg, source):
    """The scenario with its transmit pulse replaced by a sampled one in the same band and of
    the same length (`type: sampled`): an m-sequence or a sampled LFM."""
    cfg = copy.deepcopy(cfg)
    w = SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides")).waveform_config
    f0, f1, duration = float(w["f_start"]), float(w.get("f_stop", w["f_start"])), float(w["duration"])
    wave = {"type": "sampled", "source": source, "f_start": f0, "f_stop": f1, "duration": duration}
    if source == "mseq":
        degree = min(int(np.floor(np.log2(abs(f1 - f0) * duration + 1.0))), 10)
        if degree < 2:
            raise ValueError("the band is too narrow for an m-sequence of this length")
        wave["degree"] = degree
    else:
        wave["window_alpha"] = float(w.get("window_alpha", 0.1))
    cfg["sonar_overrides"] = deep_merge(cfg.get("sonar_overrides") or {}, {"transmit": {"waveform": wave}})
    return cfg


def capture_synthesis(cfg):
    """Run the scenario's path geometry on the host and keep every call it makes of the
    synthesis kernel: the record as it was before the call, and the paths."""
    calls, orig = [], core.synthesize_into

    def wrapper(record, *a, **k):
        calls.append((record.copy(), a, dict(k)))            # the kernel itself is not run here: only its inputs are wanted

    wrapper.__doc__ = orig.__doc__
    core.synthesize_into = wrapper
    try:
        result = run_scenario(echoes_only(cfg))
    finally:
        core.synthesize_into = orig
    if not calls:
        raise RuntimeError("the scenario made no call of the C++ synthesis kernel (no scatterers, or the NumPy engine ran)")
    return calls, result


def omnidirectional(record):
    """The record with every beam pattern removed (NaN beamwidth: no pattern)."""
    r = record.copy()
    geo = r.geometry
    elements = geo.rx
    for e in elements:
        e.beamwidth_az = e.beamwidth_el = float("nan")
    geo.rx = elements
    tx = geo.tx
    tx.beamwidth_az = tx.beamwidth_el = float("nan")
    geo.tx = tx
    r.geometry = geo
    return r


def replay_synthesis(calls, plain=False):
    """(prep, run) replaying captured kernel calls. `plain`: no beam patterns, no absorption."""
    kernel = core.synthesize_into
    base = [(omnidirectional(r) if plain else r, a, dict(k, absorption=None) if plain else k) for r, a, k in calls]

    def prep(backend):
        return [r.copy() for r, _, _ in base]

    def run(records, backend):
        for rec, (_, a, k) in zip(records, base):
            kernel(rec, *a, **dict(k, backend=backend))
        return records

    return prep, run


def synthesis_sizes(calls, result):
    rec = calls[0][0]
    paths = [len(a[0]) for _, a, _ in calls]
    return {"pings": len(calls), "channels": int(rec.samples.n_channels), "samples": int(rec.samples.n_samples),
            "scatterers": len(result.scene), "paths/ping": int(round(float(np.mean(paths))))}


# --- groups ---------------------------------------------------------------------------------

def group_synthesis(b):
    q, s = b.quick, b.scale

    def pings(quick, full):
        return quick if q else max(int(round(full * s)), 1)

    dens = 0.1 if q else s
    # (case, what, scenario, pings, scatterer density factor, config change, also without patterns/absorption)
    table = [
        ("fls_120k_transit", "FLS, analytic LFM, moving hull, image-source bounces", "fls_120k_transit", pings(1, 4), 1.0 if not q else 0.3, None, False),
        ("fls_glints_on_sediment", "FLS, aspect-dependent glints on a sediment scatterer seabed", "fls_glints_on_sediment", pings(1, 4), dens, None, False),
        ("point_grid_fls_300k", "FLS, analytic pulse, point grid", "point_grid_fls_300k", pings(1, 1), 1.0, None, False),
        ("minefield_sas", "SAS, analytic LFM, scatterer seafloor and objects", "minefield_sas", pings(2, 20), dens, None, False),
        ("rail_cylinder_glint", "rail SAS, aspect-dependent glint of a cylinder", "rail_cylinder_glint", pings(3, 30), 1.0, None, False),
        ("flank_sst_shallow", "flank array, analytic LFM, time spreading (taps on bounce paths)", "flank_sst_shallow", pings(1, 4), 1.0, None, False),
        ("flank_sst_spread", "flank array, time spreading with doppler: true and angle spread", "flank_sst_spread", pings(1, 1), 1.0, None, False),
        ("sst_pursuit (CW)", "analytic CW, moving sonar and moving three-highlight target", "sst_pursuit", pings(1, 4), 1.0, None, False),
        ("flank, sampled m-sequence", "sampled pulse (type: sampled, mseq), time spreading", "flank_sst_shallow", pings(1, 1), 1.0, "mseq", True),
        ("flank, sampled LFM", "sampled pulse (type: sampled, lfm), time spreading", "flank_sst_shallow", pings(1, 1), 1.0, "lfm", True),
        ("FLS, sampled m-sequence", "sampled pulse (mseq), moving hull, image-source bounces", "fls_120k_transit", pings(1, 2), 1.0 if not q else 0.1, "mseq", True),
        ("FLS, sampled LFM", "sampled pulse (lfm), moving hull, image-source bounces", "fls_120k_transit", pings(1, 2), 1.0 if not q else 0.1, "lfm", False),
    ]
    for name, what, scenario, n, density, source, plain in table:
        def build(name=name, what=what, scenario=scenario, n=n, density=density, source=source, plain=plain):
            cfg = scenario_config(scenario, n, density, quick=q)
            if needs_bellhop(cfg) and not find_bellhop():
                return b.skip(name, "needs Bellhop (GPUSONAR_BELLHOP)")
            if source:
                cfg = sampled_waveform(cfg, source)
            calls, result = capture_synthesis(cfg)
            sizes = synthesis_sizes(calls, result)
            prep, run = replay_synthesis(calls)
            b.case(name, what, sizes, run, prep, note=f"scenario {scenario}; kernel only (paths prepared once on the host)")
            if plain:
                prep, run = replay_synthesis(calls, plain=True)
                b.case(name + ", no patterns/absorption", what + "; beam patterns and absorption off", sizes, run, prep,
                       note=f"scenario {scenario}; kernel only")
        b.guard(name, build)


def coarse_bistatic(item, n_azimuth=24, n_elevation=18):
    """A bistatic reverberation item with at most that many azimuth sectors and elevation bins."""
    if item.get("geometry") == "bistatic":
        item = dict(item)
        item["n_azimuth"] = min(int(item.get("n_azimuth") or n_azimuth), n_azimuth)
        item["n_elevation"] = min(int(item.get("n_elevation") or 9), n_elevation)
    return item


def capture_reverberation(cfg):
    """Run the scenario's reverberation on the host and keep every call of the generator with
    the record and the random generator as they were before it."""
    calls, orig = [], reverb.add_reverberation

    def wrapper(record, provider, boundary_z, mu_db, rng, *a, **k):
        calls.append((record.copy(), provider, boundary_z, mu_db, copy.deepcopy(rng), a, dict(k)))
        return orig(record, provider, boundary_z, mu_db, rng, *a, **k)

    reverb.add_reverberation = wrapper
    try:
        run_scenario(reverberation_only(cfg))
    finally:
        reverb.add_reverberation = orig
    if not calls:
        raise RuntimeError("the scenario has no reverberation")
    return calls


def group_reverb(b):
    q = b.quick
    n_pings = 1 if q else max(int(round(2 * b.scale)), 1)

    def items_of(name, change=None, keep=None, overrides=None):
        cfg = scenario_config(name, n_pings, overrides=overrides, quick=q)
        items = [dict(i) for i in cfg.get("reverberation") or []]
        if keep is not None:
            items = [items[i] for i in keep]
        if change is not None:
            items = [change(dict(i)) for i in items]
        cfg["reverberation"] = items
        return cfg

    def coarse(item):
        # The quick run: fewer sectors and elevation bins in the bistatic mode.
        return coarse_bistatic(item) if q else item

    def moving(item):
        # Moving scatterers multiply the components by the Doppler lines of each direction: the
        # direction grid is thinned (36 x 30; quick 8 x 6) so that the host run stays in minutes.
        item = coarse_bistatic(item, 8, 6) if q else coarse_bistatic(item, 36, 30)
        return dict(item, scatterer_motion={"velocity": [0.2, 0.1, 0.0], "wind_speed": 8.0})

    def bathymetry_item(item):
        depth = float(item.get("boundary_z", 100.0))
        out = {"mu_db": -27.0, "boundary_z": depth, "geometry": "bistatic",
               "bathymetry": {"plane": {"depth": depth, "slope": [0.01, 0.02], "point": [0.0, 0.0]}}}
        return coarse(out)

    table = [
        ("monostatic sectors, flank (sediment + surface + volume)", "flank_sst_shallow", lambda: items_of("flank_sst_shallow")),
        ("monostatic sectors, moving FLS (bottom + surface)", "fls_120k_transit", lambda: items_of("fls_120k_transit")),
        ("monostatic, fluid-sediment bottom", "sst_sediment_reverb", lambda: items_of("sst_sediment_reverb")),
        ("K-distributed (statistics: k), flank", "flank_sst_shallow",
         lambda: items_of("flank_sst_shallow", lambda i: dict(i, statistics={"model": "k", "shape": 1.5}), keep=[1])),
        ("bistatic, moving platforms (Doppler-binned components)", "sst_pursuit", lambda: items_of("sst_pursuit", coarse)),
        ("bistatic with scatterer_motion (drift and wind sea)", "sst_pursuit", lambda: items_of("sst_pursuit", moving)),
        ("bistatic over bathymetry (sloping plane)", "flank_sst_shallow",
         lambda: items_of("flank_sst_shallow", bathymetry_item, keep=[0])),
    ]
    for name, scenario, make in table:
        def build(name=name, scenario=scenario, make=make):
            calls = capture_reverberation(make())
            timer = StageTimer()
            timer.patch(core, "sim_reverb_accumulate", S_RKERNEL,
                        lambda a, k: {"rows": int(np.shape(a[4])[0]), "n_low": int(np.shape(a[0])[1]), "calls": 1})
            timer.patch(core, "sim_reverb_finish", S_RFINISH)
            timer.patch(reverb, "_series_cpp", S_RDRAW)
            timer.patch(reverb, "reverberation_power", S_RPOWER)
            timer.patch(reverb_bistatic, "directional_power", S_RPOWER)
            timer.patch(reverb_relief, "directional_power", S_RPOWER)
            generator = reverb.add_reverberation

            def prep(backend):
                return [(r.copy(), copy.deepcopy(rng)) for r, _, _, _, rng, _, _ in calls]

            profiles = {}

            def run(state, backend):
                out = []
                if hasattr(core, "sim_reverb_profile"):
                    core.sim_reverb_profile(reset=True)
                for (rec, rng), (_, provider, z, mu, _, a, k) in zip(state, calls):
                    generator(rec, provider, z, mu, rng, *a, **dict(k, backend=backend))
                    out.append(rec)
                if hasattr(core, "sim_reverb_profile"):
                    profiles["cuda" if backend == CUDA else "host"] = core.sim_reverb_profile(reset=True)
                return out

            def sizes(out, row):
                c = row.get("host_counts") or {}
                rec = calls[0][0]
                k = max(c.get("calls", 0), 1)
                return {"calls": len(calls), "channels": int(rec.samples.n_channels), "samples": int(rec.samples.n_samples),
                        "components": int(c.get("rows", 0)), "low-rate samples": int(c.get("n_low", 0) // k)}

            try:
                row = b.case(name, "whole generator: power, random draws, kernel, interpolation", sizes, run, prep, timer=timer,
                             note=f"reverberation items of {scenario}")
            finally:
                timer.restore()
            # The kernel alone, from the same runs.
            hs, cs = row.get("host_stages") or {}, row.get("cuda_stages") or {}
            if row["host_s"] is not None and S_RKERNEL not in hs:
                row["note"] += "; the C++ kernel was NOT called: the NumPy generator ran (GPUSONAR_SIM_ENGINE, or a build whose C++ generator lacks components)"
            def steps(label, keys):
                # Where the time of the last run went inside the C++ generator (core.sim_reverb_profile).
                prof = profiles.get(label)
                if not prof:
                    return ""
                return f"{label} steps: " + ", ".join(f"{k} {fmt_time(prof[k])} s" for k in keys if prof.get(k, 0.0) > 0.0)
            detail = "; ".join(t for t in (steps("host", ("transform", "steer", "active")),
                                           steps("cuda", ("shape", "upload", "shift", "transform", "pulse", "product", "steer", "download"))) if t)
            kernel = {"group": b.group, "case": name + ": kernel", "what": "transforms, Doppler shifts, steering and summation (core.sim_reverb_accumulate) inside the run above",
                      "sizes": row["sizes"], "host_s": hs.get(S_RKERNEL), "cuda_s": cs.get(S_RKERNEL), "speedup": None,
                      "agreement": row["agreement"], "note": "agreement is the whole generator's (same random numbers on both)" + ("; " + detail if detail else ""),
                      "host_error": row["host_error"], "cuda_error": row["cuda_error"], "host_runs": [], "cuda_runs": []}
            if kernel["host_s"] and kernel["cuda_s"]:
                kernel["speedup"] = kernel["host_s"] / kernel["cuda_s"]
            b.cases.append(kernel)
            finish = {"group": b.group, "case": name + ": interpolation", "what": "interpolation to the record's rate, window, carrier (core.sim_reverb_finish) inside the run above",
                      "sizes": row["sizes"], "host_s": hs.get(S_RFINISH), "cuda_s": cs.get(S_RFINISH), "speedup": None,
                      "agreement": row["agreement"], "note": "agreement is the whole generator's",
                      "host_error": row["host_error"], "cuda_error": row["cuda_error"], "host_runs": [], "cuda_runs": []}
            if finish["host_s"] and finish["cuda_s"]:
                finish["speedup"] = finish["host_s"] / finish["cuda_s"]
            b.cases.append(finish)
            if row["host_s"] is not None:
                rest = row["host_s"] - hs.get(S_RKERNEL, 0.0) - hs.get(S_RFINISH, 0.0)
                parts = ", ".join(f"{label.split(' (')[0]} {fmt_time(hs[label])} s" for label in (S_RPOWER, S_RDRAW) if label in hs)
                if cs:
                    parts += "; in the CUDA run: " + ", ".join(f"{label.split(' (')[0]} {fmt_time(cs[label])} s" for label in (S_RPOWER, S_RDRAW) if label in cs)
                b.host_row("reverberation, " + name + ": power and random draws", "power computation, random draws, texture (NumPy on the host on both backends)",
                           row["sizes"], rest, note=parts)
                log(f"    kernel: host {fmt_time(kernel['host_s'])} s, cuda {cuda_cell(kernel)}; interpolation: host {fmt_time(finish['host_s'])} s, "
                    f"cuda {cuda_cell(finish)}; the rest on the host {fmt_time(rest)} s ({parts})")
                if detail:
                    log("    " + detail)
            if row["host_s"] is not None and getattr(core, "SIM_REVERB_ROWS", False) and not q:
                # The host kernel as it was called before `rows`: the sectors with power searched for in every call.
                state = prep(HOST)
                timer.reset()
                core.SIM_REVERB_ROWS = False
                try:
                    run(state, HOST)
                finally:
                    core.SIM_REVERB_ROWS = True
                before = timer.snapshot()
                seconds = (before[0] if isinstance(before, tuple) else before).get(S_RKERNEL)
                prof = profiles.get("host") or {}
                if seconds:
                    b.host_row("reverberation, " + name + ": kernel without `rows`",
                               "the host kernel called as before: every call searches the density for the sectors with power", row["sizes"],
                               seconds, note=f"one run; of which the search {fmt_time(prof.get('active', 0.0))} s; with `rows` {fmt_time(kernel['host_s'])} s")
                    log(f"    host kernel without rows: {fmt_time(seconds)} s (search {fmt_time(prof.get('active', 0.0))} s)")
        b.guard(name, build)


def group_scenarios(b):
    q = b.quick
    known = list_configs("scenarios")
    names = sorted(n for n in known if n.startswith("sst_"))
    names += [n for n in ("flank_sst_shallow", "flank_sst_spread", "popoto_elevation_free", "popoto_elevation_shallow",
                          "popoto_12km", "popoto_far_mover") if n in known]
    have_bellhop = bool(find_bellhop())
    totals = {}
    for name in names:
        def build(name=name):
            full = load_scenario_config(name)
            count = int((full.get("pings") or {}).get("count", 1))
            n = min(count, 2 if name.startswith("popoto_elevation") else 1) if q else max(int(round(count * b.scale)), 1)
            cfg = scenario_config(name, n, quick=q)
            if q and cfg.get("reverberation"):
                cfg["reverberation"] = [coarse_bistatic(i) for i in cfg["reverberation"]]
            variants = [(name, cfg)]
            if needs_bellhop(cfg):
                variants = [(name + " (image-source stand-in)", without_bellhop(cfg))]
                if have_bellhop and q and name != "popoto_12km":
                    b.skip(name + " (Bellhop)", "left out of the quick run (Bellhop alone takes most of a minute); the image-source stand-in is run")
                elif have_bellhop:
                    variants.insert(0, (name + " (Bellhop)", cfg))
                else:
                    b.skip(name + " (Bellhop)", "Bellhop is absent (GPUSONAR_BELLHOP); the image-source stand-in is run")
            for label, c in variants:
                timer = simulator_timer()

                def run(_, backend, c=c):
                    return run_scenario(copy.deepcopy(c), backend=backend).records

                def sizes(out, row):
                    counts = row.get("host_counts") or {}
                    d = {"pings": len(out), "channels": int(out[0].samples.n_channels), "samples": int(out[0].samples.n_samples)}
                    if counts.get("synth_calls"):
                        d["paths/ping"] = int(round(counts.get("paths", 0) / counts["synth_calls"]))
                    if counts.get("reverb_rows"):
                        d["reverb components"] = int(counts["reverb_rows"])
                    return d

                try:
                    row = b.case(label, "run_scenario end to end: echoes, reverberation, noise, receiver", sizes, run, timer=timer)
                finally:
                    timer.restore()
                for backend in ("host", "cuda"):
                    total, stages = row.get(f"{backend}_s"), row.get(f"{backend}_stages")
                    if total is None or stages is None:
                        continue
                    stages = dict(stages)
                    stages[S_OTHER] = max(total - sum(stages.values()), 0.0)
                    b.breakdown.append({"case": label, "backend": backend, "total_s": total, "stages": stages})
                    if backend == "host":
                        on_device = sum(stages.get(k, 0.0) for k in CUDA_STAGES)
                        row["note"] = (row["note"] + "; " if row["note"] else "") + \
                            f"{100.0 * on_device / max(total, 1e-12):.0f} % of the host time is in the stages with a CUDA build"
                        for k, v in stages.items():
                            totals[k] = totals.get(k, 0.0) + v
        b.guard(name, build)
    grand = sum(totals.values())
    for label in HOST_STAGES:
        if totals.get(label):
            b.host_row("simulator: " + label, "no CUDA build; sum over the whole-scenario cases (host runs)", {},
                       totals[label], note=f"{100.0 * totals[label] / grand:.1f} % of the host time of those cases")


def load_imaging_benchmark():
    spec = importlib.util.spec_from_file_location("gpusonar_imaging_benchmark", ROOT / "tools" / "benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def group_imaging(b):
    bm = load_imaging_benchmark()
    q = b.quick
    builders = {
        "sas": ("stripmap SAS backprojection (tools/benchmark.py sas)", lambda: bm.sas_dataset(0.1 if q else b.scale, 0.2 if q else 0.05)),
        "fls": ("forward-looking fan backprojection (tools/benchmark.py fls)", lambda: bm.fls_dataset(0.2 if q else b.scale)),
        "flank": ("flank array range x bearing map (tools/benchmark.py flank)", lambda: bm.flank_dataset(0.2 if q else b.scale)),
    }
    for mode, (what, make) in builders.items():
        def build(mode=mode, what=what, make=make):
            d = make()
            pts, kw = d["points"], d["kwargs"]
            flat = pts.reshape(-1, 3)
            recs = d["records"]
            sizes = {"pings": len(recs), "channels": int(recs[0].samples.n_channels), "samples": int(recs[0].samples.n_samples),
                     "pixels": int(flat.shape[0])}
            b.case(mode, what, sizes, lambda _, be: core.backproject(recs, flat, backend=be, **kw),
                   note=f"collection time {d['collection_time']:.3g} s")
            session = core.BackprojectSession(recs)
            b.case(mode + ", session", what + "; samples resident (BackprojectSession)", sizes,
                   lambda _, be: session.backproject(flat, backend=be, **kw))
            if mode == "sas":
                b.case(mode + ", culled", what + "; ping blocks outside the beam skipped (cull_block 16)", sizes,
                       lambda _, be: core.backproject(recs, flat, backend=be, cull_block=16, grid_shape=pts.shape[:2], **kw))
        b.guard(mode, build)


def raw_pings(sonar_name, overrides, targets, n, sound_speed=1500.0, advance=0.0):
    sonar = SonarModel.load(sonar_name, overrides)
    scene = Scene().add_points(targets)
    return [simulate_ping(sonar, make_pose(position=(i * advance, 0.0, 0.0)), scene, environment={"sound_speed": sound_speed},
                          absorption_model="none", adc=False, sequence=i) for i in range(n)]


def group_conditioning(b):
    q = b.quick

    def conditioning(name, what, raw):
        sizes = {"pings": len(raw), "channels": int(raw[0].samples.n_channels), "samples": int(raw[0].samples.n_samples),
                 "kind": "complex" if np.iscomplexobj(raw[0].samples.array) else "real"}
        copies = lambda be: [r.copy() for r in raw]  # noqa: E731
        based = None
        if not np.iscomplexobj(raw[0].samples.array):
            def to_baseband(recs, be):
                for r in recs:
                    core.to_baseband(r, backend=be)
                return recs
            b.case(name + ": to_baseband", what + "; real passband to complex baseband (FFT)", sizes, to_baseband, copies)
            based = [r.copy() for r in raw]
            for r in based:
                core.to_baseband(r)
        else:
            based = raw
        sizes_b = dict(sizes, samples=int(based[0].samples.n_samples), kind="complex")

        def compress(recs, be):
            for r in recs:
                core.pulse_compress(r, backend=be)
            return recs
        b.case(name + ": pulse_compress", what + "; matched filter", sizes_b, compress, lambda be: [r.copy() for r in based])

        def prepare(recs, be):
            recon.prepare(recs, backend=be)
            return recs
        b.case(name + ": recon.prepare", what + "; both steps as processing calls them", sizes, prepare, copies)
        return based

    def sas():
        n = 2 if q else max(int(round(8 * b.scale)), 1)
        raw = raw_pings("sas_auv", {"receive": {"window": {"range": [30.0, 45.0 if q else 100.0]}}}, [[1.0, 40.0, 13.0]], n,
                        advance=0.32)
        conditioning("sas_auv raw", "SAS pings at the ADC rate", raw)
    b.guard("sas_auv raw", sas)

    def flank():
        raw = raw_pings("flank_32", {"receive": {"window": {"range": [700.0, 1300.0] if q else [150.0, 150.0 + 9850.0 * min(b.scale, 1.0)]}}},
                        [[0.0, 1000.0, 20.0]], 1 if q else max(int(round(b.scale)), 1))
        conditioning("flank_32 raw", "flank array ping, 1 s LFM", raw)
    b.guard("flank_32 raw", flank)

    def sampled():
        cfg = sampled_waveform(echoes_only(scenario_config("flank_sst_shallow", 1, quick=q)), "mseq")
        cfg.pop("time_spread", None)
        raw = run_scenario(cfg).records
        if raw[0].transmit.waveform.type != core.WaveformType.Arbitrary:
            raise RuntimeError("the record does not carry a sampled waveform")
        based = conditioning("flank, sampled m-sequence", "arbitrary-waveform matched filter (replica from the record's samples)", raw)
        rates = np.linspace(-8.0, 8.0, 5 if q else 17)
        from gpusonar.processing import doppler_bank

        def bank(recs, be):
            best, _, peaks = doppler_bank(recs[0], rates, backend=be)
            return [best, np.asarray(peaks, dtype=np.float32)]
        sizes = {"hypotheses": len(rates), "channels": int(based[0].samples.n_channels), "samples": int(based[0].samples.n_samples)}
        b.case("Doppler bank, sampled m-sequence", "processing.doppler_bank: pulse_compress once per range-rate hypothesis, on the backend",
               sizes, bank, lambda be: [based[0]], note="agreement: the best hypothesis's record and the peak of every hypothesis")
    b.guard("flank, sampled m-sequence", sampled)


_LIVE_PINGS = {}


def live_pings(rows_sonar, n_pings, window=None):
    """Pings of the stacked-row free-field scenario with the named sonar in place of its own
    (simulated once on the host and kept; copies are returned)."""
    key = (rows_sonar, tuple(window) if window else None)
    if key not in _LIVE_PINGS or len(_LIVE_PINGS[key]) < n_pings:
        cfg = scenario_config("popoto_elevation_free", n_pings)
        cfg["sonar"] = rows_sonar
        if window is not None:
            cfg["sonar_overrides"] = deep_merge(cfg.get("sonar_overrides") or {}, {"receive": {"window": {"range": window}}})
        _LIVE_PINGS[key] = run_scenario(cfg).records
    return [r.copy() for r in _LIVE_PINGS[key][:n_pings]]


def live_window(b):
    if b.quick or b.scale == 1.0:
        return None
    w = (load_scenario_config("popoto_elevation_free").get("sonar_overrides") or {}).get("receive", {}).get("window", {}).get("range")
    return None if not w else [float(w[0]), float(w[0]) + (float(w[1]) - float(w[0])) * b.scale]


def group_detect(b):
    q = b.quick

    def cfar():
        shape = (121, 2000) if q else (241, int(8000 * b.scale))
        rng = np.random.default_rng(5)
        power = rng.exponential(1.0, shape).astype(np.float32)
        power[::17, ::211] *= 300.0

        def run(_, be):
            det, ratio = core.cfar(power, pfa=1e-5, guard=50, train=150, backend=be, return_ratio=True)
            return [np.asarray(ratio), np.asarray(det, dtype=np.float32)]
        b.case("CFAR", "cell-averaging CFAR along range on a bearing x range power map", {"bearings": shape[0], "range cells": shape[1]}, run)
    b.guard("CFAR", cfar)

    def sbl():
        rec = live_pings("popoto_long_range", 1, live_window(b))[0]
        recon.prepare([rec])
        s = rec.samples
        chain = (load_yaml(ROOT / "configs" / "deploy" / "popoto_vehicle.yaml").get("chain") or {})
        first, last, _ = chain.get("bearings_deg", [-60.0, 60.0, 0.5])
        bearings = [float(v) for v in np.radians(np.arange(first, last + 0.125, 0.25))]
        from gpusonar.live import DEFAULTS as LIVE
        o = dict(LIVE["sbl"], **(chain.get("sbl") or {}))
        sbl_kw = dict(max_iterations=int(o["max_iterations"]), tolerance=float(o["tolerance"]), max_sources=int(o["max_sources"]),
                      dynamic_range_db=float(o["dynamic_range_db"]), min_snr_db=float(o["min_snr_db"]))
        n_gates = 2 if q else max(int(round(int(o["max_gates"]) * b.scale)), 1)
        times = np.linspace(s.start_time + 0.2 * s.n_samples / s.sample_rate, s.start_time + 0.8 * s.n_samples / s.sample_rate, n_gates)
        sizes = {"gates": n_gates, "channels": int(s.n_channels), "bearing grid": len(bearings), "n_fft": 256}

        def gates(method, be):
            return [np.asarray(core.doa_gate(rec, float(t), 256, bearings, method=method, backend=be, **sbl_kw)["spectrum"]) for t in times]
        b.case("doa_gate (sbl)", "bearings at range gates of a conditioned ping: sparse Bayesian learning", sizes,
               lambda _, be: gates("sbl", be), note="SBL options of the live chain (iterations, tolerance, sources)")
        b.group, saved = "hostonly", b.group
        b.host_case("doa_gate (conventional)", "the conventional beamformer on the same snapshots: a few small matrix products per gate, on the host "
                    "whatever the backend (the result's \"backend\" entry says so)", sizes, lambda _: gates("conventional", HOST))
        b.group = saved
    b.guard("doa_gate", sbl)

    def passive():
        rec = live_pings("popoto_long_range", 1, live_window(b))[0]

        def run(_, be):
            frames, _ = core.doa_passive(rec, method="sbl", backend=be, step_deg=1.0 if q else 0.5, max_iterations=20, tolerance=0.01)
            return [np.asarray(f["spectrum"]) for f in frames]
        s = rec.samples
        b.case("doa_passive (sbl)", "passive bearing-time record of one stretch of data, sparse Bayesian learning",
               lambda out, row: {"frames": len(out), "channels": int(s.n_channels), "samples": int(s.n_samples), "bearing grid": int(out[0].size) if out else 0},
               run)
    b.guard("doa_passive", passive)


def beamformer_cases(b):
    """The far-field beamformers alone, on one conditioned ping of each array (1, 2, 3 rows of 32)."""
    from gpusonar import flank
    q = b.quick
    sonars = list_configs("sonars")
    for rows, sonar in ((1, "popoto_long_range"), (2, "popoto_long_range_2row"), (3, "popoto_long_range_3row")):
        if sonar not in sonars or (q and rows > 1):
            continue

        def build(rows=rows, sonar=sonar):
            rec = live_pings(sonar, 1, live_window(b))[0]
            recon.prepare([rec])
            if q:
                samples = rec.samples                    # the quick run: the first 4096 samples of the ping
                kind = samples.kind
                samples.array = np.ascontiguousarray(samples.array[:, :4096])
                samples.kind = kind
                rec.samples = samples
            bearings = np.radians(np.arange(-60.0, 60.25, 2.0 if q else 0.5))
            sizes = {"rows": rows, "channels": int(rec.samples.n_channels), "samples": int(rec.samples.n_samples), "bearings": len(bearings)}
            label = f"{rows} row{'s' if rows > 1 else ''} ({sizes['channels']} channels at once)"
            b.case(f"flank.farfield_beamform, {label}", "far-field delay and sum of one conditioned ping (C++; the device: transforms and steered sums)",
                   sizes, lambda r, be: flank.farfield_beamform(r, bearings, window="hann", backend=be).data, lambda be: rec.copy())
            b.case(f"flank.mvdr_beamform, {label}", "sub-band MVDR of one conditioned ping (C++; the device: covariances, Cholesky, solves, weights applied)",
                   sizes, lambda r, be: flank.mvdr_beamform(r, bearings, backend=be).data, lambda be: rec.copy())
        b.guard(f"beamformers, {rows} row(s)", build)


def group_live(b):
    from gpusonar.live import LiveChain
    q = b.quick
    sonars = list_configs("sonars")
    beamformer_cases(b)
    deploy = ROOT / "configs" / "deploy"
    rows_sonar = {1: "popoto_long_range", 2: "popoto_long_range_2row", 3: "popoto_long_range_3row"}
    n_pings = 2 if q else max(int(round(5 * min(b.scale, 4.0))), 3)
    window = live_window(b)
    for rows in (1, 2, 3):
        if rows_sonar[rows] not in sonars:
            b.skip(f"{rows} row(s)", f"no sonar config {rows_sonar[rows]}")
            continue
        chain_file = deploy / ("popoto_vehicle.yaml" if rows == 1 else "popoto_vehicle_elevation.yaml")
        if not chain_file.exists():
            b.skip(f"{rows} row(s)", f"no chain config {chain_file.name}")
            continue
        chain = load_yaml(chain_file).get("chain") or {}
        # The chain's sync stage lines the boards up on a reference summed into the raw samples by
        # hardware; the simulated pings carry none, so the stage is left out here.
        chain.pop("sync", None)
        if rows > 1 and chain.get("elevation") is None:
            b.skip(f"{rows} row(s)", f"{chain_file.name} has no chain.elevation")
            continue
        try:
            records = live_pings(rows_sonar[rows], n_pings, window)
        except Exception as e:  # noqa: BLE001
            b.guard(f"{rows} row(s)", lambda e=e: (_ for _ in ()).throw(e))
            continue
        sizes = {"rows": rows, "pings": len(records), "channels": int(records[0].samples.n_channels),
                 "samples": int(records[0].samples.n_samples)}
        bd = chain.get("bearings_deg") or [-40.0, 40.0, 0.5]
        sizes["bearings"] = int(round((bd[1] - bd[0]) / bd[2])) + 1
        combos = [("python", "map", "backprojection"), ("cpp", "map", "backprojection"), ("cpp", "sbl", "backprojection"),
                  ("cpp", "map", "farfield"), ("cpp", "map", "mvdr")]
        if rows == 1 and not q:
            combos += [("python", "map", "farfield"), ("python", "map", "mvdr"), ("cpp", "sbl", "farfield"), ("cpp", "sbl", "mvdr")]
        elif rows > 1 and q:
            combos = combos[:3]
        for engine, doa, beamformer in combos:
            name = f"{rows} row{'s' if rows > 1 else ''}{' + elevation' if rows > 1 else ''}: engine {engine}, doa {doa}, {beamformer}"
            what = f"{rows_sonar[rows]} with the chain of {chain_file.name}; seconds per ping (median after the first)"

            def run(recs, be, engine=engine, doa=doa, beamformer=beamformer):
                cfg = dict(chain, engine=engine, doa=doa, beamformer=beamformer, backend="cuda" if be == CUDA else "host")
                times, report = [], None
                with LiveChain(cfg) as live:
                    for r in recs:
                        t0 = time.perf_counter()
                        report = live.process(r)
                        times.append(time.perf_counter() - t0)
                    power = np.array(live.last["power"], dtype=np.float32)
                per_ping = float(np.median(times[1:])) if len(times) > 1 else times[0]
                return Timed(per_ping, (power, report))

            def compare(h, c):
                dh, dc = h[1]["detections"], c[1]["detections"]
                same = len(dh) == len(dc) and all(abs(x["range_m"] - y["range_m"]) < 0.5 and abs(x["bearing_deg"] - y["bearing_deg"]) < 0.05
                                                  for x, y in zip(dh, dc))
                return relative_rms(h[0], c[0]), f"detections of the last ping host {len(dh)} / cuda {len(dc)}, {'the same' if same else 'DIFFERENT'}"

            note = "the first ping of each run is the warm-up; agreement: power map of the last ping"
            if engine == "python" and beamformer != "backprojection":
                note += "; the Python engine calls the C++ beamformer on the backend (flank.py)"
            if doa == "sbl":
                note += "; doa sbl always runs the C++ chain"
            row = b.case(name, what, sizes, run, lambda be: [r.copy() for r in records], compare=compare, note=note, keep=True,
                         warmup=False)
            out = row.pop("_host_out", None)
            if out is not None:
                row["sizes"] = dict(row["sizes"], detections=len(out[1]["detections"]))


def group_hostonly(b):
    q = b.quick
    from gpusonar import flank

    def beamformers():
        rec = live_pings("popoto_long_range", 1, live_window(b))[0]
        recon.prepare([rec])
        if q:
            samples = rec.samples                    # the quick run: the first 4096 samples of the ping
            kind = samples.kind
            samples.array = np.ascontiguousarray(samples.array[:, :4096])
            samples.kind = kind
            rec.samples = samples
        bearings = np.radians(np.arange(-60.0, 60.25, 2.0 if q else 0.5))
        sizes = {"channels": int(rec.samples.n_channels), "samples": int(rec.samples.n_samples), "bearings": len(bearings)}
        b.host_case("flank.farfield_beamform, engine numpy", "far-field delay-and-sum in NumPy: the reference of the C++ beamformer (group live), "
                    "no longer what the Python chain runs", sizes, lambda _: flank.farfield_beamform(rec, bearings, window="hann", engine="numpy"))
        b.host_case("flank.mvdr_beamform, engine numpy", "sub-band MVDR in NumPy: the reference of the C++ beamformer (group live), no longer what "
                    "the Python chain runs", sizes, lambda _: flank.mvdr_beamform(rec, bearings, engine="numpy"))
    b.guard("beamformers", beamformers)

    def noise():
        n, fs = (64000 if q else int(640000 * b.scale)), 64000.0
        positions = np.column_stack([0.094 * np.arange(32), np.zeros(32), np.zeros(32)])
        b.host_case("sim.noise.gaussian_noise", "ambient noise with the Wenz spectrum and isotropic coherence across the array",
                    {"channels": 32, "samples": n},
                    lambda _: noise_models.gaussian_noise(n, fs, lambda f: noise_models.wenz_spectrum_db(f, 5.0, 0.5),
                                                          np.random.default_rng(1), positions=positions, sound_speed=1500.0))
    b.guard("ambient noise", noise)

    def pairing():
        cfg = echoes_only(scenario_config("flank_sst_spread", 1, quick=q))
        timer = StageTimer()
        timer.patch(simulator, "two_way_paths", S_PAIR)
        timer.patch(timespread.TimeSpreadProvider, "eigenrays", S_SPREAD)
        timer.patch(propagation.ImageSourceProvider, "eigenrays", S_EIGEN)
        try:
            m, _ = measure(lambda _: run_scenario(copy.deepcopy(cfg)), None, b.repeats, b.budget, timer)
        finally:
            timer.restore()
        for label, what in ((S_PAIR, "two_way_paths: pairing outgoing and returning rays (C++ on the host)"),
                            (S_SPREAD, "TimeSpreadProvider.eigenrays: taps, Doppler lines, angle spread (NumPy)"),
                            (S_EIGEN, "ImageSourceProvider.eigenrays (C++ on the host)")):
            b.host_row(f"flank_sst_spread, one ping: {label}", what, {"pings": 1}, m["stages"].get(label, 0.0))
            log(f"[hostonly] flank_sst_spread {label}: {fmt_time(m['stages'].get(label, 0.0))} s")
    b.guard("path pairing", pairing)

    def modifiers():
        cfg = echoes_only(scenario_config("fls_glints_on_sediment", 1, 0.1 if q else b.scale))
        timer = StageTimer()
        timer.patch(scene_mod.Scene, "evaluate", S_SCENE)
        timer.patch(scene_mod.Scene, "visible", S_SCENE)
        timer.patch(simulator, "build_scene", S_BUILD)
        try:
            m, _ = measure(lambda _: run_scenario(copy.deepcopy(cfg)), None, b.repeats, b.budget, timer)
        finally:
            timer.restore()
        b.host_row("fls_glints_on_sediment, one ping: " + S_SCENE, "Scene.evaluate and Scene.visible (NumPy): aspect-dependent targets, glints, seabed model, occlusion",
                   {"pings": 1}, m["stages"].get(S_SCENE, 0.0))
        b.host_row("fls_glints_on_sediment: " + S_BUILD, "build_scene (NumPy)", {}, m["stages"].get(S_BUILD, 0.0))
    b.guard("scene modifiers", modifiers)

    def motion():
        if not getattr(core, "HAVE_MOTION", False):
            return b.skip("motion estimation", "built without Ceres (GPUSONAR_CERES=1 in the job builds it); it has no CUDA path either way")
        from gpusonar.motion import estimate_motion
        records = run_scenario(scenario_config("minefield_sas", 12 if q else max(int(round(80 * b.scale)), 12), 0.1 if q else 1.0)).records
        b.host_case("motion.estimate_motion", "displaced-phase-centre delays (NumPy) and the Ceres solve", {"pings": len(records)},
                    lambda recs: estimate_motion(recs), lambda: [r.copy() for r in records])
    b.guard("motion estimation", motion)

    def chain_parts():
        from gpusonar import detect
        rng = np.random.default_rng(2)
        power = rng.exponential(1.0, (241, 2000 if q else 8000)).astype(np.float32)
        b.host_case("detect.cfar (NumPy)", "the NumPy CFAR used when no backend is passed", {"bearings": power.shape[0], "range cells": power.shape[1]},
                    lambda _: detect.cfar(power, pfa=1e-5, guard=50, train=150))
    b.guard("chain parts", chain_parts)


RUNNERS = {"synthesis": group_synthesis, "reverb": group_reverb, "scenarios": group_scenarios, "imaging": group_imaging,
           "conditioning": group_conditioning, "detect": group_detect, "live": group_live, "hostonly": group_hostonly}

NOT_TIMED = [
    "recon.image_volume, recon.sas_image, recon.fan_image, flank.backprojection_map, elevation.estimate / elevation.profile and the "
    "PICMUS evaluation (benchmarks.py) all call the one backprojection kernel timed under `imaging` and inside the `live` cases "
    "(the elevation stage is in the rows with `+ elevation`); they have no kernel of their own.",
    "recon.multigpu.backproject_tiles (several devices) is timed by tools/benchmark.py on machines with more than one GPU.",
    "The Holoscan operators (holoscan_ops.py, C++ operators) wrap the same kernels; they need the Holoscan SDK and are covered by their tests.",
    "pipeline.run takes a backend and passes it to prepare and to the imaging stages timed here.",
]


# --- report ---------------------------------------------------------------------------------

def header(b, started):
    n_dev = core.cuda_device_count() if b.cuda and core.cuda_available() else 0
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = os.cpu_count()
    return {"date_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(started)), "commit": commit,
            "machine": platform.machine(), "processor": cpu_model() or platform.processor(), "python": platform.python_version(),
            "numpy": np.__version__, "host_threads": affinity, "cpu_count": os.cpu_count(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "cuda_compiled": bool(getattr(core, "cuda_compiled", lambda: False)()), "cuda_available": bool(core.cuda_available()),
            "cuda_used": b.cuda, "gpus": [core.cuda_device_name(i) for i in range(n_dev)],
            "bellhop": find_bellhop() or None, "reverb_components_in_cpp": bool(reverb.cpp_has_components()), "sim_engine": importlib.import_module("gpusonar.sim.engine").resolve(),
            "quick": b.quick, "scale": b.scale, "repeats": b.repeats, "case_budget_s": b.budget,
            "groups": b.args.groups, "wall_seconds": time.time() - started}


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def markdown(b, head):
    out = ["# Host against CUDA: every stage with a CUDA backend", ""]
    gpus = ", ".join(head["gpus"]) if head["gpus"] else ("none (CUDA not built)" if not head["cuda_compiled"] else "none available")
    out += [f"- commit `{head['commit']}`, {head['date_utc']} UTC, {head['wall_seconds']:.0f} s for the whole run",
            f"- host: {head['processor'] or head['machine']}, {head['host_threads']} threads usable ({head['cpu_count']} CPUs"
            + (f", OMP_NUM_THREADS={head['omp_num_threads']}" if head["omp_num_threads"] else "") + ")",
            f"- GPU: {gpus}" + ("" if head["cuda_used"] or not head["cuda_available"] else " (not used: --backend host)"),
            f"- timing: one warm-up run (not counted), then the best of {head['repeats']}; a case that would take more than "
            f"{head['case_budget_s']:.0f} s for that is run fewer times, and its note says so",
            f"- sizes: {'quick (small)' if head['quick'] else 'full'}, scale {head['scale']:g}; simulator engine {head['sim_engine']}; "
            f"Bellhop {'present' if head['bellhop'] else 'absent'}",
            "- agreement: sqrt(sum |host - cuda|^2 / sum |host|^2) over the samples, image or map the case produces", ""]
    out += ["## Stages with a CUDA backend", "",
            "| Group | Case | What | Sizes | Host, s | CUDA, s | Speed-up | Host-CUDA rel. rms | Note |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in b.cases:
        host = fmt_time(r["host_s"]) if not r.get("host_error") else "error: " + r["host_error"]
        note = r.get("note") or ""
        for side in ("host", "cuda"):
            runs = len(r.get(f"{side}_runs") or [])
            if r.get(f"{side}_s") is not None and runs and (runs < head["repeats"] or r.get(f"{side}_warmup_counted")):
                note = (note + "; " if note else "") + f"{side}: {runs} run(s)" + (", no separate warm-up (over the time budget)" if r.get(f"{side}_warmup_counted") else "")
        out.append("| " + " | ".join(md_escape(v) for v in (
            r["group"], r["case"], r["what"], fmt_sizes(r["sizes"]), host, cuda_cell(r), fmt_ratio(r["speedup"]),
            fmt_agreement(r["agreement"]), note)) + " |")
    out += ["", "## Host only: stages without a CUDA build", "",
            "| Group | Stage | What | Sizes | Host, s | Note |", "|---|---|---|---|---|---|"]
    for r in b.host_only:
        host = fmt_time(r["host_s"]) if not r.get("host_error") else "error: " + r["host_error"]
        out.append("| " + " | ".join(md_escape(v) for v in (r["group"], r["case"], r["what"], fmt_sizes(r["sizes"]), host, r.get("note") or "")) + " |")
    if not b.host_only:
        out.append("| - | - | none run (groups `reverb`, `scenarios`, `hostonly` fill this table) | | | |")
    if b.breakdown:
        labels = [k for k in CUDA_STAGES + HOST_STAGES if any(row["stages"].get(k) for row in b.breakdown)]
        out += ["", "## Where the time goes in whole scenarios", "",
                "Seconds inside each stage in the best run. The first " + str(sum(1 for k in labels if k in CUDA_STAGES))
                + " stage column(s) run on the device with the CUDA backend; all the others are on the host on both.", "",
                "| Scenario | Backend | Total | " + " | ".join(labels) + " |", "|---|---|---|" + "---|" * len(labels)]
        for row in b.breakdown:
            out.append(f"| {md_escape(row['case'])} | {row['backend']} | {fmt_time(row['total_s'])} | "
                       + " | ".join(fmt_time(row["stages"].get(k)) if row["stages"].get(k) else "-" for k in labels) + " |")
    out += ["", "## Not run, and not timed separately", ""]
    out += [f"- `{s['group']}` {s['case']}: {s['reason']}" for s in b.skipped] or ["- every case ran"]
    out += ["- " + line for line in NOT_TIMED]
    failed = [r for r in b.cases + b.host_only if r.get("host_error") or (r.get("cuda_error") not in (None, "unavailable"))]
    out += ["", f"{len(b.cases)} cases with a CUDA backend, {len(b.host_only)} host-only rows, {len(b.skipped)} skipped, {len(failed)} with an error."]
    return "\n".join(out) + "\n"


def to_json(b, head):
    def clean(row):
        return {k: v for k, v in row.items() if not k.startswith("_")}
    return {"header": head, "cases": [clean(r) for r in b.cases], "host_only": [clean(r) for r in b.host_only],
            "scenario_breakdown": b.breakdown, "skipped": b.skipped, "not_timed": NOT_TIMED}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small sizes and one counted run per case: a few minutes on a CPU")
    ap.add_argument("--only", default="", help="comma-separated groups: " + ", ".join(GROUPS))
    ap.add_argument("--scale", type=float, default=1.0, help="grow the full sizes (pings, scatterer density, pixels)")
    ap.add_argument("--repeats", type=int, default=None, help="counted runs per case after the warm-up (default 3; 1 with --quick)")
    ap.add_argument("--budget", type=float, default=None,
                    help="seconds a case's counted runs may take on one backend before fewer are made (default 180; 20 with --quick)")
    ap.add_argument("--backend", default="auto", choices=("auto", "host", "cuda"),
                    help="auto: the device when the build has CUDA and one is present; host: never; cuda: try it whatever the build says "
                         "(every failure is then recorded in the CUDA column)")
    ap.add_argument("--out", default="", help="write the Markdown table here (default: stdout)")
    ap.add_argument("--json", default="", help="write the results as JSON here")
    ap.add_argument("--traceback", action="store_true", help="print the traceback of a failing case to stderr")
    args = ap.parse_args(argv)
    args.repeats = args.repeats if args.repeats is not None else (1 if args.quick else 3)
    args.budget = args.budget if args.budget is not None else (20.0 if args.quick else 180.0)
    args.groups = [g.strip() for g in args.only.split(",") if g.strip()] or list(GROUPS)
    unknown = [g for g in args.groups if g not in GROUPS]
    if unknown:
        ap.error(f"unknown group(s) {unknown}: choose from {', '.join(GROUPS)}")
    if args.repeats < 1:
        ap.error("--repeats must be at least 1")
    started = time.time()
    b = Bench(args)
    devices = [core.cuda_device_name(i) for i in range(core.cuda_device_count())] if core.cuda_available() else []
    log(f"backend: host{' and cuda (' + (', '.join(devices) or 'forced: no device reported') + ')' if b.cuda else ' only (CUDA ' + ('not built' if not getattr(core, 'cuda_compiled', lambda: False)() else 'not available or not asked for') + ')'}; "
        f"{'quick' if args.quick else 'full'} sizes, scale {args.scale:g}, best of {args.repeats}; groups {', '.join(args.groups)}")
    for g in GROUPS:
        if g not in args.groups:
            continue
        b.group = g
        t0 = time.time()
        try:
            RUNNERS[g](b)
        except Exception as e:  # noqa: BLE001 - a group that fails as a whole is recorded, the others still run
            b.guard(g, lambda e=e: (_ for _ in ()).throw(e))
        log(f"[{g}] done in {time.time() - t0:.0f} s")
    head = header(b, started)
    text = markdown(b, head)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(to_json(b, head), indent=2, default=float))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        log(f"wrote {args.out}" + (f" and {args.json}" if args.json else ""))
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
