"""Config-file-driven processing chain (ASASIN Sec. II, Fig. 1).

One YAML file names the data source, the processing steps and the outputs. Running it
produces the image products, a processing log (timings, the choices made, data issues)
and, if asked, a debug product after every step. The resolved config is stored in the log
so a run can be repeated exactly.

    source: {type: scenario, name: stripmap_points}        # or container / wav / picmus
    steps:
      - {stage: prepare}
      - {stage: whiten, gamma: 0.1}
      - {stage: median_tvg, alpha: 0.01}
      - {stage: motion}                                    # SAS micronavigation
      - {stage: sas_image, ground_range: [36, 47], pixel: [0.02, 0.04]}
      - {stage: drc, target_median: 0.2}
    output: {geotiff: out.tif, log: run.jsonl, debug_dir: debug/, taps: {dir: taps/}}

`taps` writes the ping records after the source and after every step as container files
(gpusonar/taps.py). A later run can start from any of them:

    source: {type: tap, dir: taps/, stage: whiten}
    steps: [...the steps after whiten...]

and `source: {type: scenario, name: ..., sim_taps: dir/}` also saves the samples after
each stage inside the simulator.

A step passes on only the keys it is given: the default of every other parameter is the
default of the function it calls (recon.sas_image, recon.fan_image, ...), stated there once
(docs/HARDCODED_AUDIT.md R-5). Keys that are derived from the records instead of typed:

    - {stage: tvg}                                         # absorption from the record (the default)
    - {stage: tvg, spreading: auto, absorption: none}      # recon.tvg_parameters; numbers also work
    - {stage: sas_image, ..., plane_z: seafloor}           # from the DVL altitude or the water depth
    - {stage: motion, seafloor_z: auto, max_lag_m: 0.01}   # windows in metres of range

The log of a `tvg` step says where the gain law was cut at `max_gain_db`, and that of a
`sas_image` step the plane used and the seabed the records imply.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import _gpusonar as core
from . import recon
from .config import dump_yaml, load_yaml
from .sources import make_source


@dataclass
class RunResult:
    records: list
    image: np.ndarray | None = None       # complex image or volume from the imaging step
    display: np.ndarray | None = None     # after dynamic range compression
    points: np.ndarray | None = None
    products: dict = field(default_factory=dict)
    log: list = field(default_factory=list)


class _Log:
    def __init__(self, path):
        self.entries = []
        self._file = open(path, "w", encoding="utf-8") if path else None

    def write(self, **entry):
        entry["wall_time"] = time.time()
        self.entries.append(entry)
        if self._file:
            self._file.write(json.dumps(entry, default=str) + "\n")
            self._file.flush()

    def close(self):
        if self._file:
            self._file.close()


def _load_records(cfg: dict, log: _Log) -> list:
    kind = cfg.get("type")
    if kind == "scenario":
        from .sim import run_scenario
        from . import taps
        if cfg.get("sim_taps"):
            # Samples after every simulator stage (echoes, reverberation, noise, gain, ADC).
            with taps.sim_capture(cfg["sim_taps"]):
                records = run_scenario(cfg["name"], cfg.get("overrides")).records
        else:
            records = run_scenario(cfg["name"], cfg.get("overrides")).records
    elif kind == "tap":
        # Records captured after a stage of an earlier run: start the chain from there.
        from . import taps
        records = taps.read_tap(cfg["dir"], cfg["stage"])
    else:
        with make_source(cfg) as src:
            records = list(src)
    issues = 0
    for i, r in enumerate(records):
        report = core.validate(r)
        for issue in report.issues:
            issues += 1
            log.write(event="data_issue", ping=i, severity=str(issue.severity).split(".")[-1],
                      field=issue.field, message=issue.message)
        if not report.ok:
            raise ValueError(f"ping {i} failed validation:\n{report}")
    log.write(event="source", type=kind, pings=len(records), data_issues=issues,
              channels=records[0].samples.n_channels if records else 0)
    return records


def _stage(name, params, state: RunResult, backend):
    """Run one step. Returns a dict of facts for the log."""
    r = state.records
    if name == "prepare":
        recon.prepare(r, compress=params.get("compress", True), window=params.get("replica_window", "none"))
        return {"sample_rate": r[0].samples.sample_rate, "samples": r[0].samples.n_samples}
    if name == "calibrate":
        for rec in r:
            core.apply_calibration(rec)
        return {}
    if name == "tvg":
        # `spreading` and `absorption` (names or numbers) are resolved per record; the older
        # `spreading_db_per_decade` and `absorption_db_per_m` keys pass through unchanged.
        derive = {k: params[k] for k in ("spreading", "absorption", "frequency", "temperature", "salinity", "depth",
                                         "ph", "model") if k in params}
        plain = {k: v for k, v in params.items() if k not in derive}
        facts = {}
        if "spreading" in derive and "spreading_db_per_decade" in plain:
            raise ValueError("tvg: give `spreading` or `spreading_db_per_decade`, not both")
        if "absorption" in derive and "absorption_db_per_m" in plain:
            raise ValueError("tvg: give `absorption` or `absorption_db_per_m`, not both")
        from .recon.gain import tvg_parameters
        # Every tvg step goes through recon.tvg_parameters, whose default absorption is "auto":
        # derived from the record (DECISIONS.md D-91). `absorption: none` is the law before that.
        d = dict(derive)
        if "spreading_db_per_decade" in plain:
            d["spreading"] = plain.pop("spreading_db_per_decade")
        if "absorption_db_per_m" in plain:
            d["absorption"] = plain.pop("absorption_db_per_m")
        d.update(plain)
        for rec in r:
            kw = tvg_parameters(rec, **d)
            report = core.apply_tvg(rec, **kw)
            if not facts:
                facts = {"spreading_db_per_decade": kw.get("spreading_db_per_decade", 40.0),
                         "absorption_db_per_m": kw.get("absorption_db_per_m", 0.0),
                         "gain_last_db": report["gain_last_db"], "clamped_samples": report["clamped_samples"],
                         "clamp_range_m": report["clamp_range"] if report["clamped_samples"] else None,
                         "shortfall_db": report["shortfall_db"]}
        return facts
    if name == "whiten":
        gain = core.whiten(r, **params)
        state.products["whiten_gain"] = gain
        return {"min_gain_db": float(10 * np.log10(gain.min()))}
    if name == "median_tvg":
        gain = core.median_tvg(r, **params)
        state.products["tvg_gain"] = gain
        return {"gain_span_db": float(10 * np.log10(gain.max() / gain.min()))}
    if name == "motion":
        from . import motion
        result = motion.estimate_motion(r, motion.MotionOptions(**params))
        motion.apply_motion(r, result)
        state.products["motion_y"], state.products["motion_z"] = result.y, result.z
        return {"measurements": result.n_measurements, "rejected": result.n_rejected,
                "unwrap_corrections": result.unwrap_corrections, "solver": result.log, **result.windows}
    if name == "sas_image":
        # Only what the config gives is passed on: every default lives in recon.sas.
        grid = recon.stripmap_grid(r, params["ground_range"], params["pixel"],
                                   **{k: params[k] for k in ("plane_z", "full_support", "beamwidth_az") if k in params})
        stats = {}
        state.points = grid.points
        image_keys = ("interp", "fov_factor", "cull_block", "directivity_weight", "replica_window", "position_lookup",
                      "trajectory_allowance")
        state.image = recon.sas_image(r, grid.points, stats=stats, backend=backend,
                                      **{k: params[k] for k in image_keys if k in params})
        # The plane used, and the seabed the records themselves imply (None: they do not say).
        try:
            implied, source = recon.seafloor_z(r)
        except ValueError:
            implied, source = None, None
        return {"shape": list(state.image.shape), "plane_z": float(grid.z), "seafloor_z_from_records": implied,
                "seafloor_z_source": source, **stats}
    if name == "fan_image":
        ranges = np.linspace(*params["ranges"])
        bearings = np.radians(np.linspace(*params["bearings_deg"]))
        fan_keys = ("interp", "window", "fov_factor", "directivity_weight", "stop_and_hop", "replica_window")
        fan = recon.fan_image(r, ranges, bearings, backend=backend, **{k: params[k] for k in fan_keys if k in params})
        state.image, state.points = fan.image, fan.points
        return {"shape": list(fan.image.shape)}
    if name == "volume":
        x, y, z = (np.arange(*params[k]) for k in ("x", "y", "depths"))
        vol = recon.image_volume(r, x, y, z, interface_z=params.get("interface_z"),
                                 sediment_speed=params.get("sediment_speed"), backend=backend)
        X, Y = np.meshgrid(x, y, indexing="ij")
        state.image = vol.data
        state.points = np.stack([X, Y, np.zeros_like(X)], axis=-1)
        state.products["depths"] = z
        return {"shape": list(vol.data.shape), "peak_xyz": list(vol.peak())}
    if name == "drc":
        if state.image is None:
            raise ValueError("drc needs an image: put an imaging step before it")
        mag = np.abs(state.image).astype(np.float32)
        state.display, q = core.drc(mag, **params)
        return {"q": float(q), "median": float(np.median(state.display))}
    raise ValueError(f"unknown stage {name!r}")


def run(config, backend=None) -> RunResult:
    """Run a processing chain from a config mapping or a YAML file path."""
    cfg = load_yaml(config) if isinstance(config, (str, os.PathLike)) else config
    out_cfg = cfg.get("output", {}) or {}
    want = cfg.get("backend", "auto")
    if want not in ("auto", "host", "cuda"):
        raise ValueError(f"backend: {want!r} (auto, host or cuda; auto, the default, and cuda both use CUDA when a device is present)")
    backend = backend if backend is not None else (
        core.Backend.Cuda if want != "host" and core.cuda_available() else core.Backend.Host)
    log = _Log(out_cfg.get("log"))
    debug_dir = out_cfg.get("debug_dir")
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
    tap = None
    if out_cfg.get("taps"):
        from .taps import TapWriter
        tap_cfg = out_cfg["taps"] if isinstance(out_cfg["taps"], dict) else {"dir": out_cfg["taps"]}
        tap = TapWriter(tap_cfg["dir"], tap_cfg.get("stages", "all"))
    t_start = time.perf_counter()
    try:
        log.write(event="start", config_yaml=dump_yaml(cfg), backend=str(backend).split(".")[-1],
                  cuda_devices=[core.cuda_device_name(i) for i in range(core.cuda_device_count())])
        state = RunResult(records=_load_records(cfg["source"], log))
        if tap:
            tap.records("source", state.records)
        for i, step in enumerate(cfg.get("steps", [])):
            params = dict(step)
            name = params.pop("stage")
            t = time.perf_counter()
            facts = _stage(name, params, state, backend)
            elapsed = time.perf_counter() - t
            log.write(event="stage", index=i, stage=name, seconds=elapsed, parameters=params, **facts)
            if tap:
                # The records as they now stand (a container file, usable as a source), and the image.
                tap.records(name, state.records)
                products = {k: v for k, v in (("image", state.image), ("display", state.display)) if v is not None}
                tap.arrays(name, **products)
            if debug_dir:
                # A debug product per stage: the first ping's samples, and the image if there is one.
                product = {"samples_ping0": state.records[0].samples.array}
                if state.image is not None:
                    product["image"] = state.image
                if state.display is not None:
                    product["display"] = state.display
                np.savez_compressed(os.path.join(debug_dir, f"{i:02d}_{name}.npz"), **product)
        if out_cfg.get("geotiff"):
            if state.image is None:
                raise ValueError("geotiff output needs an imaging step")
            from .output import write_geotiff
            layers = state.display if state.display is not None else np.abs(state.image)
            names = None
            if layers.ndim == 3:
                names = [f"depth {z:.3f} m" for z in state.products.get("depths", range(layers.shape[0]))]
            geo = write_geotiff(out_cfg["geotiff"], layers, state.points,
                                state.records[0].geometry.frame_origin_lla, names,
                                {"array_id": state.records[0].metadata.array_id, "pings": len(state.records)})
            log.write(event="output", product="geotiff", path=out_cfg["geotiff"], **geo)
        collected = sum((r.transmit.time_ns for r in state.records[-1:]), 0) - state.records[0].transmit.time_ns
        total = time.perf_counter() - t_start
        log.write(event="done", seconds=total, collection_seconds=collected * 1e-9,
                  real_time_factor=(collected * 1e-9 / total) if total > 0 else None)
        state.log = log.entries
        return state
    finally:
        log.close()
