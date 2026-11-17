"""The long-range array chain, one ping at a time: the part that runs on a vehicle.

    source (boards over UDP, a recording, or the simulator)
      -> record raw pings                      (always: the recording is the flight data)
      -> synchronise the capture boards        (when there are several: shared reference, sync.py)
      -> prepare (baseband, pulse compression)
      -> beamform (range x bearing)
      -> detect (cell-averaging CFAR, clustering)
      -> track in a world-fixed frame (ownship's motion removed)
      -> classify (mover / static / multipath / uncertain)
      [with `elevation` and an array of stacked rows: elevation and depth of each detection,
       bearing corrected for it, depth carried by the tracks (elevation.py)]
      -> report: one JSON object per ping, to a file and optionally a UDP port

The same object runs on simulated pings, on a replayed recording and on the live stream,
so what is tested ashore is what runs at sea. Every intermediate product can be tapped
(gpusonar/taps.py) for the first `taps.pings` pings.
"""

from __future__ import annotations

import json
import socket
import time

import numpy as np

from . import _gpusonar as core
from . import classify, detect, flank
from .taps import TapWriter

DEFAULTS = {
    "bearings_deg": [-40.0, 40.0, 0.5],        # first, last, step: from broadside
    "window": "hann",                          # aperture weighting
    "sync": None,                              # several boards: {sonar: popoto_4x7} or {reference: {...}}, see sync.py
    "beamformer": "backprojection",            # the kernel (host or CUDA); or "farfield" / "mvdr" (host; C++ with engine: cpp)
    "backend": "auto",                         # "auto": CUDA when built and present, else host; "host"; "cuda"
    "look": "starboard",                       # which side of a line array the bearings are on
    "range_cell_m": 2.0,                       # power is averaged into cells this long before detection
    "cfar": {"pfa": 1e-5, "guard_m": 100.0, "train_m": 300.0},
    "min_snr_db": 9.0,
    "engine": "python",                        # "cpp": libs/live does every stage after recording (what build/bin/sonar_live runs)
    "doa": "map",                              # "sbl": sparse Bayesian bearings at each detected range (engine cpp)
    "sbl": {"grid_step_deg": 0.25, "max_sources": 6, "min_snr_db": 6.0, "dynamic_range_db": 20.0,
            "max_gates": 24, "min_contrast_db": 8.0, "max_iterations": 20, "tolerance": 0.01},
    "tracker": {"range_sigma": 15.0, "bearing_sigma_deg": 1.0, "confirm_hits": 3, "max_misses": 2,
                "initial_speed_sigma": 8.0, "process_noise": 0.05},
    "elevation": None,                         # stacked rows: {} or {sector_deg: ..., ...} switches elevation and depth on (elevation.py)
    "dead_channels": [],                       # channel rows (after synchronisation) left out of every stage
    "classify": {"min_speed": 1.0, "static_speed": 1.0, "max_fit_rms": 60.0,
                 "multipath": {"max_excess_range": 800.0, "bearing_deg": 3.0, "velocity": 2.5}},
}


def _merge(base: dict, over: dict | None) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def cpp_chain_options(cfg: dict, reference: dict | None = None) -> dict:
    """The chain section of a deployment file as the options of the C++ chain (`core.LiveChainCpp`,
    and `key=value` arguments of build/bin/sonar_live). `cfg` is merged over the defaults first."""
    cfg = _merge(DEFAULTS, cfg)
    if cfg["beamformer"] not in ("backprojection", "farfield", "mvdr"):
        raise ValueError(f"beamformer: {cfg['beamformer']!r} (backprojection, farfield or mvdr)")
    b, t, c, s = cfg["bearings_deg"], cfg["tracker"], cfg["classify"], cfg["sbl"]
    o = {"bearing_first_deg": b[0], "bearing_last_deg": b[1], "bearing_step_deg": b[2], "range_cell_m": cfg["range_cell_m"],
         "cfar_pfa": cfg["cfar"]["pfa"], "cfar_guard_m": cfg["cfar"]["guard_m"], "cfar_train_m": cfg["cfar"]["train_m"],
         "min_snr_db": cfg["min_snr_db"], "hann": cfg["window"] == "hann", "starboard": cfg["look"] == "starboard",
         "beamformer": cfg["beamformer"], "doa": cfg["doa"], "sbl_grid_step_deg": s["grid_step_deg"], "sbl_max_sources": int(s["max_sources"]),
         "sbl_min_snr_db": s["min_snr_db"], "sbl_dynamic_range_db": s["dynamic_range_db"],
         "sbl_max_gates": int(s["max_gates"]), "sbl_min_contrast_db": s["min_contrast_db"],
         "sbl_max_iterations": int(s["max_iterations"]), "sbl_tolerance": s["tolerance"],
         "tracker_range_sigma": t["range_sigma"], "tracker_bearing_sigma": float(np.radians(t["bearing_sigma_deg"])),
         "tracker_confirm_hits": int(t["confirm_hits"]), "tracker_max_misses": int(t["max_misses"]),
         "tracker_initial_speed_sigma": t["initial_speed_sigma"], "tracker_process_noise": t["process_noise"],
         "classify_min_speed": c["min_speed"], "classify_static_speed": c["static_speed"], "classify_max_fit_rms": c["max_fit_rms"],
         "multipath": bool(c.get("multipath"))}
    if c.get("multipath"):
        m = c["multipath"]
        o.update(multipath_max_excess_range=m["max_excess_range"], multipath_bearing=float(np.radians(m["bearing_deg"])),
                 multipath_velocity=m["velocity"])
    if cfg.get("dead_channels"):
        o["dead_channels"] = [int(v) for v in cfg["dead_channels"]]
    if cfg.get("elevation") is not None:
        from .elevation import DEFAULTS as EL
        e = _merge(EL, cfg["elevation"])
        o["elevation"] = {"first_deg": float(e["sector_deg"][0]), "last_deg": float(e["sector_deg"][1]), "step_deg": float(e["sector_deg"][2]),
                          "range_window_m": float(e["range_window_m"] or 0.0), "max_detections": int(e["max_detections"]),
                          "min_quality": float(e["min_quality"]), "ambiguity_db": float(e["ambiguity_db"]),
                          "correct_bearing": bool(e["correct_bearing"]), "depth": bool(e["depth"]), "fold": bool(e["fold"]),
                          "classify": bool(e["classify"]), "classify_sigma": float(e["classify_sigma"]),
                          "water_depth": float(e["water_depth"] or 0.0), "surface_z": float(e["surface_z"]),
                          "roll_sigma": float(np.radians(e["roll_sigma_deg"])), "depth_process_noise": float(e["depth_process_noise"]),
                          "row_tolerance_m": float(e["row_tolerance_m"])}
    if reference is not None:
        o["sync"] = {k: (bool(reference[k]) if k == "dedicated" else float(reference[k]))
                     for k in ("f_start", "f_stop", "duration", "lead", "tail", "dedicated") if k in reference}
    return o


def sync_reference(cfg: dict) -> dict | None:
    """The shared reference signal a chain's `sync` section names, or None."""
    sc = cfg.get("sync")
    if not sc:
        return None
    if "reference" in sc:
        return sc["reference"]
    from .sonar import SonarModel
    return SonarModel.load(sc["sonar"], sc.get("sonar_overrides")).config["receive"]["reference"]


class NavListener:
    """The vehicle's navigation, received as JSON datagrams
    {"time_ns": ..., "x": north m, "y": east m, "z": down m, "roll": rad, "pitch": rad, "yaw": rad,
     "vx": ..., "vy": ..., "vz": ...}
    on a UDP port. `at(time_ns)` gives the pose at a ping's transmit time, interpolated between
    the two fixes around it (held from the nearest one outside them). A fixed pose can be given
    instead for a bench or a moored array."""

    def __init__(self, port: int | None = None, host: str = "0.0.0.0", static: dict | None = None, keep: int = 600):
        import collections
        import threading
        self._fixes = collections.deque(maxlen=keep)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.socket = None
        if static is not None:
            self.push(dict(static, time_ns=0))
        if port is not None:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind((host, port))
            self.socket.settimeout(0.2)
            self.port = self.socket.getsockname()[1]
            threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self) -> None:
        while not self._stop.is_set():
            try:
                self.push(json.loads(self.socket.recv(4096)))
            except socket.timeout:
                continue
            except (OSError, ValueError, KeyError):
                continue                       # a bad datagram is dropped, not fatal

    def push(self, fix: dict) -> None:
        row = [float(fix.get(k, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz")]
        with self._lock:
            self._fixes.append((int(fix["time_ns"]), np.array(row)))

    def at(self, time_ns: int):
        """(position [3], rpy [3], velocity [3]) at `time_ns`, or None before any fix has arrived."""
        with self._lock:
            fixes = list(self._fixes)
        if not fixes:
            return None
        times = np.array([f[0] for f in fixes], dtype=np.int64)
        i = int(np.searchsorted(times, time_ns))
        if i == 0 or i == len(fixes) or len(fixes) == 1:
            row = fixes[min(i, len(fixes) - 1)][1]
        else:
            w = (time_ns - times[i - 1]) / max(times[i] - times[i - 1], 1)
            row = (1.0 - w) * fixes[i - 1][1] + w * fixes[i][1]
            # Angles go the short way round.
            row[3:6] = fixes[i - 1][1][3:6] + w * np.angle(np.exp(1j * (fixes[i][1][3:6] - fixes[i - 1][1][3:6])))
        return row[0:3], row[3:6], row[6:9]

    def close(self) -> None:
        self._stop.set()
        if self.socket:
            self.socket.close()


def apply_pose(record, position, rpy, velocity=(0.0, 0.0, 0.0)) -> None:
    """Put the vehicle's pose into a record (both the projector's and the array's)."""
    g = record.geometry
    for name in ("tx_pose", "rx_pose"):
        pose = getattr(g, name)
        pose.position, pose.rpy, pose.velocity = (tuple(float(v) for v in a) for a in (position, rpy, velocity))
        setattr(g, name, pose)
    record.geometry = g


class LiveChain:
    """Process ping records as they arrive. `process(record)` returns the report for that ping."""

    def __init__(self, config: dict | None = None, *, recording: str | None = None, report_path: str | None = None,
                 report_udp: tuple | None = None, taps: dict | None = None, nav: NavListener | None = None):
        self.cfg = _merge(DEFAULTS, config)
        t = self.cfg["tracker"]
        self._elevation = None
        if self.cfg.get("elevation") is not None:
            from .elevation import DEFAULTS as EL
            self._elevation = _merge(EL, self.cfg["elevation"])
        self.tracker = detect.Tracker(range_sigma=t["range_sigma"], bearing_sigma=np.radians(t["bearing_sigma_deg"]),
                                      confirm_hits=t["confirm_hits"], max_misses=t["max_misses"],
                                      initial_speed_sigma=t["initial_speed_sigma"], process_noise=t["process_noise"],
                                      depth_process_noise=(self._elevation or {}).get("depth_process_noise", 0.5))
        b = self.cfg["bearings_deg"]
        self.bearings = np.radians(np.arange(b[0], b[1] + 0.5 * b[2], b[2]))
        self._writer = core.RecordWriter(recording) if recording else None
        self._report = open(report_path, "w", encoding="utf-8") if report_path else None
        self._udp = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM), tuple(report_udp)) if report_udp else None
        self._tap_cfg = taps or {}
        self.nav = nav                         # None: the records already carry the pose (simulation, replay)
        self.pings_without_nav = 0
        self._reference = sync_reference(self.cfg)
        self.boards_out_of_sync = 0
        want = self.cfg["backend"]
        self.backend = core.Backend.Cuda if want == "cuda" or (want == "auto" and core.cuda_available()) else core.Backend.Host
        if self.cfg["doa"] not in ("map", "sbl"):
            raise ValueError(f"doa: {self.cfg['doa']!r} (map or sbl)")
        if self.cfg["engine"] not in ("python", "cpp"):
            raise ValueError(f"engine: {self.cfg['engine']!r} (python or cpp)")
        self._cpp = None
        if self.cfg["engine"] == "cpp" or self.cfg["doa"] == "sbl":
            # One C++ object does synchronisation, conditioning, beamforming, detection, bearings,
            # tracking and classification; this class keeps the navigation, recording, taps and report.
            self._cpp = core.LiveChainCpp(dict(cpp_chain_options(self.cfg, self._reference), backend=self.backend))
        self._origin = None                    # world position of the array at the first ping
        self._t0 = None
        self.pings = 0
        self.last = {}                         # the most recent beam map and detections, for display

    # -- one ping ----------------------------------------------------------------------------

    def process(self, record) -> dict:
        t_wall = time.perf_counter()
        index = self.pings
        self.pings += 1
        tap = None
        if self._tap_cfg.get("dir") and index < int(self._tap_cfg.get("pings", 1)):
            tap = TapWriter(f"{self._tap_cfg['dir']}/ping{index:04d}")
        if self.nav is not None:
            fix = self.nav.at(record.transmit.time_ns)
            if fix is None:
                self.pings_without_nav += 1    # processed where the record says it is; counted and reported
            else:
                apply_pose(record, *fix)
        if self._writer:
            self._writer.write(record)         # raw samples, with the pose, before anything else touches it
            self._writer.flush()
        if tap:
            tap.records("raw", [record])
        cfg = self.cfg
        if self._cpp is not None:
            report = json.loads(self._cpp.process(record))
            report.update(ping=index, pings_without_nav=self.pings_without_nav,
                          processing_seconds=time.perf_counter() - t_wall)
            if report.get("sync") and not report["sync"]["ok"]:
                self.boards_out_of_sync += 1
            power, ranges, _ = self._cpp.last_map()
            if tap:
                tap.records("prepare", [record])               # conditioned in place by the C++ chain
                tap.arrays("beamform", power=power, ranges=ranges, bearings=self.bearings)
                tap.json("report", report)
            self.last = {"power": power, "ranges": ranges, "hits": report["detections"], "classes": report["tracks"]}
            return self._emit(report)
        sync_facts = None
        if self._reference is not None:
            # Boards run on their own clocks: line them up on the shared reference and drop its inputs.
            from . import sync
            rep_ = sync.synchronise(record, self._reference, enforce=False)
            sync_facts = {"ok": rep_.ok, "worst_residual_us": float(rep_.residual.max() * 1e6),
                          "offsets_us": [float(v * 1e6) for v in rep_.estimate.offset],
                          "drift_ppm": [float(v) for v in rep_.estimate.drift_ppm]}
            if not rep_.ok:
                self.boards_out_of_sync += 1   # reported; this ping's detections are still passed on, flagged
            if self._reference.get("dedicated"):
                record = sync.drop_reference_channels(record)   # those inputs carry no hydrophone
            if tap:
                tap.records("sync", [record])
        if cfg["dead_channels"]:
            from . import sync
            dead = {int(v) for v in cfg["dead_channels"]}
            record = sync.select_channels(record, [r for r in range(record.samples.n_channels) if r not in dead])
        from . import recon
        recon.prepare([record], backend=self.backend)        # baseband and pulse compression (cuFFT on a GPU)
        if tap:
            tap.records("prepare", [record])
        rows = subs = None
        if self._elevation is not None:
            from . import elevation
            rows = elevation.find_rows(record, self._elevation["row_tolerance_m"])
            if len(rows) < 2:
                rows = None                    # one row: exactly the chain without the option
            else:
                subs = elevation.row_records(record, rows)
        if rows is None:
            fine, fine_ranges, k = self._fine_map(record)
        else:
            # Each row beamformed on its own, the rows' power added: a row's bearing response
            # does not depend on the elevation the arrival comes from.
            fine = None
            for sub in subs:
                f_, fine_ranges, k = self._fine_map(sub)
                fine = f_ if fine is None else fine + f_
        self._rows, self._subs = rows, subs
        return self._after_map(record, fine, fine_ranges, k, index, tap, sync_facts, t_wall)

    def _fine_map(self, record):
        """Power over bearing x fine range for one line of hydrophones, the fine ranges and
        the number of fine samples per range cell."""
        cfg = self.cfg
        if cfg["beamformer"] == "backprojection":
            # Delay and sum by the backprojection kernel onto range x bearing: exact in the near
            # field too, and the same code on the host and on a GPU.
            c = record.environment.sound_speed
            w = record.transmit.waveform
            band = abs(w.f_stop - w.f_start) or 1.0 / w.duration
            s_ = record.samples
            r0 = 0.5 * c * s_.start_time
            r1 = 0.5 * c * (s_.start_time + (s_.n_samples - 1) / s_.sample_rate - w.duration)
            k = max(int(np.ceil(cfg["range_cell_m"] / (0.75 * c / (2.0 * band)))), 1)   # samples of the image per cell
            step = cfg["range_cell_m"] / k
            fine_r = r0 + step * np.arange(int((r1 - r0) / step) // k * k)
            fan = flank.backprojection_map(record, fine_r, self.bearings, window=cfg["window"], interp="linear",
                                           backend=self.backend)
            return (np.abs(fan.image) ** 2).T, fine_r, k      # [bearing, range]
        else:
            bm = (flank.mvdr_beamform(record, self.bearings) if cfg["beamformer"] == "mvdr"
                  else flank.farfield_beamform(record, self.bearings, window=cfg["window"])).valid()
            fine, fine_ranges = np.abs(bm.data) ** 2, bm.ranges
            k = max(int(round(cfg["range_cell_m"] / (fine_ranges[1] - fine_ranges[0]))), 1)
            return fine, fine_ranges, k

    def _after_map(self, record, fine, fine_ranges, k, index, tap, sync_facts, t_wall):
        cfg = self.cfg
        rows, subs = self._rows, self._subs
        n = fine.shape[1] // k * k
        power = fine[:, :n].reshape(len(self.bearings), -1, k).mean(axis=2)
        ranges = fine_ranges[:n].reshape(-1, k).mean(axis=1)
        cell = ranges[1] - ranges[0]
        guard, train = (max(int(round(cfg["cfar"][key] / cell)), 1) for key in ("guard_m", "train_m"))
        hit_map = detect.cfar(power, pfa=cfg["cfar"]["pfa"], guard=guard, train=train, backend=self.backend)
        hits = detect.cluster(hit_map, power, ranges, self.bearings, guard=guard, train=train)
        hits = [h for h in hits if h.snr_db > cfg["min_snr_db"]]
        if self._reference is not None and not self._reference.get("dedicated"):
            # A reference summed into hydrophone inputs leaves its two bursts in the data, one just
            # after the window opens and one just before it closes: no detections are taken there.
            c = record.environment.sound_speed
            s_ = record.samples
            t_open, t_close = s_.start_time, s_.start_time + s_.n_samples / s_.sample_rate
            pulse, burst = record.transmit.waveform.duration, float(self._reference["duration"])
            lo = 0.5 * c * (t_open + float(self._reference["lead"]) + burst)
            hi = 0.5 * c * (t_close - float(self._reference["tail"]) - pulse - burst)
            hits = [h for h in hits if lo < h.range < hi]
        if tap:
            tap.arrays("beamform", power=power.astype(np.float32), ranges=ranges, bearings=self.bearings)

        # World position of each detection: the array's own position and heading come with the record.
        rx = record.rx_positions()
        origin = rx.mean(axis=0)
        axis = rx[-1, :2] - rx[0, :2]
        el_cfg = self._elevation if rows is not None else None
        els = [None] * len(hits)
        if el_cfg is not None:
            # Stacked rows: the elevation of the strongest detections, from the whole array.
            from . import elevation
            frame = elevation.array_frame(record, rows, cfg["look"])
            origin, axis = frame.origin, frame.along[:2]
            mean, _ = detect._window_mean(np.asarray(power, dtype=float), guard, train)
            order = sorted(range(len(hits)), key=lambda i: -hits[i].snr_db)[:int(el_cfg["max_detections"])]
            background = [float(mean[int(np.argmin(np.abs(self.bearings - hits[i].bearing))),
                                     int(np.argmin(np.abs(ranges - hits[i].range)))]) for i in order]
            window = "rect" if cfg["beamformer"] == "mvdr" else cfg["window"]
            got = elevation.estimate(record, rows, subs, [(hits[i].range, hits[i].bearing) for i in order], background, frame,
                                     el_cfg, noise_weight=sum(elevation.weight_power(len(r), window) for r in rows),
                                     range_cell=cfg["range_cell_m"], backend=self.backend)
            for i, e in zip(order, got):
                els[i] = e
            water = el_cfg["water_depth"] or record.environment.water_depth
            water = float(water) if water and np.isfinite(water) and water > 0.0 else None
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        side = 1.0 if cfg["look"] == "starboard" else -1.0
        broadside = side * np.array([-axis[1], axis[0]])
        if self._origin is None:
            self._origin, self._t0 = origin.copy(), record.transmit.time_ns
        t = (record.transmit.time_ns - self._t0) * 1e-9
        local = []
        extra = [None] * len(hits)             # what elevation adds to each detection's report
        for i, h in enumerate(hits):
            p = origin[:2] - self._origin[:2] + h.range * (np.sin(h.bearing) * axis + np.cos(h.bearing) * broadside)
            depth = depth_sigma = float("nan")
            e = els[i]
            if el_cfg is not None:
                extra[i] = {"elevation_deg": None}
            if e is not None and e.valid:
                u = np.asarray(e.world)
                level = float(np.arcsin(np.clip(-u[2], -1.0, 1.0)))       # elevation above the horizontal, roll and pitch applied
                extra[i] = {"elevation_deg": float(np.degrees(level)), "elevation_sigma_deg": float(np.degrees(e.sigma)),
                            "elevation_quality": e.quality, "elevation_ambiguous": e.ambiguous}
                if el_cfg["correct_bearing"]:
                    # The line array gave the cone angle; with the elevation the horizontal bearing follows.
                    p = origin[:2] - self._origin[:2] + e.range * u[:2]
                    extra[i].update(cone_bearing_deg=float(np.degrees(h.bearing)),
                                    bearing_deg=float(np.degrees(np.arctan2(u[:2] @ axis, u[:2] @ broadside))))
                if el_cfg["depth"]:
                    image = float(origin[2] + e.range * u[2])             # straight-line depth: outside the water for a bounce
                    sigma_z = float(e.range * np.cos(level) * e.sigma)
                    folded, path = (elevation.fold_depth(image, water, el_cfg["surface_z"]) if el_cfg["fold"]
                                    else (image, "direct"))
                    extra[i].update(depth_m=folded, depth_sigma_m=sigma_z, image_depth_m=image, path=path)
                    if not e.ambiguous:
                        depth, depth_sigma = image, sigma_z
            # The tracker's measurement is (range, bearing) about the frame origin, across then forward.
            local.append(detect.Detection(float(np.hypot(*p)), float(np.arctan2(p[1], p[0])), h.snr_db, h.cells, depth, depth_sigma))
        tracks = self.tracker.update(t, local)
        c = cfg["classify"]
        own = origin[:2] - self._origin[:2]
        mp = dict(c["multipath"], bearing=np.radians(c["multipath"]["bearing_deg"]), origin=(own[1], own[0])) if c.get("multipath") else None
        classes = classify.classify_tracks(tracks, min_speed=c["min_speed"], static_speed=c["static_speed"],
                                           max_fit_rms=c["max_fit_rms"], multipath=mp)
        track_extra = {}
        if el_cfg is not None and el_cfg["depth"]:
            by_id = {tr.id: tr for tr in tracks}
            for c_ in classes:
                tr = by_id[c_.id]
                if not np.isfinite(tr.depth):
                    track_extra[c_.id] = {"depth_m": None}
                    continue
                sigma_z = float(np.sqrt(tr.depth_var))
                folded, path = (elevation.fold_depth(tr.depth, water, el_cfg["surface_z"]) if el_cfg["fold"] else (tr.depth, "direct"))
                track_extra[c_.id] = {"depth_m": folded, "depth_sigma_m": sigma_z, "path": path}
                if el_cfg["classify"] and elevation.outside_water(tr.depth, sigma_z, water, el_cfg["surface_z"], el_cfg["classify_sigma"]):
                    c_.kind = "multipath"      # its straight-line depth is above the surface or below the bottom: a bounce
        report = {
            "ping": index, "sequence": int(record.metadata.sequence), "time_ns": int(record.transmit.time_ns),
            "ownship_xyz": [float(v) for v in origin], "pings_without_nav": self.pings_without_nav,
            "sync": sync_facts,
            "detections": [dict({"range_m": h.range, "bearing_deg": float(np.degrees(h.bearing)), "snr_db": h.snr_db}, **(x or {}))
                           for h, x in zip(hits, extra)],
            "tracks": [{"id": c_.id, "kind": c_.kind, "parent": c_.parent,
                        # World coordinates: x north, y east, metres, in the frame the records use.
                        "x": float(c_.position[1] + self._origin[0]), "y": float(c_.position[0] + self._origin[1]),
                        "vx": c_.velocity[1], "vy": c_.velocity[0], "speed_m_s": c_.speed,
                        "range_m": float(np.hypot(c_.position[1] - own[0], c_.position[0] - own[1])),
                        "mean_snr_db": c_.mean_snr_db, "seen": c_.seen, **track_extra.get(c_.id, {})} for c_ in classes],
            "processing_seconds": time.perf_counter() - t_wall,
        }
        if el_cfg is not None:
            report["rows"] = len(rows)
        if tap:
            tap.json("report", report)
        self.last = {"power": power, "ranges": ranges, "hits": hits, "classes": classes}
        return self._emit(report)

    def _emit(self, report: dict) -> dict:
        line = json.dumps(report)
        if self._report:
            self._report.write(line + "\n")
            self._report.flush()
        if self._udp:
            sock, addr = self._udp
            try:
                sock.sendto(line.encode()[:65000], addr)
            except OSError:
                pass                           # nobody listening is not a reason to stop
        return report

    def run(self, source, max_pings: int | None = None) -> list[dict]:
        """Process every record a source yields."""
        reports = []
        for record in source:
            reports.append(self.process(record))
            if max_pings is not None and len(reports) >= max_pings:
                break
        return reports

    def close(self) -> None:
        if self._writer:
            self._writer.close()
        if self._report:
            self._report.close()
        if self._udp:
            self._udp[0].close()
        if self.nav is not None:
            self.nav.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
