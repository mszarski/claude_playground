"""The long-range array chain, one ping at a time: the part that runs on a vehicle.

    source (boards over UDP, a recording, or the simulator)
      -> record raw pings                      (always: the recording is the flight data)
      -> prepare (baseband, pulse compression)
      -> beamform (range x bearing)
      -> detect (cell-averaging CFAR, clustering)
      -> track in a world-fixed frame (ownship's motion removed)
      -> classify (mover / static / multipath / uncertain)
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
    "beamformer": "backprojection",            # the kernel (host or CUDA); or "farfield" / "mvdr" (NumPy, host)
    "backend": "auto",                         # "auto": CUDA when built and present, else host; "host"; "cuda"
    "look": "starboard",                       # which side of a line array the bearings are on
    "range_cell_m": 2.0,                       # power is averaged into cells this long before detection
    "cfar": {"pfa": 1e-5, "guard_m": 100.0, "train_m": 300.0},
    "min_snr_db": 9.0,
    "tracker": {"range_sigma": 15.0, "bearing_sigma_deg": 1.0, "confirm_hits": 3, "max_misses": 2,
                "initial_speed_sigma": 8.0, "process_noise": 0.05},
    "classify": {"min_speed": 1.0, "static_speed": 1.0, "max_fit_rms": 60.0,
                 "multipath": {"max_excess_range": 800.0, "bearing_deg": 3.0, "velocity": 2.5}},
}


def _merge(base: dict, over: dict | None) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


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
        self.tracker = detect.Tracker(range_sigma=t["range_sigma"], bearing_sigma=np.radians(t["bearing_sigma_deg"]),
                                      confirm_hits=t["confirm_hits"], max_misses=t["max_misses"],
                                      initial_speed_sigma=t["initial_speed_sigma"], process_noise=t["process_noise"])
        b = self.cfg["bearings_deg"]
        self.bearings = np.radians(np.arange(b[0], b[1] + 0.5 * b[2], b[2]))
        self._writer = core.RecordWriter(recording) if recording else None
        self._report = open(report_path, "w", encoding="utf-8") if report_path else None
        self._udp = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM), tuple(report_udp)) if report_udp else None
        self._tap_cfg = taps or {}
        self.nav = nav                         # None: the records already carry the pose (simulation, replay)
        self.pings_without_nav = 0
        want = self.cfg["backend"]
        self.backend = core.Backend.Cuda if want == "cuda" or (want == "auto" and core.cuda_available()) else core.Backend.Host
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
        from . import recon
        recon.prepare([record], backend=self.backend)        # baseband and pulse compression (cuFFT on a GPU)
        if tap:
            tap.records("prepare", [record])
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
            fine = (np.abs(fan.image) ** 2).T                 # [bearing, range]
            fine_ranges = fine_r
        else:
            bm = (flank.mvdr_beamform(record, self.bearings) if cfg["beamformer"] == "mvdr"
                  else flank.farfield_beamform(record, self.bearings, window=cfg["window"])).valid()
            fine, fine_ranges = np.abs(bm.data) ** 2, bm.ranges
            k = max(int(round(cfg["range_cell_m"] / (fine_ranges[1] - fine_ranges[0]))), 1)
        n = fine.shape[1] // k * k
        power = fine[:, :n].reshape(len(self.bearings), -1, k).mean(axis=2)
        ranges = fine_ranges[:n].reshape(-1, k).mean(axis=1)
        cell = ranges[1] - ranges[0]
        guard, train = (max(int(round(cfg["cfar"][key] / cell)), 1) for key in ("guard_m", "train_m"))
        hit_map = detect.cfar(power, pfa=cfg["cfar"]["pfa"], guard=guard, train=train, backend=self.backend)
        hits = detect.cluster(hit_map, power, ranges, self.bearings, guard=guard, train=train)
        hits = [h for h in hits if h.snr_db > cfg["min_snr_db"]]
        if tap:
            tap.arrays("beamform", power=power.astype(np.float32), ranges=ranges, bearings=self.bearings)

        # World position of each detection: the array's own position and heading come with the record.
        rx = record.rx_positions()
        origin = rx.mean(axis=0)
        axis = rx[-1, :2] - rx[0, :2]
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        side = 1.0 if cfg["look"] == "starboard" else -1.0
        broadside = side * np.array([-axis[1], axis[0]])
        if self._origin is None:
            self._origin, self._t0 = origin.copy(), record.transmit.time_ns
        t = (record.transmit.time_ns - self._t0) * 1e-9
        local = []
        for h in hits:
            p = origin[:2] - self._origin[:2] + h.range * (np.sin(h.bearing) * axis + np.cos(h.bearing) * broadside)
            # The tracker's measurement is (range, bearing) about the frame origin, across then forward.
            local.append(detect.Detection(float(np.hypot(*p)), float(np.arctan2(p[1], p[0])), h.snr_db, h.cells))
        tracks = self.tracker.update(t, local)
        c = cfg["classify"]
        own = origin[:2] - self._origin[:2]
        mp = dict(c["multipath"], bearing=np.radians(c["multipath"]["bearing_deg"]), origin=(own[1], own[0])) if c.get("multipath") else None
        classes = classify.classify_tracks(tracks, min_speed=c["min_speed"], static_speed=c["static_speed"],
                                           max_fit_rms=c["max_fit_rms"], multipath=mp)
        report = {
            "ping": index, "sequence": int(record.metadata.sequence), "time_ns": int(record.transmit.time_ns),
            "ownship_xyz": [float(v) for v in origin], "pings_without_nav": self.pings_without_nav,
            "detections": [{"range_m": h.range, "bearing_deg": float(np.degrees(h.bearing)), "snr_db": h.snr_db} for h in hits],
            "tracks": [{"id": c_.id, "kind": c_.kind, "parent": c_.parent,
                        # World coordinates: x north, y east, metres, in the frame the records use.
                        "x": float(c_.position[1] + self._origin[0]), "y": float(c_.position[0] + self._origin[1]),
                        "vx": c_.velocity[1], "vy": c_.velocity[0], "speed_m_s": c_.speed,
                        "range_m": float(np.hypot(c_.position[1] - own[0], c_.position[0] - own[1])),
                        "mean_snr_db": c_.mean_snr_db, "seen": c_.seen} for c_ in classes],
            "processing_seconds": time.perf_counter() - t_wall,
        }
        if tap:
            tap.json("report", report)
        self.last = {"power": power, "ranges": ranges, "hits": hits, "classes": classes}
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
