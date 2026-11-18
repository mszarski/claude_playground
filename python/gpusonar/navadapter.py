"""Navigation adapter: a vendor's navigation stream in, our navigation datagrams out.

The chain takes navigation as JSON datagrams in one fixed convention (live.py `NavListener`,
libs/live `Navigation`): time in ns since the Unix epoch, position north / east / down in
metres, roll / pitch / yaw in radians (the rotation z-y-x from the world's north-east-down
axes to the vehicle's forward-right-down axes), velocity over ground north / east / down in
m/s, and optionally sound speed, altitude above the bottom, bottom lock, water depth and
one-sigma uncertainties.

A real unit speaks its own format. This module converts one vendor record (a dict of named
fields: a CSV line with a header or with listed columns, or a JSON object per line) into one
datagram, with the vendor's side described by a mapping in a YAML file, so that a new unit
needs a mapping and, at most, a small reader for its framing: not a change to the chain.

Mapping (every section optional except `time`; see configs/nav/generic_csv.yaml):

    format: csv | json                # line oriented
    csv: {delimiter: ",", header: true, columns: [...], comment: "#"}
    time: {field: t, unit: s | ms | us | ns, offset_s: 0.0}      # offset_s is added: the unit's clock to
                                                                 # the Unix epoch, and any known latency
    position: {fields: [n, e, d], frame: ned | enu, unit: m,     # enu: east, north, up
               depth: {field: depth, unit: m, positive: down | up}}      # replaces the vertical position
    attitude: {kind: euler | quaternion,
               fields: [roll, pitch, heading] | [w, x, y, z], unit: deg | rad,
               order: zyx | xyz,          # euler: the axes turned about, first to last, intrinsic
               signs: [1, 1, 1],          # multiply each angle (a unit whose pitch is positive nose down: -1)
               world: ned | enu,          # the axes the rotation starts from
               body: frd | flu}           # the vehicle's axes it ends at: forward-right-down or forward-left-up
    velocity: {fields: [a, b, c], frame: ned | enu | body_frd | body_flu, unit: m/s}
    sound_speed: {field: c, unit: m/s}
    altitude: {field: alt, unit: m, lock: {field: status, equals: 1}, invalid: [0, -1]}
    water_depth: {field: wd, unit: m}
    sigma: {roll: {field: ..., unit: deg}, pitch: ..., yaw: ..., position: ..., velocity: ...,
            depth: ..., altitude: ..., water_depth: ..., sound_speed: ...}
    output: {rate_hz: 20.0}           # at most this many datagrams a second (the rest are dropped)

Any `unit` may be a number instead of a name: the factor that turns the vendor's value into
SI (m, m/s, rad, s).

Nothing here knows a particular unit. For the vehicle's SPRINT-Nav the mapping and, if its
output is not line-oriented text, the reader are still to be written from its interface
document: docs/DEPLOYMENT.md lists what it has to supply.
"""

from __future__ import annotations

import json
import socket
import time

import numpy as np

UNITS = {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "km": 1e3, "ft": 0.3048,
         "m/s": 1.0, "cm/s": 1e-2, "mm/s": 1e-3, "kn": 1852.0 / 3600.0,
         "rad": 1.0, "deg": np.pi / 180.0, "mrad": 1e-3,
         "s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
SIGMA_KEYS = ("roll", "pitch", "yaw", "position", "velocity", "depth", "altitude", "water_depth", "sound_speed")
ENU_TO_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])      # also its own inverse
FLU_TO_FRD = np.diag([1.0, -1.0, -1.0])


class MappingError(ValueError):
    """The mapping does not describe a usable stream."""


def _scale(unit, default: str) -> float:
    unit = default if unit is None else unit
    if isinstance(unit, (int, float)):
        return float(unit)
    if unit not in UNITS:
        raise MappingError(f"unknown unit {unit!r} (one of {sorted(UNITS)}, or a number)")
    return UNITS[unit]


def _rot(axis: str, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis].astype(float)


def rotation_from_euler(angles, order: str = "zyx") -> np.ndarray:
    """Body-to-world rotation from intrinsic rotations about the axes in `order`, first to last;
    `angles` are (about x, about y, about z) whatever the order. "zyx" is yaw, then pitch, then roll."""
    about = dict(zip("xyz", angles))
    m = np.eye(3)
    for axis in order:
        m = m @ _rot(axis, about[axis])
    return m


def rotation_from_quaternion(w, x, y, z) -> np.ndarray:
    """Body-to-world rotation of a unit quaternion (scalar first)."""
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def euler_zyx(m: np.ndarray) -> tuple[float, float, float]:
    """(roll, pitch, yaw) of a body-to-world rotation R = Rz(yaw) Ry(pitch) Rx(roll)."""
    pitch = -np.arcsin(np.clip(m[2, 0], -1.0, 1.0))
    return float(np.arctan2(m[2, 1], m[2, 2])), float(pitch), float(np.arctan2(m[1, 0], m[0, 0]))


class Mapping:
    """One vendor record (a dict of fields) -> one of our datagrams (a dict)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg = dict(cfg or {})
        known = {"format", "csv", "time", "position", "attitude", "velocity", "sound_speed", "altitude", "water_depth", "sigma",
                 "output", "unconfirmed", "name"}
        if set(cfg) - known:
            raise MappingError(f"unknown mapping section(s) {sorted(set(cfg) - known)}")
        if "time" not in cfg or "field" not in cfg["time"]:
            raise MappingError("the mapping needs time: {field: ...}")
        self.format = cfg.get("format", "csv")
        if self.format not in ("csv", "json"):
            raise MappingError(f"format: {self.format!r} (csv or json)")
        csv = cfg.get("csv") or {}
        self.delimiter, self.comment = csv.get("delimiter", ","), csv.get("comment", "#")
        self.columns = list(csv["columns"]) if csv.get("columns") else None
        self.header = bool(csv.get("header", self.columns is None))
        a = cfg.get("attitude")
        if a:
            if a.get("kind", "euler") not in ("euler", "quaternion") or a.get("order", "zyx") not in ("zyx", "xyz"):
                raise MappingError("attitude: kind euler | quaternion, order zyx | xyz")
            if a.get("world", "ned") not in ("ned", "enu") or a.get("body", "frd") not in ("frd", "flu"):
                raise MappingError("attitude: world ned | enu, body frd | flu")
            if len(a["fields"]) != (4 if a.get("kind") == "quaternion" else 3):
                raise MappingError("attitude.fields: three angles (roll, pitch, heading) or four quaternion parts (w, x, y, z)")
        for key in ("position", "velocity"):
            if cfg.get(key) and cfg[key].get("fields") and len(cfg[key]["fields"]) != 3:
                raise MappingError(f"{key}.fields: three fields")
        if set(cfg.get("sigma") or {}) - set(SIGMA_KEYS):
            raise MappingError(f"sigma: unknown key(s) {sorted(set(cfg['sigma']) - set(SIGMA_KEYS))}")
        rate = (cfg.get("output") or {}).get("rate_hz")
        self._min_step_ns = int(1e9 / rate) if rate else 0
        self._last_ns = None

    @classmethod
    def load(cls, path) -> "Mapping":
        import yaml
        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    # -- one record --------------------------------------------------------------------------------

    @staticmethod
    def _value(rec: dict, field):
        v = rec.get(field)
        if v is None or v == "":
            return float("nan")
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    def convert(self, rec: dict) -> dict | None:
        """A datagram, or None when the record has no usable time (or `output.rate_hz` drops it)."""
        cfg = self.cfg
        t = self._value(rec, cfg["time"]["field"])
        if not np.isfinite(t):
            return None
        # The stream's time and the offset are rounded to ns separately: a sum in seconds at the Unix epoch keeps 0.2 us.
        time_ns = int(round(t * _scale(cfg["time"].get("unit"), "s") * 1e9)) + int(round(float(cfg["time"].get("offset_s", 0.0)) * 1e9))
        if self._min_step_ns and self._last_ns is not None and 0 <= time_ns - self._last_ns < 0.999 * self._min_step_ns:
            return None
        self._last_ns = time_ns
        out = {"time_ns": time_ns}
        rot = np.eye(3)
        a = cfg.get("attitude")
        if a:
            raw = np.array([self._value(rec, f) for f in a["fields"]])
            if np.all(np.isfinite(raw)):
                if a.get("kind", "euler") == "quaternion":
                    m = rotation_from_quaternion(*raw)
                else:
                    m = rotation_from_euler(raw * _scale(a.get("unit"), "rad") * np.asarray(a.get("signs", [1, 1, 1]), dtype=float),
                                            a.get("order", "zyx"))
                # The unit's rotation takes its body axes to its world axes; ours takes forward-right-down to north-east-down.
                if a.get("world", "ned") == "enu":
                    m = ENU_TO_NED @ m
                if a.get("body", "frd") == "flu":
                    m = m @ FLU_TO_FRD
                rot = m
                out["roll"], out["pitch"], out["yaw"] = euler_zyx(m)
        p = cfg.get("position")
        if p:
            if p.get("fields"):
                v = np.array([self._value(rec, f) for f in p["fields"]]) * _scale(p.get("unit"), "m")
                if p.get("frame", "ned") == "enu":
                    v = ENU_TO_NED @ v
                elif p.get("frame", "ned") != "ned":
                    raise MappingError("position.frame: ned or enu")
                if np.all(np.isfinite(v)):
                    out["x"], out["y"], out["z"] = (float(q) for q in v)
            d = p.get("depth")
            if d:
                z = self._value(rec, d["field"]) * _scale(d.get("unit"), "m") * (1.0 if d.get("positive", "down") == "down" else -1.0)
                if np.isfinite(z):
                    out["z"] = float(z)
        vel = cfg.get("velocity")
        if vel:
            v = np.array([self._value(rec, f) for f in vel["fields"]]) * _scale(vel.get("unit"), "m/s")
            frame = vel.get("frame", "ned")
            if frame == "enu":
                v = ENU_TO_NED @ v
            elif frame == "body_frd":
                v = rot @ v
            elif frame == "body_flu":
                v = rot @ (FLU_TO_FRD @ v)
            elif frame != "ned":
                raise MappingError("velocity.frame: ned, enu, body_frd or body_flu")
            if np.all(np.isfinite(v)) and (frame in ("ned", "enu") or "roll" in out):
                out["vx"], out["vy"], out["vz"] = (float(q) for q in v)
        for key, default in (("sound_speed", "m/s"), ("water_depth", "m")):
            c = cfg.get(key)
            if c:
                v = self._value(rec, c["field"]) * _scale(c.get("unit"), default)
                if np.isfinite(v) and v > 0.0:
                    out[key] = float(v)
        alt = cfg.get("altitude")
        if alt:
            raw = self._value(rec, alt["field"])
            v = raw * _scale(alt.get("unit"), "m")
            lock = np.isfinite(v) and v > 0.0 and raw not in [float(q) for q in alt.get("invalid", [])]
            if alt.get("lock"):
                lock = lock and self._value(rec, alt["lock"]["field"]) == float(alt["lock"].get("equals", 1))
            out["bottom_lock"] = int(bool(lock))
            if lock:
                out["altitude"] = float(v)
        for key, c in (cfg.get("sigma") or {}).items():
            default = "rad" if key in ("roll", "pitch", "yaw") else "m/s" if key in ("velocity", "sound_speed") else "m"
            v = self._value(rec, c["field"]) * _scale(c.get("unit"), default)
            if np.isfinite(v):
                out[f"{key}_sigma"] = float(v)
        return out

    # -- lines ---------------------------------------------------------------------------------------

    def parse_line(self, line: str) -> dict | None:
        """One line of the vendor's stream as a record, or None (a comment, an empty line, the header)."""
        line = line.strip()
        if not line or (self.comment and line.startswith(self.comment)):
            return None
        if self.format == "json":
            try:
                rec = json.loads(line)
            except ValueError:
                return None
            return rec if isinstance(rec, dict) else None
        parts = [p.strip() for p in line.split(self.delimiter)] if self.delimiter.strip() else line.split()
        if self.columns is None:
            if not self.header:
                raise MappingError("csv: give `columns`, or `header: true` for a first line of names")
            self.columns = parts
            return None
        if self.header and parts == self.columns:
            return None
        return dict(zip(self.columns, parts))

    def convert_line(self, line: str) -> dict | None:
        rec = self.parse_line(line)
        return None if rec is None else self.convert(rec)

    def convert_lines(self, lines):
        """Every datagram a stream of lines gives, in order."""
        for line in lines:
            fix = self.convert_line(line)
            if fix is not None:
                yield fix


def replay(mapping: Mapping, lines, send, speed: float = 0.0, sleep=time.sleep) -> int:
    """Replay a file of vendor lines: every datagram to `send(dict)`. `speed` > 0 paces them by their own
    time stamps (1.0: as recorded; 10: ten times faster); 0: as fast as they convert. Returns the count."""
    first_ns = wall0 = None
    n = 0
    for fix in mapping.convert_lines(lines):
        if speed > 0.0:
            if first_ns is None:
                first_ns, wall0 = fix["time_ns"], time.monotonic()
            wait = (fix["time_ns"] - first_ns) * 1e-9 / speed - (time.monotonic() - wall0)
            if wait > 0.0:
                sleep(wait)
        send(fix)
        n += 1
    return n


def udp_sender(host: str, port: int):
    """A `send` for replay() and run(): each datagram as JSON to a UDP port (the chain's nav port)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(fix: dict) -> None:
        sock.sendto(json.dumps(fix).encode(), (host, port))
    send.close = sock.close
    return send


def run_udp(mapping: Mapping, in_port: int, send, in_host: str = "0.0.0.0", stop=None, timeout: float = 0.2) -> int:
    """Live: vendor lines arriving as UDP datagrams (one or more lines each) -> our datagrams. Runs until
    `stop` (a threading.Event) is set. Returns the count."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((in_host, in_port))
    sock.settimeout(timeout)
    n = 0
    try:
        while stop is None or not stop.is_set():
            try:
                data = sock.recv(65535)
            except socket.timeout:
                continue
            for fix in mapping.convert_lines(data.decode(errors="replace").splitlines()):
                send(fix)
                n += 1
    finally:
        sock.close()
    return n
