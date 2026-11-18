"""Stages wrapped as Holoscan operators, so pipelines can be composed for live use.

Each operator is a thin shell around a library function of this package: the processing
is the same code the tests exercise without Holoscan. Messages between operators are
Python objects (ping records, images, detection lists) passed by reference.

Requires the Holoscan SDK Python package (`holoscan`). Importing this module without it
raises ImportError; nothing else in gpusonar depends on it.
"""

from __future__ import annotations

import time

import numpy as np
from holoscan.core import Application, Operator, OperatorSpec

from . import _gpusonar as core
from . import detect, flank, recon


class PingSourceOp(Operator):
    """Emits ping records from any gpusonar source (file replay, simulator, capture adapter).

    With `realtime=True` each record is released no earlier than its transmit time
    relative to the first one, so a replay runs at the pace of the original collection.
    Attach a CountCondition to stop after a number of pings.
    """

    def __init__(self, fragment, *args, source=None, realtime: bool = False, **kwargs):
        self._iter = iter(source)
        self._realtime = realtime
        self._t0_wall = None
        self._t0_ping = None
        self.emitted = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("ping")

    def compute(self, op_input, op_output, context):
        try:
            record = next(self._iter)
        except StopIteration:
            return
        if self._realtime:
            now = time.perf_counter()
            if self._t0_wall is None:
                self._t0_wall, self._t0_ping = now, record.transmit.time_ns
            due = self._t0_wall + (record.transmit.time_ns - self._t0_ping) * 1e-9
            if due > now:
                time.sleep(due - now)
        self.emitted += 1
        op_output.emit({"record": record, "t_source": time.perf_counter()}, "ping")


class ConditionOp(Operator):
    """Conditioning and pulse compression: calibration, baseband conversion, matched
    filter, optional deterministic TVG."""

    def __init__(self, fragment, *args, compress: bool = True, replica_window: str = "none",
                 tvg_db_per_decade: float | None = None, **kwargs):
        self._compress = compress
        self._window = replica_window
        self._tvg = tvg_db_per_decade
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("ping")
        spec.output("ping")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("ping")
        record = msg["record"]
        recon.prepare([record], compress=self._compress, window=self._window)
        if any(g != 1.0 for g in record.calibration.gain) or any(record.calibration.phase):
            core.apply_calibration(record)
        if self._tvg is not None:
            core.apply_tvg(record, spreading_db_per_decade=self._tvg)
        op_output.emit(msg, "ping")


class FanImageOp(Operator):
    """Backprojection of each ping (or group of `pings_per_image` pings, e.g. both mux
    halves) onto a range x bearing fan: forward-looking and flank imaging."""

    def __init__(self, fragment, *args, ranges=None, bearings_rad=None, pings_per_image: int = 1,
                 interp: str = "linear", fov_factor: float = 0.0, backend=None, **kwargs):
        self._ranges = np.asarray(ranges, dtype=float)
        self._bearings = np.asarray(bearings_rad, dtype=float)
        self._group = int(pings_per_image)
        self._interp = interp
        self._fov = fov_factor
        self._backend = backend if backend is not None else (
            core.Backend.Cuda if core.cuda_available() else core.Backend.Host)
        self._pending = []
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("ping")
        spec.output("image")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("ping")
        self._pending.append(msg)
        if len(self._pending) < self._group:
            return
        group, self._pending = self._pending, []
        fan = recon.fan_image([m["record"] for m in group], self._ranges, self._bearings,
                              interp=self._interp, fov_factor=self._fov, backend=self._backend)
        op_output.emit({"image": fan.image, "ranges": fan.ranges, "bearings": fan.bearings,
                        "record": group[-1]["record"], "t_source": group[0]["t_source"]}, "image")


class DetectOp(Operator):
    """CFAR detection and clustering on a range x bearing image."""

    def __init__(self, fragment, *args, pfa: float | None = None, guard: int | None = None, train: int | None = None, **kwargs):
        # One default for the image detector, the library's (libs/detect CfarOptions: 1e-4, 4 guard and 32 training
        # cells, for a map sampled at about its range resolution; core.cfar_window gives the cells for another map).
        # This operator used to say 1e-6 by itself. The live chain's own setting is `chain.cfar.pfa` (1e-5).
        self._pfa = core.CFAR_DEFAULT_PFA if pfa is None else pfa
        self._guard = core.CFAR_DEFAULT_GUARD if guard is None else guard
        self._train = core.CFAR_DEFAULT_TRAIN if train is None else train
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("image")
        spec.output("detections")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("image")
        power = np.abs(msg["image"].T) ** 2            # detect.* expects [bearing, range]
        record = msg["record"]
        # Leave out the tail of the record that has no full matched-filter overlap.
        c = record.environment.sound_speed
        t_end = (record.samples.start_time + record.samples.n_samples / record.samples.sample_rate
                 - record.transmit.waveform.duration)
        valid = msg["ranges"] <= 0.5 * c * t_end
        power, ranges = power[:, valid], msg["ranges"][valid]
        hits = detect.cfar(power, pfa=self._pfa, guard=self._guard, train=self._train)
        found = detect.cluster(hits, power, ranges, msg["bearings"], self._guard, self._train)
        msg = dict(msg)
        msg["detections"] = found
        op_output.emit(msg, "detections")


class DisplayPrepOp(Operator):
    """Post-processing for display: dynamic range compression (ASASIN Eq. 11-12)."""

    def __init__(self, fragment, *args, target_median: float = 0.2, **kwargs):
        self._b = target_median
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("image")
        spec.output("image")

    def compute(self, op_input, op_output, context):
        msg = dict(op_input.receive("image"))
        msg["display"], msg["drc_q"] = core.drc(np.abs(msg["image"]).astype(np.float32), target_median=self._b)
        op_output.emit(msg, "image")


class CollectOp(Operator):
    """Terminal operator: keeps what arrives and how long each item took from source to here."""

    def __init__(self, fragment, *args, port: str = "in", **kwargs):
        self.items = []
        self.latency = []
        self._port = port
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self._port)

    def compute(self, op_input, op_output, context):
        msg = op_input.receive(self._port)
        self.latency.append(time.perf_counter() - msg["t_source"])
        self.items.append(msg)


class FanPipeline(Application):
    """source -> condition -> fan image -> [detect] -> collect.

    The same graph serves the forward-looking sonar (two mux pings per image, display
    output) and the flank array (one ping per map, detections).
    """

    def __init__(self, source, n_pings: int, ranges, bearings_rad, pings_per_image: int = 1,
                 detect_targets: bool = False, realtime: bool = False, interp: str = "linear",
                 backend=None):
        self._cfg = dict(source=source, n_pings=n_pings, ranges=ranges, bearings=bearings_rad,
                         group=pings_per_image, detect=detect_targets, realtime=realtime,
                         interp=interp, backend=backend)
        self.sink = None
        self.source_op = None
        super().__init__()

    def compose(self):
        from holoscan.conditions import CountCondition
        c = self._cfg
        src = PingSourceOp(self, CountCondition(self, c["n_pings"]), source=c["source"],
                           realtime=c["realtime"], name="source")
        cond = ConditionOp(self, name="condition")
        img = FanImageOp(self, ranges=c["ranges"], bearings_rad=c["bearings"], pings_per_image=c["group"],
                         interp=c["interp"], backend=c["backend"], name="fan_image")
        self.source_op = src
        self.add_flow(src, cond, {("ping", "ping")})
        self.add_flow(cond, img, {("ping", "ping")})
        if c["detect"]:
            det = DetectOp(self, name="detect")
            self.sink = CollectOp(self, port="detections", name="collect")
            self.add_flow(img, det, {("image", "image")})
            self.add_flow(det, self.sink, {("detections", "detections")})
        else:
            disp = DisplayPrepOp(self, name="display_prep")
            self.sink = CollectOp(self, port="image", name="collect")
            self.add_flow(img, disp, {("image", "image")})
            self.add_flow(disp, self.sink, {("image", "image")})
