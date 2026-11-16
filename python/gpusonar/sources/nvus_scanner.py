"""Live source built on the NVIDIA Holoscan ultrasound reference
(`holoscan-ultrasound-imaging` at commit 24d405e): the reference's own native runtime
programs the scan sequence and acquires; this module turns each observation into ping
records. No reference code is changed and nothing here touches the FPGA.

The scan is the forward-looking one this suite needs from the rig: one unsteered plane-wave
transmit per multiplexer half ("odd" then "even"), each with a receive window as deep as
asked. Upstream, the reference's sequence compiler stops at 65,536 samples (0.88 m of
range at 50 MHz). This repository carries a small patch to its host C++
(`third_party/patches/`, applied by `tools/build_nvus_reference.sh`) that raises the limit
to 262,144 samples (3.5 m), so the 2 to 3 m pool window stays on the fast native path.
The patch touches the sequence compiler and one runtime check; no FPGA code. A 21 MB frame
per transmit has not been tried on hardware (docs/HOLOSCAN_NOTES.md 9.1).

Hardware: by default the reference's `FakeHardware` (no device, placeholder frames), which
exercises sequence compilation, the acquisition loop and the hand-over. On the rig, build
the reference with its SDK (`./setup`, Sensor Bridge 2.7.0 and Holoscan 4.4.0 as it pins)
and pass the hardware object it makes (`nvus._scanner.make_hololink_hardware(...)`, as in
its `examples/scanner_cpp_live.py`). That path has not been run: there is no rig here.

The reference's Python package must be importable as `nvus` with its native module built.
Set GPUSONAR_NVUS_PYTHON to the directory that holds the `nvus` package if it is not
installed.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Iterator

import numpy as np

from .. import _gpusonar as core
from ..sonar import SonarModel
from .base import Source, register
from . import nvus as frames

REFERENCE_COMMIT = "24d405e"


class ReferenceUnavailable(RuntimeError):
    """The reference's `nvus` package or its native module cannot be imported."""


def load_reference():
    """The reference's native module (`nvus._scanner`)."""
    path = os.environ.get("GPUSONAR_NVUS_PYTHON")
    if path and path not in sys.path:
        sys.path.insert(0, path)
    try:
        from nvus import _scanner
    except Exception as exc:                                  # ImportError, or a broken build
        raise ReferenceUnavailable(
            "the reference's native module is not importable (build holoscan-ultrasound-imaging "
            f"at {REFERENCE_COMMIT} and set GPUSONAR_NVUS_PYTHON): {type(exc).__name__}: {exc}") from exc
    return _scanner


def compile_scan(native, depth_m: float, interval_us: float, sound_speed: float):
    """One 0-degree plane-wave transmit per multiplexer half, through the reference's compiler."""
    cfg = native.PlaneWaveConfig()
    cfg.depth_mm = 1e3 * float(depth_m)
    cfg.interval_us = float(interval_us)
    cfg.sound_speed_m_s = float(sound_speed)
    return native.compile_plane_waves([0.0], cfg)


def max_depth(native=None, sound_speed: float = 1480.0, interval_us: float = 20000.0) -> float:
    """The deepest window (m) the reference's sequence compiler accepts, found by bisection."""
    native = native or load_reference()
    lo, hi = 0.04, 8.0
    while hi - lo > 1e-4:
        mid = 0.5 * (lo + hi)
        try:
            compile_scan(native, mid, interval_us, sound_speed)
            lo = mid
        except ValueError:
            hi = mid
    return lo


@register
class NvusScannerSource(Source):
    """Ping records from the reference rig (or its simulated hardware).

    `frame_hook(event_index, rx_select, data)`, when given, may return replacement frame
    bytes; tests and dry runs use it to put simulated echoes in place of the simulated
    hardware's placeholder content.
    """

    type_name = "nvus_scanner"

    def __init__(self, sonar: str = "l12_5d", depth_m: float | None = None, interval_us: float = 20000.0,
                 sound_speed: float = 1480.0, repetitions: int = 1, hardware=None, frame_hook=None,
                 timeout_ms: int = 5000, pose=None):
        self.native = load_reference()
        self.sound_speed = float(sound_speed)
        self.depth_m = float(depth_m) if depth_m is not None else max_depth(self.native, sound_speed, interval_us)
        try:
            self.action = compile_scan(self.native, self.depth_m, interval_us, sound_speed)
        except ValueError as exc:
            raise ValueError(
                f"the reference's sequence compiler refused a {self.depth_m:.3f} m window ({exc}); its limit is "
                f"{max_depth(self.native, sound_speed, max(interval_us, 20000.0)):.3f} m") from exc
        self.prepared = self.action.sequence
        self.manifest = json.loads(self.prepared.programming_manifest)
        # The transmit is a plane wave along the boresight, not the point source of the
        # stock config.
        self.sonar = SonarModel.load(sonar, {"transmit": {"kind": "plane_wave"}})
        fs = float(self.manifest["sample_rate_hz"])
        if abs(fs - float(self.sonar.config["receive"]["sample_rate"])) > 1e-6 * fs:
            raise ValueError(f"sonar config {sonar!r} samples at {self.sonar.config['receive']['sample_rate']} Hz, "
                             f"the reference sequence at {fs} Hz")
        # Capture opens `rx_start_ticks` after the sync edge and the pulser fires
        # `tx_delay_ticks` after it, so the window opens before the transmit.
        self.start_time = (float(self.manifest["rx_start_ticks"]) - float(self.manifest["tx_delay_ticks"])) \
            / float(self.manifest["sync_clock_hz"])
        self.repetitions = int(repetitions)
        self.hardware = hardware if hardware is not None else self.native.FakeHardware()
        self.simulated = hardware is None
        self.frame_hook = frame_hook
        self.timeout_ms = int(timeout_ms)
        self.pose = pose
        self.incomplete_observations = 0

    @property
    def events(self) -> list:
        """(event_id, 'odd' or 'even', samples) of the compiled sequence."""
        return [(e.event_id, "odd" if e.rx_side == 0 else "even", e.samples) for e in self.prepared.events]

    def __iter__(self) -> Iterator["core.PingRecord"]:
        options = self.native.RunOptions()
        options.repetitions = self.repetitions
        run = self.native.Scanner(self.hardware).run(self.prepared, options)
        sequence = 0
        try:
            for _ in range(self.repetitions):
                obs = run.next_observation(self.timeout_ms)
                if obs is None:
                    return
                if not obs.complete or obs.error:
                    self.incomplete_observations += 1
                    continue
                for index, (rec, buffer) in enumerate(zip(obs.records, obs.buffers)):
                    rx_select = "odd" if rec.rx_side == 0 else "even"
                    data = buffer.to_bytes()
                    if len(data) != rec.samples * frames.BYTES_PER_SAMPLE:
                        raise ValueError("observation byte count differs from its receive geometry")
                    if self.frame_hook is not None:
                        data = self.frame_hook(index, rx_select, data) or data
                    record = frames.frame_to_record(
                        data, self.sonar, rx_select, pose=self.pose, start_time=self.start_time,
                        time_ns=int(rec.timestamp_ns), sequence=sequence, sound_speed=self.sound_speed,
                        frame_number=int(obs.id))
                    meta = record.metadata
                    extra = json.loads(meta.extra_json)
                    extra.update(simulated_hardware=self.simulated, observation=int(obs.id), run=int(obs.run_id),
                                 event_id=int(rec.event_id), sequence_id=self.prepared.id,
                                 config_id=self.prepared.config_id,
                                 receive_attenuation_code=int(self.manifest["receive_attenuation_code"]),
                                 timestamp_origin=str(rec.timestamp_origin))
                    meta.extra_json = json.dumps(extra)
                    record.metadata = meta
                    sequence += 1
                    yield record
        finally:
            run.stop()
            run.wait()
