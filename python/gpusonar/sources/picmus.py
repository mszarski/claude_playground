"""PICMUS plane-wave challenge data (Liebgott et al., IEEE IUS 2016) as ping records.

Reads the zea-format HDF5 conversion published at huggingface.co/datasets/zeahub/picmus
(IQ files). Each plane-wave transmit becomes one record with a plane-wave transmit
description, so the data runs through the same backprojection kernel as everything else.

Frame mapping: the probe's lateral axis becomes body y (across), depth becomes body x
(forward), matching the FLS configs.
"""

from __future__ import annotations

import json
from typing import Iterator

import h5py

try:  # the zea files use compression filters that ship in this package
    import hdf5plugin  # noqa: F401
except ImportError:  # pragma: no cover
    hdf5plugin = None
import numpy as np

from .. import _gpusonar as core
from .base import Source, register


@register
class PicmusSource(Source):
    type_name = "picmus"

    def __init__(self, path: str, track: str = "track_0", frame: int = 0):
        self._file = h5py.File(str(path), "r")
        t = self._file["tracks"][track]
        scan = t["scan"]
        self._raw = t["data"]["raw_data"]            # [frames, tx, samples, elements, 2]
        self._frame = frame
        self._angles = np.asarray(scan["polar_angles"], dtype=float)
        self._t0 = np.asarray(scan["initial_times"], dtype=float)
        self._origins = np.asarray(scan["transmit_origins"], dtype=float)
        # zea time starts when the first element fires. Our transmit instant is when the
        # plane wave crosses the transmit origin (the array centre), which is later by the
        # firing delay there: for a symmetric array, the mean of the element delays.
        self._t_origin = np.asarray(scan["t0_delays"], dtype=float).mean(axis=1)
        self._fs = float(scan["sampling_frequency"][()])
        self._fd = float(scan["demodulation_frequency"][()])
        self._fc = float(scan["center_frequency"][()])
        self._c = float(scan["sound_speed"][()])
        probe = self._file["probe"]
        self._geometry = np.asarray(probe["probe_geometry"], dtype=float)   # probe x, y, z
        self._element_width = float(probe["element_width"][()])
        self._element_height = float(probe["element_height"][()])
        self._bandwidth = float(probe["probe_bandwidth_percent"][()]) / 100.0 * self._fc
        if self._raw.shape[-1] != 2:
            raise ValueError("PicmusSource reads the IQ files (last axis of size 2)")

    @property
    def angles(self) -> np.ndarray:
        return self._angles

    @property
    def sound_speed(self) -> float:
        return self._c

    def __len__(self) -> int:
        return len(self._angles)

    def read(self, index: int) -> core.PingRecord:
        if not 0 <= index < len(self):
            raise IndexError(index)
        iq = np.asarray(self._raw[self._frame, index])            # [samples, elements, 2]
        data = (iq[..., 0] + 1j * iq[..., 1]).T.astype(np.complex64)   # [elements, samples]
        r = core.PingRecord()
        s = r.samples
        s.kind = core.SampleKind.ComplexBaseband
        s.array = data
        s.sample_rate = self._fs
        s.center_frequency = self._fd
        s.start_time = float(self._t0[index] - self._t_origin[index])
        r.samples = s

        wavelength = self._c / self._fc
        w = r.transmit.waveform
        w.type = core.WaveformType.Gaussian
        w.f_start = self._fc - 0.5 * self._bandwidth
        w.f_stop = self._fc + 0.5 * self._bandwidth
        w.duration = 4.0 / self._bandwidth
        t = r.transmit
        t.waveform = w
        t.kind = core.TransmitKind.PlaneWave
        a = float(self._angles[index])
        t.direction = (np.cos(a), np.sin(a), 0.0)
        r.transmit = t

        geo = r.geometry
        elements = []
        for px, py, pz in self._geometry:
            e = core.Element()
            e.lever_arm = (pz, px, py)                 # depth -> forward, lateral -> across
            e.boresight = (1.0, 0.0, 0.0)
            e.beamwidth_az = 0.886 * wavelength / self._element_width
            e.beamwidth_el = 0.886 * wavelength / self._element_height
            elements.append(e)
        geo.rx = elements
        txe = core.Element()
        ox, oy, oz = self._origins[index]
        txe.lever_arm = (oz, ox, oy)
        txe.boresight = (1.0, 0.0, 0.0)
        geo.tx = txe
        r.geometry = geo

        env = r.environment
        env.sound_speed = self._c
        r.environment = env
        r.set_default_channel_tables()
        m = r.metadata
        m.array_id = "picmus_l11_4v"
        m.mode = core.Mode.Fls
        m.sequence = 0
        m.sub_sequence = index
        # The data's time origin already marks the pulse centre: treat it as compressed.
        m.extra_json = json.dumps({"gpusonar": {"compressed": True}, "source": "picmus"})
        r.metadata = m
        return r

    def __iter__(self) -> Iterator[core.PingRecord]:
        for i in range(len(self)):
            yield self.read(i)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
