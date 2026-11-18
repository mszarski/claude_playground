"""File replay of .wav recordings with a YAML sidecar.

Covers Popoto modem recordings (one multichannel file or one file per channel)
and generic multichannel .wav. The sidecar supplies what a .wav cannot carry:
element positions, the transmitted chirp and the ping times. Format:
docs/RECORD_FORMAT.md, "WAV sidecar".
"""

from __future__ import annotations

import json
import os
from typing import Iterator

import numpy as np
import yaml
from scipy.io import wavfile

from .. import _gpusonar as core
from ..acoustics import NOMINAL_SOUND_SPEED
from .base import Source, register

_WAVEFORMS = {
    "cw": core.WaveformType.Cw,
    "lfm": core.WaveformType.Lfm,
    "gaussian": core.WaveformType.Gaussian,
}
_MODES = {
    "unknown": core.Mode.Unknown,
    "fls": core.Mode.Fls,
    "sas": core.Mode.Sas,
    "flank": core.Mode.Flank,
}


class SidecarError(ValueError):
    """The sidecar is missing a field or contradicts the audio."""


def _load_wav(path: str) -> tuple[int, np.ndarray, int]:
    """Return (sample_rate, float32 [channels, samples] in [-1, 1), bits)."""
    rate, data = wavfile.read(path, mmap=False)
    if data.ndim == 1:
        data = data[:, None]
    if data.dtype == np.int16:
        bits, out = 16, data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        # scipy returns 24-bit PCM left-justified in int32, so one scale fits both.
        bits, out = 32, (data.astype(np.float64) / 2147483648.0).astype(np.float32)
    elif data.dtype == np.uint8:
        bits, out = 8, (data.astype(np.float32) - 128.0) / 128.0
    elif data.dtype.kind == "f":
        bits, out = 0, data.astype(np.float32)
    else:
        raise SidecarError(f"{path}: unsupported sample format {data.dtype}")
    return int(rate), np.ascontiguousarray(out.T), bits


def _element_table(spec, n_channels: int) -> np.ndarray:
    """Element lever arms [channels, 3] from an explicit list or {count, pitch, axis}."""
    if isinstance(spec, dict):
        count = int(spec.get("count", n_channels))
        pitch = float(spec["pitch"])
        axis = np.asarray(spec.get("axis", [1.0, 0.0, 0.0]), dtype=float)
        axis = axis / np.linalg.norm(axis)
        origin = np.asarray(spec.get("origin", [0.0, 0.0, 0.0]), dtype=float)
        idx = np.arange(count) - (count - 1) / 2.0
        table = origin + idx[:, None] * pitch * axis
    else:
        table = np.asarray(spec, dtype=float)
    if table.ndim != 2 or table.shape[1] != 3:
        raise SidecarError("elements must be a list of [x, y, z] or {count, pitch, axis}")
    if table.shape[0] != n_channels:
        raise SidecarError(
            f"sidecar describes {table.shape[0]} elements but the audio has {n_channels} channels"
        )
    return table


def _ping_times(spec, duration: float) -> np.ndarray:
    if spec is None:
        return np.array([0.0])
    if isinstance(spec, dict):
        first = float(spec.get("first", 0.0))
        interval = float(spec["interval"])
        count = spec.get("count")
        if count is None:
            count = int(np.floor((duration - first) / interval)) + 1
        return first + interval * np.arange(int(count))
    return np.asarray(spec, dtype=float)


@register
class WavSource(Source):
    """Cuts a recording into ping records using the sidecar's ping times."""

    type_name = "wav"

    def __init__(self, sidecar: str):
        self._sidecar_path = str(sidecar)
        with open(self._sidecar_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise SidecarError(f"{sidecar}: sidecar must be a YAML mapping")
        self._cfg = cfg
        base = os.path.dirname(os.path.abspath(self._sidecar_path))

        files = cfg.get("files") or ([cfg["file"]] if "file" in cfg else None)
        if not files:
            raise SidecarError("sidecar needs `file` (multichannel) or `files` (one per channel)")
        blocks, rates, bits = [], set(), 0
        for name in files:
            rate, data, b = _load_wav(os.path.join(base, name))
            blocks.append(data)
            rates.add(rate)
            bits = max(bits, b)
        if len(rates) != 1:
            raise SidecarError(f"audio files disagree on sample rate: {sorted(rates)}")
        lengths = {b.shape[1] for b in blocks}
        if len(lengths) != 1:
            # Per-channel files of unequal length would silently misalign channels.
            raise SidecarError(f"audio files disagree on length: {sorted(lengths)} samples")
        self._rate = float(rates.pop())
        if "sample_rate" in cfg and float(cfg["sample_rate"]) != self._rate:
            raise SidecarError(
                f"sidecar sample_rate {cfg['sample_rate']} disagrees with the audio ({self._rate})"
            )
        self._audio = np.concatenate(blocks, axis=0) * float(cfg.get("scale", 1.0))
        self._bits = int(cfg.get("adc_bits", bits))
        n_channels, n_total = self._audio.shape

        if "elements" not in cfg:
            raise SidecarError("sidecar needs `elements`")
        self._elements = _element_table(cfg["elements"], n_channels)
        if "transmit" not in cfg:
            raise SidecarError("sidecar needs `transmit`")

        pings = cfg.get("pings", {}) or {}
        duration = n_total / self._rate
        self._times = _ping_times(pings.get("times"), duration)
        window = pings.get("window")
        if window is None:
            interval = np.diff(self._times).min() if len(self._times) > 1 else duration
            window = [0.0, float(interval)]
        self._win_start, self._win_len = float(window[0]), float(window[1])
        self._n_win = int(round(self._win_len * self._rate))
        if self._n_win <= 0:
            raise SidecarError("pings.window duration must be positive")
        # Keep only pings whose window lies fully inside the recording.
        first = np.round((self._times + self._win_start) * self._rate).astype(np.int64)
        keep = (first >= 0) & (first + self._n_win <= n_total)
        self._dropped = int((~keep).sum())
        self._times = self._times[keep]
        self._first = first[keep]

    @property
    def dropped_pings(self) -> int:
        """Pings listed in the sidecar whose window ran off the recording."""
        return self._dropped

    def __len__(self) -> int:
        return len(self._times)

    def read(self, index: int) -> core.PingRecord:
        if not 0 <= index < len(self):
            raise IndexError(index)
        cfg = self._cfg
        n_channels = self._audio.shape[0]
        i0 = int(self._first[index])
        r = core.PingRecord()

        s = r.samples
        s.array = self._audio[:, i0 : i0 + self._n_win]
        s.sample_rate = self._rate
        # The window is snapped to whole samples; report where it really starts.
        s.start_time = i0 / self._rate - float(self._times[index])
        tx = cfg["transmit"]
        wf = tx["waveform"]
        w = r.transmit.waveform
        try:
            w.type = _WAVEFORMS[str(wf["type"]).lower()]
        except KeyError as e:
            raise SidecarError(f"unknown waveform type {wf.get('type')!r}") from e
        w.f_start = float(wf["f_start"])
        w.f_stop = float(wf.get("f_stop", wf["f_start"]))
        w.duration = float(wf["duration"])
        w.window_alpha = float(wf.get("window_alpha", 0.0))
        from ..sonar import PEAK_OVER_RMS_DB
        # the sidecar's figure is an RMS level (either key); the record holds the peak level (core/record.hpp)
        w.source_level_db = float(wf.get("source_level_db_rms", wf.get("source_level_db", float("nan")))) + PEAK_OVER_RMS_DB
        w.amplitude = float(wf.get("amplitude", 1.0))
        s.center_frequency = w.center()
        r.samples = s
        t = r.transmit
        t.waveform = w
        start_ns = int(cfg.get("recording_start_ns", 0))
        t.time_ns = start_ns + int(round(float(self._times[index]) * 1e9))
        r.transmit = t

        geo = r.geometry
        pose = core.Pose()
        p = cfg.get("pose", {}) or {}
        pose.position = tuple(p.get("position", (0.0, 0.0, 0.0)))
        pose.rpy = tuple(p.get("rpy", (0.0, 0.0, 0.0)))
        pose.velocity = tuple(p.get("velocity", (0.0, 0.0, 0.0)))
        pose.rates = tuple(p.get("rates", (0.0, 0.0, 0.0)))
        geo.rx_pose = pose
        geo.tx_pose = pose
        boresight = tuple(cfg.get("boresight", (0.0, 1.0, 0.0)))
        bw = cfg.get("beamwidth", {}) or {}
        elements = []
        for row in self._elements:
            e = core.Element()
            e.lever_arm = tuple(row)
            e.boresight = boresight
            e.beamwidth_az = float(bw.get("az", float("nan")))
            e.beamwidth_el = float(bw.get("el", float("nan")))
            elements.append(e)
        geo.rx = elements
        txe = core.Element()
        txe.lever_arm = tuple(tx.get("lever_arm", (0.0, 0.0, 0.0)))
        txe.boresight = tuple(tx.get("boresight", boresight))
        geo.tx = txe
        if "frame_origin_lla" in cfg:
            geo.frame_origin_lla = tuple(cfg["frame_origin_lla"])
        r.geometry = geo

        env_cfg = cfg.get("environment", {}) or {}
        env = r.environment
        env.sound_speed = float(env_cfg.get("sound_speed", NOMINAL_SOUND_SPEED))
        env.temperature = float(env_cfg.get("temperature", float("nan")))
        env.salinity = float(env_cfg.get("salinity", float("nan")))
        env.water_depth = float(env_cfg.get("water_depth", float("nan")))
        r.environment = env

        r.set_default_channel_tables()
        meta = r.metadata
        meta.array_id = str(cfg.get("array_id", "wav"))
        meta.mode = _MODES[str(cfg.get("mode", "unknown")).lower()]
        meta.sequence = index
        meta.adc_bits = self._bits
        meta.adc_full_scale = float(cfg.get("scale", 1.0))
        meta.hardware_gain_db = float(cfg.get("hardware_gain_db", 0.0))
        boards_cfg = cfg.get("boards")
        if boards_cfg:
            boards, channel_board = [], [0] * n_channels
            for b_index, b in enumerate(boards_cfg):
                board = core.Board()
                board.reference_channel = int(b.get("reference_channel", -1))
                boards.append(board)
                for ch in b["channels"]:
                    channel_board[int(ch)] = b_index
            meta.boards = boards
            meta.channel_board = channel_board
        meta.extra_json = json.dumps({"sidecar": os.path.basename(self._sidecar_path)})
        r.metadata = meta
        return r

    def __iter__(self) -> Iterator[core.PingRecord]:
        for i in range(len(self)):
            yield self.read(i)

    def close(self) -> None:
        self._audio = None
