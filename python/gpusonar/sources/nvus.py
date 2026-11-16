"""Adapter for the NVIDIA Holoscan ultrasound reference (`holoscan-ultrasound-imaging`,
commit 24d405e): raw capture frames to ping records.

This is the piece between the reference's acquisition (which we reuse unchanged) and our
record format. It holds no acquisition logic: it decodes a frame's bytes and attaches
geometry from the sonar config. Layout, per `python/nvus/raw.py` and `recon.unpack` of
the reference (see docs/HOLOSCAN_NOTES.md section 3):

- a frame is a run of 48-byte records with no header;
- a record holds one 24-byte little-endian word per AFE (two AFEs), each word 16 lanes of
  12-bit two's-complement samples, least significant lane first;
- two records make one sample instant: physical channel c of an AFE rides lane c >> 1 of
  the record with parity c & 1, so a sample is 96 bytes for 64 channels;
- capture column = afe * 32 + channel; the element it sees is column (mux setting "odd",
  elements 0-63) or 64 + column ("even", elements 64-127).
"""

from __future__ import annotations

import json

import numpy as np

from .. import _gpusonar as core
from ..sonar import SonarModel

RECORD_BYTES = 48
BYTES_PER_SAMPLE = 2 * RECORD_BYTES
N_CHANNELS = 64
CAPTURE_SPIKE_SAMPLES = 8        # the reference zeroes these at the start of every frame
MUX_FIRST_ELEMENT = {"odd": 0, "even": 64}


def decode_frame(data) -> np.ndarray:
    """Frame bytes to int16 samples [64 capture columns, n_samples]. Trailing bytes that
    do not make a whole sample are ignored."""
    raw = np.frombuffer(data, dtype=np.uint8)
    n = raw.size // BYTES_PER_SAMPLE
    b = raw[: n * BYTES_PER_SAMPLE].reshape(-1, 3).astype(np.int32)
    low = b[:, 0] | ((b[:, 1] & 0x0F) << 8)
    high = (b[:, 1] >> 4) | (b[:, 2] << 4)
    words = np.empty(low.size * 2, dtype=np.int32)
    words[0::2], words[1::2] = low, high
    words = np.where(words >= 2048, words - 4096, words)
    # (sample, record parity, afe, lane) -> column = afe * 32 + lane * 2 + parity
    w = words.reshape(n, 2, 2, 16).transpose(0, 2, 3, 1).reshape(n, N_CHANNELS)
    return np.ascontiguousarray(w.T).astype(np.int16)


def encode_frame(samples: np.ndarray) -> bytes:
    """Inverse of `decode_frame`: int samples [64, n_samples] in -2048..2047 to frame bytes.
    For tests, the emulator and recorded-data fixtures."""
    s = np.asarray(samples)
    if s.shape[0] != N_CHANNELS:
        raise ValueError("expected 64 capture columns")
    if s.min() < -2048 or s.max() > 2047:
        raise ValueError("samples do not fit in 12 bits")
    n = s.shape[1]
    words = (s.T.astype(np.int32) & 0xFFF).reshape(n, 2, 16, 2).transpose(0, 3, 1, 2).reshape(-1)
    low, high = words[0::2], words[1::2]
    out = np.empty((low.size, 3), dtype=np.uint8)
    out[:, 0] = low & 0xFF
    out[:, 1] = (low >> 8) | ((high & 0x0F) << 4)
    out[:, 2] = high >> 4
    return out.tobytes()


def frame_to_record(data, sonar: SonarModel, rx_select: str, *, pose=None, start_time: float = 0.0,
                    time_ns: int = 0, sequence: int = 0, sound_speed: float = 1480.0,
                    zero_capture_spike: bool = True, frame_number: int | None = None) -> "core.PingRecord":
    """One captured frame as a ping record.

    `sonar` is the probe's config (`l12_5d`). `rx_select` is the mux setting the frame was
    captured with, "odd" or "even" in the reference's naming. `start_time` is when the
    capture window opened relative to the transmit, which the reference sets through its
    sync timers; the frame itself carries no timing.
    """
    if rx_select not in MUX_FIRST_ELEMENT:
        raise ValueError('rx_select must be "odd" or "even"')
    counts = decode_frame(data)
    if counts.shape[1] == 0:
        raise ValueError("frame holds no complete sample")
    sub = 0 if rx_select == "odd" else 1
    first = MUX_FIRST_ELEMENT[rx_select]
    if sub >= len(sonar.mux_groups) or sonar.mux_groups[sub] != list(range(first, first + N_CHANNELS)):
        raise ValueError(f"sonar config {sonar.name!r} does not describe the reference's mux halves")
    if pose is None:
        pose = core.Pose()
    record = sonar.make_record(pose, sequence=sequence, sub_sequence=sub, time_ns=time_ns,
                               environment={"sound_speed": sound_speed})
    step = (record.metadata.adc_full_scale if np.isfinite(record.metadata.adc_full_scale) else 1.0) / 2048.0
    x = counts.astype(np.float32) * np.float32(step)
    if zero_capture_spike:
        x[:, :CAPTURE_SPIKE_SAMPLES] = 0.0
    s = record.samples
    s.array = x
    s.start_time = float(start_time)
    s.storage = core.StorageType.Int16
    s.storage_scale = step
    record.samples = s
    meta = record.metadata
    meta.extra_json = json.dumps({"source": "nvus", "rx_select": rx_select, "frame_number": frame_number,
                                  "reference_commit": "24d405e"})
    record.metadata = meta
    return record
