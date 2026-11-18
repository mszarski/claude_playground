"""Forward-looking (and flank) fan imaging: range x bearing maps from one or a few pings.

The work is done by the shared backprojection kernel. This module builds the polar
grid in front of the array, chooses aperture weights and reports grating lobes.

Units and conventions: ranges are one-way metres from the mean position of the receive
elements; bearings are radians from the element boresight, positive towards increasing
element index; points are in the world frame (north, east, down). Images keep the units of
the samples (no source level enters here). `prepare` converts to complex baseband with the
band the transmit waveform states; it makes no allowance for Doppler unless
`max_range_rate` or `doppler_allowance_hz` is passed through to `core.to_baseband`
(docs/HARDCODED_AUDIT.md K-6): a long tone and a moving target need it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .. import _gpusonar as core


def array_frame(record: "core.PingRecord"):
    """(origin, forward, across, up_normal) of the receive array in the world frame.

    Forward is the element boresight; across is the direction the elements run along.
    """
    pos = record.rx_positions()
    origin = pos.mean(axis=0)
    r = core.rotation_ned(*record.geometry.rx_pose.rpy)
    forward = r @ np.asarray(record.geometry.rx[0].boresight)
    # Principal axis of the element positions.
    centred = pos - origin
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    across = vt[0]
    across = across - forward * (across @ forward)
    across /= np.linalg.norm(across)
    # Keep `across` pointing the way the element index increases.
    if (pos[-1] - pos[0]) @ across < 0:
        across = -across
    return origin, forward, across, np.cross(forward, across)


def polar_points(origin, forward, across, ranges, bearings_rad) -> np.ndarray:
    """[n_range, n_bearing, 3] points: origin + r (cos b forward + sin b across)."""
    r = np.asarray(ranges, dtype=float)[:, None, None]
    b = np.asarray(bearings_rad, dtype=float)[None, :, None]
    return np.asarray(origin) + r * (np.cos(b) * np.asarray(forward) + np.sin(b) * np.asarray(across))


def aperture_weights(record: "core.PingRecord", window: str = "rect",
                     full_positions: np.ndarray | None = None) -> np.ndarray:
    """Per-channel aperture weights. The window is laid across the full aperture
    (`full_positions`, e.g. both halves of a muxed array), not just this ping's part."""
    pos = record.rx_positions()
    if window == "rect":
        return np.ones(pos.shape[0], dtype=np.float32)
    ref = pos if full_positions is None else np.asarray(full_positions)
    centre = ref.mean(axis=0)
    _, _, vt = np.linalg.svd(ref - centre, full_matrices=False)
    u = (pos - centre) @ vt[0]
    span = np.ptp((ref - centre) @ vt[0])
    pitch = span / max(len(ref) - 1, 1)
    x = u / (span + pitch)  # in (-0.5, 0.5); the window reaches zero half a pitch beyond the ends
    if window == "hann":
        w = np.cos(np.pi * x) ** 2
    elif window == "hamming":
        w = 0.54 + 0.46 * np.cos(2 * np.pi * x)
    else:
        raise ValueError(f"unknown aperture window {window!r}")
    return w.astype(np.float32)


@dataclass
class FanImage:
    image: np.ndarray       # [n_range, n_bearing] complex
    ranges: np.ndarray      # m
    bearings: np.ndarray    # rad
    points: np.ndarray      # [n_range, n_bearing, 3] world


def _state(record) -> dict:
    return json.loads(record.metadata.extra_json or "{}").get("gpusonar", {})


def _set_state(record, **flags) -> None:
    extra = json.loads(record.metadata.extra_json or "{}")
    extra.setdefault("gpusonar", {}).update(flags)
    meta = record.metadata
    meta.extra_json = json.dumps(extra)
    record.metadata = meta


def ensure_baseband(records, **kwargs):
    """Convert real passband records to complex baseband in place; leave others alone."""
    for r in records:
        if not r.samples.is_complex():
            core.to_baseband(r, **kwargs)
    return records


def prepare(records, *, compress: bool = True, window: str = "none", backend=core.Backend.Host,
            device: int = 0, **baseband_kwargs):
    """Make records ready for backprojection, in place: complex baseband, pulse compressed.

    Each record remembers that it has been compressed (in metadata.extra_json), so
    calling this twice is harmless. With the CUDA backend the transforms run on the
    device (cuFFT); the records still come back to host memory.
    """
    ensure_baseband(records, backend=backend, device=device, **baseband_kwargs)
    if compress:
        for r in records:
            if not _state(r).get("compressed"):
                core.pulse_compress(r, window=window, backend=backend, device=device)
                _set_state(r, compressed=True, replica_window=window)
    return records


def fan_image(records, ranges, bearings_rad, *, interp: str = "sinc", window: str = "rect",
              fov_factor: float = 0.0, directivity_weight: bool = False,
              stop_and_hop: bool = False, compress: bool = True, replica_window: str = "none",
              backend=core.Backend.Host) -> FanImage:
    """Image one or more pings (e.g. both mux halves) onto a range x bearing fan.

    The fan is centred on the mean position of all receive elements used and lies in the
    plane of the array axis and the boresight.
    """
    records = list(records)
    prepare(records, compress=compress, window=replica_window, backend=backend)
    all_pos = np.vstack([r.rx_positions() for r in records])
    _, forward, _, _ = array_frame(records[0])
    origin = all_pos.mean(axis=0)
    centred = all_pos - origin
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    across = vt[0] - forward * (vt[0] @ forward)
    across /= np.linalg.norm(across)
    first = records[0].rx_positions()
    if (first[-1] - first[0]) @ across < 0:
        across = -across
    pts = polar_points(origin, forward, across, ranges, bearings_rad)
    weights = [aperture_weights(r, window, all_pos).tolist() for r in records]
    flat = core.backproject(records, pts.reshape(-1, 3), interp=interp, stop_and_hop=stop_and_hop,
                            fov_factor=fov_factor, directivity_weight=directivity_weight,
                            weights=weights, backend=backend)
    return FanImage(flat.reshape(pts.shape[:2]), np.asarray(ranges, dtype=float),
                    np.asarray(bearings_rad, dtype=float), pts)


def grating_lobes(pitch: float, wavelength: float, steer_rad: float,
                  element_beamwidth: float = float("nan")) -> list[dict]:
    """Grating lobes of a uniform line array steered to `steer_rad`.

    Returns one entry per visible lobe: its angle and its level relative to the main
    lobe in dB, set by the element pattern (one way). Narrowband: a wide pulse smears
    the lobes and lowers their peak further.
    """
    def element(theta):
        if np.isnan(element_beamwidth):
            return 1.0
        return abs(np.sinc(0.886 * np.sin(theta) / element_beamwidth))

    lobes = []
    main = element(steer_rad)
    m_max = int(np.floor(2.0 * pitch / wavelength)) + 1
    for m in range(-m_max, m_max + 1):
        if m == 0:
            continue
        s = np.sin(steer_rad) + m * wavelength / pitch
        if abs(s) <= 1.0:
            theta = float(np.arcsin(s))
            level = element(theta) / main if main > 0 else np.inf
            lobes.append({"order": m, "angle": theta,
                          "level_db": float(20 * np.log10(max(level, 1e-12)))})
    return lobes


def max_unambiguous_steer(pitch: float, wavelength: float) -> float:
    """Largest steering angle (rad) with no grating lobe in visible space: sin(theta) < lambda/d - 1."""
    s = wavelength / pitch - 1.0
    if s >= 1.0:
        return np.pi / 2
    if s <= 0.0:
        return 0.0
    return float(np.arcsin(s))
