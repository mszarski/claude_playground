"""Reverberation by the scattering-function method (after SST; docs/PHYSICS_NOTES.md).

For long ranges, placing individual scatterers over kilometres of boundary is too
expensive. Instead the reverberation *intensity* is computed against time and arrival
direction, and a Gaussian time series with that structure is generated for the array:

1. The boundary is divided into range rings. For each ring the eigenray provider gives
   the one-way paths (delay, amplitude, grazing angle) from the sonar.
2. Every pair of outbound and return paths contributes power at time tau_out + tau_back:
   |a_out|^2 |a_back|^2 x scattering strength x ring area. Pairs add incoherently, as in
   SST. This is the reverberation power against time, per unit azimuth.
3. For each azimuth sector an independent complex Gaussian sequence with that
   time-varying variance is drawn: one random "cell" per time sample and direction.
4. Each hydrophone receives every sector's sequence with its plane-wave lead for that
   direction (an exact delay, applied in frequency), and the result is convolved with the
   transmitted pulse.

The output is raw (uncompressed) reverberation, complex baseband or real passband to
match the record, added to the record, so it goes through pulse compression and
beamforming like any echo and keeps the right coherence across the array.

Only paths travelling towards the boundary count as incident (a receiver placed on a
Bellhop boundary also sees the reflected arrival; Lambert's law is defined on the incident
intensity).

What the default (`geometry="monostatic"`) models: one provider call from the array centre
serves both legs (the transmitter is taken to be at the array centre and to move with it);
the transmit pattern and the pattern of the first receive element weight each azimuth
sector and each path, at the path's own elevation at the sonar (its launch direction, not
its grazing angle at the boundary); each sector is shifted by the Doppler of the receive
platform's horizontal velocity towards the sector's azimuth, two-way, as if every path
were horizontal (no elevation factor, no vertical velocity); the arrival elevation is
ignored when steering to the hydrophones (fine at the low grazing angles of long ranges);
the boundary is flat and uniform.

`geometry="bistatic"` (reverb_bistatic.py) lifts those: separate transmitter and receiver
positions and velocities, the scattering strength at the true grazing angles and bistatic
azimuth, the Doppler of each platform along its own ray, the arrival elevation in the
steering, and a boundary strength that may vary over the boundary. The generator then
works on components (azimuth sector x elevation bin x bin of total Doppler shift), each
with its own direction and Doppler shift, instead of horizontal azimuth sectors.

`statistics={"model": "k", "shape": nu}` makes the reverberation K-distributed: the power of
each resolution cell (a cell of time and azimuth) is multiplied by a gamma variate of unit
mean and shape nu before the Gaussian speckle is drawn, so the mean power is unchanged.

Limits that remain: the boundary is flat and the medium horizontally stratified; eigenray
pairs add incoherently; there is no range walk within a ping; each ping draws new noise,
so this method cannot support synthetic aperture processing; the point-scatterer method
(scene.seafloor_patch) is the one for imaging modes.

Steps 3 and 4 (the generation of the time series, most of the cost) also exist in C++
(libs/sim/src/reverb.cpp, with a CUDA backend for the sum over sectors), which is used by
default; `engine="numpy"` or GPUSONAR_SIM_ENGINE=numpy selects the NumPy code in this file
(sim/engine.py). Random numbers come from the caller's NumPy generator on both engines,
in the same order, so a seed gives the same reverberation whichever engine runs: the C++
result differs from the NumPy one only by single-precision rounding.
"""

from __future__ import annotations

import numpy as np

import warnings

from .. import _gpusonar as core
from . import engine as _engine
from . import reverb_bistatic, reverb_relief
from .bathymetry import make_bathymetry
from .propagation import EigenrayProvider


def baseband_pulse(waveform, sample_rate: float, center_frequency: float) -> np.ndarray:
    """The transmitted pulse as complex baseband samples (unit amplitude)."""
    if waveform.type == core.WaveformType.Arbitrary:
        # A sampled pulse: the envelope of the record's replica, resampled (sim/waveform.py).
        from .waveform import SampledWaveform
        return SampledWaveform.from_record(waveform).baseband(sample_rate, center_frequency)
    n = int(np.floor(waveform.duration * sample_rate)) + 1
    u = np.arange(n) / sample_rate
    t, f0, f1 = waveform.duration, waveform.f_start, waveform.f_stop
    alpha = waveform.window_alpha
    x = u / t
    env = np.ones(n)
    if alpha > 0:
        edge = 0.5 * alpha
        env = np.where(x < edge, 0.5 * (1 - np.cos(np.pi * x / edge)),
                       np.where(x > 1 - edge, 0.5 * (1 - np.cos(np.pi * (1 - x) / edge)), 1.0))
    if waveform.type == core.WaveformType.Cw:
        phase = f0 * u
    elif waveform.type == core.WaveformType.Lfm:
        phase = (f0 + 0.5 * (f1 - f0) / t * u) * u
    elif waveform.type == core.WaveformType.Gaussian:
        sigma = np.sqrt(2 * np.log(2)) / (np.pi * (f1 - f0))
        env = np.exp(-0.5 * ((u - 0.5 * t) / sigma) ** 2)
        phase = 0.5 * (f0 + f1) * (u - 0.5 * t)
    else:
        raise ValueError("unsupported waveform for reverberation synthesis")
    return env * np.exp(2j * np.pi * (phase - center_frequency * u))


def reverberation_power(provider: EigenrayProvider, source, boundary_z: float, times: np.ndarray,
                        mu_db: float, absorption_db_per_m: float = 0.0, max_range: float | None = None,
                        ring_width: float | None = None, min_range: float = 0.0,
                        strength=None, pattern=None) -> np.ndarray:
    """Reverberation power per radian of azimuth in each time bin, for unit source
    amplitude at 1 m, from a flat boundary at depth `boundary_z`.

    With `pattern` = (azimuths [M], transmit_gain, receive_gain), each a function of unit
    direction vectors [..., 3] in the world frame returning a power gain, the result is
    [M, n_time]: the power per radian arriving from each azimuth, weighted by the two beam
    patterns along each path's own elevation.

    The scattering strength is Lambert's law with parameter `mu_db`, or `strength`: a
    function of the incident and scattered grazing angles (radians) returning the linear
    scattering strength per unit area.

    `times` are the centres of uniform time bins (two-way time after transmit).
    """
    source = np.asarray(source, dtype=float)
    dt = times[1] - times[0]
    c = provider.sound_speed
    r_max = max_range or 0.5 * c * (times[-1] + dt)
    dr = ring_width or 0.25 * c * dt                       # two rings per time bin at grazing incidence
    rings = np.arange(min_range + 0.5 * dr, r_max, dr)
    power = np.zeros(len(times))
    if len(rings) == 0:
        return power
    pts = np.column_stack([source[0] + rings, np.full_like(rings, source[1]), np.full_like(rings, boundary_z)])
    es = provider.eigenrays(source, pts)
    if strength is None:
        mu = 10.0 ** (mu_db / 10.0)
        strength = lambda g_in, g_out: mu * np.sin(g_in) * np.sin(g_out)     # noqa: E731
    side = 1.0 if boundary_z >= source[2] else -1.0        # incident paths travel towards the boundary
    incident = es.valid & (es.arrival[:, :, 2] * side > 0.0)
    graze = np.arcsin(np.clip(np.abs(es.arrival[:, :, 2]), 0.0, 1.0))   # grazing angle at the boundary
    a2 = np.where(incident, np.abs(es.amplitude) ** 2, 0.0)
    k = es.delay.shape[1]
    area = rings * dr                                      # ring area per radian of azimuth
    if pattern is not None:
        azimuths, tx_gain, rx_gain = pattern
        ca, sa = np.cos(azimuths)[:, None, None], np.sin(azimuths)[:, None, None]
        lx, ly, lz = es.launch[None, :, :, 0], es.launch[None, :, :, 1], es.launch[None, :, :, 2]
        # Each path leaves (and, by reciprocity, returns) along its launch direction,
        # turned to the sector's azimuth.
        d = np.stack([lx * ca - ly * sa, lx * sa + ly * ca, np.broadcast_to(lz, (len(azimuths),) + lz.shape[1:])], axis=-1)
        w_tx, w_rx = tx_gain(d), rx_gain(d)                # [M, rings, k]
        out = np.zeros((len(azimuths), len(times)))
        cols = np.arange(len(azimuths))[:, None]
        for i in range(k):
            for j in range(k):
                t = es.delay[:, i] + es.delay[:, j]
                p = a2[:, i] * a2[:, j] * strength(graze[:, i], graze[:, j]) * area
                if absorption_db_per_m:
                    p = p * 10.0 ** (-absorption_db_per_m * c * t / 10.0)
                idx = np.rint((t - times[0]) / dt).astype(int)
                ok = (idx >= 0) & (idx < len(times)) & (p > 0)
                if not ok.any():
                    continue
                np.add.at(out, (cols, idx[ok][None, :]), p[ok][None, :] * w_tx[:, ok, i] * w_rx[:, ok, j])
        return out
    for i in range(k):
        for j in range(k):
            t = es.delay[:, i] + es.delay[:, j]
            p = a2[:, i] * a2[:, j] * strength(graze[:, i], graze[:, j]) * area
            if absorption_db_per_m:
                p = p * 10.0 ** (-absorption_db_per_m * c * t / 10.0)
            idx = np.rint((t - times[0]) / dt).astype(int)
            ok = (idx >= 0) & (idx < len(times)) & (p > 0)
            np.add.at(power, idx[ok], p[ok])
    return power


def volume_layer_strength(sv_db: float, thickness: float):
    """A volume-scattering layer as SST treats it (TR 0702 Sec. 6.5): collapsed onto a sheet at
    its centre depth. The sheet's scattering strength per unit area is the volume scattering
    strength (per m, `sv_db` in dB re 1/m) times the layer's thickness, whatever the angles."""
    areal = 10.0 ** (float(sv_db) / 10.0) * float(thickness)
    return lambda g_in, g_out: np.full(np.broadcast(np.asarray(g_in), np.asarray(g_out)).shape, areal)


def chapman_harris_strength(wind_speed: float, frequency: float):
    """Sea-surface scattering strength function for `reverberation_power`. Chapman-Harris is
    a backscattering fit; for a path pair with different grazing angles it is evaluated at
    their mean."""
    from .. import acoustics
    return lambda g_in, g_out: 10.0 ** (acoustics.chapman_harris_db(0.5 * (g_in + g_out), wind_speed,
                                                                    frequency) / 10.0)


def element_power_pattern(element, rotation: np.ndarray, frequency_ratio: float = 1.0):
    """Power gain of a record's element (or transmitter) towards world directions [..., 3]:
    the same separable sinc model as the simulator and the kernel (core/directivity.hpp).
    Returns None for an omnidirectional element."""
    bw_az, bw_el = float(element.beamwidth_az), float(element.beamwidth_el)
    if not (np.isfinite(bw_az) or np.isfinite(bw_el)):
        return None
    b = rotation @ np.asarray(element.boresight, dtype=float)
    b /= np.linalg.norm(b)
    ref = rotation @ np.array([0.0, 0.0, 1.0])
    if abs(ref @ b) > 0.999:
        ref = rotation @ np.array([1.0, 0.0, 0.0])
        az = ref - b * (ref @ b)
        az /= np.linalg.norm(az)
        el = np.cross(b, az)
    else:
        el = ref - b * (ref @ b)
        el /= np.linalg.norm(el)
        az = np.cross(el, b)

    def gain(d):
        front = d @ b
        g = np.ones(d.shape[:-1])
        if np.isfinite(bw_az):
            g = g * np.sinc(0.886 * frequency_ratio * (d @ az) / bw_az)
        if np.isfinite(bw_el):
            g = g * np.sinc(0.886 * frequency_ratio * (d @ el) / bw_el)
        return np.where(front > 0.0, g * g, 0.0)

    return gain


def _pattern_key(element):
    """What distinguishes one element's beam pattern from another's."""
    widths = tuple(None if not np.isfinite(w) else round(float(w), 9) for w in (element.beamwidth_az, element.beamwidth_el))
    return (tuple(round(float(v), 9) for v in element.boresight) if any(w is not None for w in widths) else None, widths)


def default_azimuth_count(positions: np.ndarray, wavelength: float) -> int:
    """Two sectors per beamwidth (lambda / extent) of the array, and at least 72."""
    extent = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))
    return int(max(72, np.ceil(4.0 * np.pi * extent / wavelength)))


# Random numbers handed to the C++ generator in one call, bytes (a run of sectors).
SECTOR_RUN_BYTES = 64 << 20


def taper_pulse(pulse: np.ndarray, n: int) -> np.ndarray:
    """The pulse with its spectrum rolled off to zero at the Nyquist frequency: unchanged
    below half of it, a raised cosine from there up. Returned as `n` samples of one period
    (what precedes the pulse's first sample is at the end), so that its transform over `n`
    points is the tapered spectrum exactly."""
    if len(pulse) > n:
        return pulse
    f = np.abs(np.fft.fftfreq(n))                                  # cycles per sample, 0 .. 0.5
    window = np.where(f <= 0.25, 1.0, 0.5 * (1.0 + np.cos(np.pi * (f - 0.25) / 0.25)))
    return np.fft.ifft(np.fft.fft(pulse, n) * window)


class CellTexture:
    """The texture of K-distributed reverberation: a gamma variate of unit mean and shape
    `shape` for each resolution cell, as the factor on the amplitude (its square root) of
    one row's sequence at the reduced rate.

    A cell is `cell_samples` samples long and covers the rows of one group (`group_of_row`:
    the sectors, or the components, that lie in one azimuth cell). The variates come from a
    generator of their own, seeded by one draw from the caller's, so both engines consume
    the caller's generator alike. Rows must be asked for in increasing order of group.

    Intensity = texture x exponential speckle has mean 1 x the cell's power, and
    E{I^2} / E{I}^2 = 2 (1 + 1 / shape): the K distribution's value, 2 (Rayleigh) as
    shape -> infinity.
    """

    def __init__(self, rng, shape: float, n_low: int, cell_samples: int, group_of_row):
        if not shape > 0.0:
            raise ValueError("K-distributed reverberation needs a positive shape")
        self.rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        self.shape, self.n_low, self.cell = float(shape), int(n_low), max(int(cell_samples), 1)
        self.group_of_row = np.asarray(group_of_row)
        self.group, self.row = None, None

    def __call__(self, row: int) -> np.ndarray:
        group = int(self.group_of_row[row])
        if group != self.group:
            n_cells = -(-self.n_low // self.cell)
            tau = self.rng.gamma(self.shape, 1.0 / self.shape, n_cells)
            self.row = np.sqrt(np.repeat(tau, self.cell)[:self.n_low])
            self.group = group
        return self.row


def cpp_has_components() -> bool:
    """True when the built extension's reverberation generator takes components (a
    direction and a Doppler shift per row); an older build has azimuth sectors only."""
    return "direction" in (getattr(core.sim_reverb_accumulate, "__doc__", "") or "")


def _series_cpp(rng, coarse, density, same_in_azimuth, pulse_low, offsets, fs, n, start_time, time0, fc, c,
                n_azimuth, decim, n_low, velocity, carrier, real, backend, components=None, texture=None):
    """The reverberation of every channel in the record's window, [channels, n], by the C++
    generator. The random numbers are drawn here, in the order `_series_numpy` draws them
    (for each sector with power: n_low for the real part, then n_low for the imaginary
    part), in runs of sectors so that no more than SECTOR_RUN_BYTES of them are held at once.

    With `components` = (direction [rows, 3], doppler [rows]) each row of the density is a
    component with that arrival direction and frequency shift. `texture` (CellTexture)
    scales each row's deviates."""
    density = np.ascontiguousarray(density[:1] if same_in_azimuth else density, dtype=float)
    coarse = np.ascontiguousarray(coarse, dtype=float)
    extra = {}
    if components is not None:
        extra = dict(direction=np.ascontiguousarray(components[0], dtype=float),
                     doppler=np.ascontiguousarray(components[1], dtype=float))
        active = core.sim_reverb_active(coarse, density, fs, time0, n_azimuth, decim, n_low, components=True)
        rows = np.flatnonzero(active)
    else:
        # A sector without power draws nothing (as in `_series_numpy`).
        active = core.sim_reverb_active(coarse, density, fs, time0, n_azimuth, decim, n_low)
        rows = (np.arange(n_azimuth) if active[0] else np.arange(0)) if len(active) == 1 else np.flatnonzero(active)
    n_active = len(rows)
    offsets = np.ascontiguousarray(offsets, dtype=float)
    pulse_low = np.ascontiguousarray(pulse_low, dtype=complex)
    velocity = tuple(float(v) for v in velocity)
    total = np.zeros((offsets.shape[0], n_low), dtype=np.complex64)
    run = max(1, SECTOR_RUN_BYTES // (16 * n_low))
    for first in range(0, n_active, run):
        normals = rng.standard_normal((min(run, n_active - first), 2, n_low))
        if texture is not None:
            for k in range(normals.shape[0]):
                normals[k] *= texture(rows[first + k])[None, :]
        core.sim_reverb_accumulate(total, coarse, density, first, normals, pulse_low, offsets, fs, time0, fc, c,
                                   n_azimuth, decim, velocity=velocity, backend=backend, **extra)
    return core.sim_reverb_finish(total, fs, n, start_time, time0, fc, decim, carrier=carrier, real_output=real)


def _series_numpy(rng, coarse, density, pulse_low, positions, centre, fs, n, start_time, times, fc, c,
                  n_azimuth, decim, n_low, velocity, carrier, real, components=None, texture=None):
    """The reverberation of every channel in the record's window, [channels, n], in NumPy.
    `components` and `texture` as in `_series_cpp`."""
    fs_low = fs / decim
    t_low = times[0] + np.arange(n_low) / fs_low
    dphi = 2.0 * np.pi / n_azimuth
    azimuths = (np.arange(n_azimuth) + 0.5) * dphi
    f = np.fft.fftfreq(n_low, 1.0 / fs_low)
    total = np.zeros((positions.shape[0], n_low), dtype=np.complex64)
    offsets = positions - centre
    w = (2.0 * np.pi * (f + fc)).astype(np.float32)
    # The boundary is fixed in the world: a moving sonar sees each azimuth's reverberation
    # shifted by the Doppler of its own speed towards that azimuth (two-way).
    velocity = np.asarray(velocity, dtype=float)
    # A uniform line array needs one complex exponential per sector, not one per channel.
    along = offsets - offsets[0]
    uniform = len(offsets) > 2 and np.allclose(along[1:], np.arange(1, len(offsets))[:, None] * along[1], atol=1e-9)
    pulse_spectrum = np.fft.fft(pulse_low, n_low)
    for m in range(n_azimuth if components is None else len(density)):
        sigma = np.sqrt(np.interp(t_low, coarse, density[m], left=0.0, right=0.0) / fs_low * dphi / 2.0)
        if not sigma.any():
            continue
        b = sigma * (rng.standard_normal(n_low) + 1j * rng.standard_normal(n_low))
        if texture is not None:
            b = b * texture(m)
        if components is None:
            u = np.array([np.cos(azimuths[m]), np.sin(azimuths[m]), 0.0])
            doppler = 2.0 * (velocity @ u) * fc / c
        else:
            # A component has its own arrival direction (with elevation) and Doppler shift.
            u = np.asarray(components[0][m], dtype=float)
            doppler = float(components[1][m])
        if doppler != 0.0:
            # Every echo from this azimuth is the pulse shifted in frequency: shift the
            # scatterer sequence and the pulse together.
            b = b * np.exp(2j * np.pi * doppler * (t_low - t_low[0]))
            shifted = pulse_low * np.exp(2j * np.pi * doppler * np.arange(len(pulse_low)) / fs_low)
            spectrum = (np.fft.fft(b) * np.fft.fft(shifted, n_low)).astype(np.complex64)
        else:
            spectrum = (np.fft.fft(b) * pulse_spectrum).astype(np.complex64)
        lead = ((offsets @ u) / c).astype(np.float32)        # [channels]
        if uniform:
            row = spectrum * np.exp(1j * (w * lead[0])).astype(np.complex64)
            stride = np.exp(1j * (w * (lead[1] - lead[0]))).astype(np.complex64)
            for ch in range(len(offsets)):
                total[ch] += row
                row = row * stride
        else:
            arg = lead[:, None] * w[None, :]
            total += spectrum[None, :] * (np.cos(arg) + 1j * np.sin(arg))
    # Band-limited interpolation to the record's rate: zero-pad the spectrum.
    n_full = n_low * decim
    if decim > 1:
        padded = np.zeros((positions.shape[0], n_full), dtype=np.complex64)
        half = n_low // 2
        padded[:, :half] = total[:, :half]
        padded[:, n_full - (n_low - half):] = total[:, half:]
        series = np.fft.ifft(padded, axis=1) * decim
    else:
        series = np.fft.ifft(total, axis=1)
    # Sample k of the convolution is at times[0] + k / fs; the record starts one pulse later.
    start = int(round((start_time - times[0]) * fs))
    out = series[:, start:start + n]
    if carrier:
        t_abs = start_time + np.arange(n) / fs
        out = out * np.exp(2j * np.pi * fc * t_abs)[None, :]
    if real:
        out = np.real(out)
    return out


def add_reverberation(record, provider: EigenrayProvider, boundary_z: float, mu_db: float,
                      rng: np.random.Generator, n_azimuth: int | None = None,
                      absorption_db_per_m: float = 0.0, max_range: float | None = None,
                      power_step: float | None = None, strength=None, engine: str | None = None,
                      backend=core.Backend.Host, geometry: str = "monostatic", n_elevation: int | None = None,
                      strength_map=None, azimuth_subsamples: int = 1, statistics: dict | None = None,
                      guard: float = 0.0, band_taper: bool = False, bathymetry=None,
                      scatterer_motion: dict | None = None, shadowing: bool = True,
                      bottom_bounces: str = "exclude", max_doppler_bins: int = 128) -> np.ndarray:
    """Add boundary reverberation to a raw record (complex baseband or real passband), in place.

    Returns the reverberation power per radian in each sample of the impulse response
    (for the processing log and for tests).

    `engine` ("cpp" or "numpy"; default: sim/engine.py) chooses the implementation that
    generates the time series; `backend` is where the C++ one sums the sectors (host or
    CUDA). `rng` is advanced identically by both.

    `geometry="bistatic"` takes the transmitter's and the receiver's own positions and
    velocities from the record (reverb_bistatic.py; see the header). With it, `n_elevation`
    is the number of bins of arrival elevation (default 9: 20 degrees each), `max_doppler_bins`
    (128) the most bins of total Doppler shift the components are keyed by (they are half the
    pulse's resolution 1 / T wide, or wider when that would take more bins), `strength_map`
    a function of world (x, y) giving a linear factor on the boundary's strength, and
    `azimuth_subsamples` the azimuths evaluated within a sector. `strength` may then take
    the bistatic azimuth as a third argument.

    `statistics` = {"model": "k", "shape": nu, "cell_time": s, "cell_azimuth_deg": degrees}
    draws K-distributed reverberation with shape nu; a cell is by default 1 / bandwidth
    long and one nominal beamwidth (wavelength / array extent) wide. {"model": "rayleigh"}
    or None is the Gaussian default.

    `bathymetry` (sim/bathymetry.py: a Bathymetry, a function depth(x, y) or a mapping) is a
    seabed with relief in place of the flat boundary at `boundary_z` (reverb_relief.py:
    cells at their own depth, true facet area, angles to the facet, shadowing unless
    `shadowing=False`). Paths with a bottom bounce before the cell are left out, or with
    `bottom_bounces="flat"` taken as the provider's flat-bottom paths (an approximation). An
    exactly level bathymetry runs the flat code at that depth. Needs `geometry="bistatic"`.

    `scatterer_motion` = {"velocity": [vx, vy, vz], "rms_speed": m/s, "rms_vertical_speed":
    m/s, "wind_speed": m/s} moves the scatterers (needs `geometry="bistatic"`): `velocity`
    is their mean velocity in the world (a drifting surface, a current carrying a volume
    layer) and shifts each cell by TR 0702 eq. (72); `rms_speed` is random motion on every
    axis and `rms_vertical_speed` on the vertical only (`wind_speed` sets the latter to the
    rms vertical speed of the wind sea, acoustics.rms_surface_vertical_speed), which spread
    each component's Doppler into a Gaussian of standard deviation
    (f / c) sqrt(rms_speed^2 |k|^2 + rms_vertical_speed^2 k_z^2), k = a_in + a_back the sum
    of the two legs' directions of travel on arrival at the cell (|k| = 2 for backscatter).
    The spread is realised as Doppler lines closer than the pulse can resolve
    (reverb_bistatic.doppler_lines), each an independent sequence.

    `band_taper` and `guard` deal with leakage along the record. The generator works in
    frequency at a reduced rate: the delays to the hydrophones and the interpolation to the
    record's rate are ideal (sinc) operations, whose response to one sample falls only as
    1 / distance wherever the spectrum is not zero at the reduced rate's Nyquist frequency
    (a CW pulse's sidelobes are about 28 dB down there). The strong start of a record
    therefore leaks into its weak far end, forwards and, the series being periodic,
    backwards round the period: visible when the power falls by more than about 50 dB
    across the record, as it does over a long-range window. `band_taper=True` rolls the
    pulse's spectrum off smoothly to zero between half the Nyquist frequency and Nyquist
    (outside the band the rate was chosen for), which removes the 1 / distance tail, and
    appends two pulse lengths of silence to the generated series to keep what is left of
    the wrap-round out of the record; `guard` (seconds) sets that silence directly. The
    defaults (off, 0) keep earlier results.
    """
    if geometry not in ("monostatic", "bistatic"):
        raise ValueError(f"unknown reverberation geometry {geometry!r}: use 'monostatic' or 'bistatic'")
    bistatic = geometry == "bistatic"
    if not bistatic and (strength_map is not None or n_elevation is not None or azimuth_subsamples != 1
                         or bathymetry is not None or scatterer_motion):
        raise ValueError("strength_map, n_elevation, azimuth_subsamples, bathymetry and scatterer_motion need "
                         "geometry='bistatic'")
    motion = dict(scatterer_motion or {})
    unknown = set(motion) - {"velocity", "rms_speed", "rms_vertical_speed", "wind_speed"}
    if unknown:
        raise ValueError(f"unknown scatterer_motion keys {sorted(unknown)}")
    rms_vertical = float(motion.get("rms_vertical_speed", 0.0))
    if "wind_speed" in motion and "rms_vertical_speed" not in motion:
        from .. import acoustics
        rms_vertical = acoustics.rms_surface_vertical_speed(float(motion["wind_speed"]))
    motion_kw = dict(scatterer_velocity=motion.get("velocity"), scatterer_rms_speed=float(motion.get("rms_speed", 0.0)),
                     scatterer_rms_vertical_speed=rms_vertical)
    if motion_kw["scatterer_rms_speed"] < 0.0 or rms_vertical < 0.0:
        raise ValueError("scatterer_motion: rms speeds must not be negative")
    s = record.samples
    wf = record.transmit.waveform
    real = s.kind == core.SampleKind.RealPassband
    carrier = s.kind != core.SampleKind.ComplexBaseband
    fs, n = s.sample_rate, s.n_samples
    fc = 0.5 * (wf.f_start + wf.f_stop) if carrier else s.center_frequency
    c = record.environment.sound_speed
    pulse = baseband_pulse(wf, fs, fc) * wf.amplitude
    # Impulse-response time axis: echoes of scatterers at two-way time t occupy [t, t + T].
    times = s.start_time - wf.duration + np.arange(n + len(pulse)) / fs
    positions = record.rx_positions()
    centre = positions.mean(axis=0)
    n_azimuth = n_azimuth or default_azimuth_count(positions, c / fc)
    # Power on a coarse time grid (it varies slowly), as a density, then per sample.
    step = power_step or max(1.0 / fs, min(0.25 * wf.duration, 2e-3))
    coarse = np.arange(max(times[0], 0.0), times[-1] + step, step)
    if len(coarse) < 2:
        coarse = np.array([times[0], times[0] + step])
    min_range = 0.6 * 0.5 * c * coarse[0] if coarse[0] > 0.2 else 0.0
    dphi = 2.0 * np.pi / n_azimuth
    azimuths = (np.arange(n_azimuth) + 0.5) * dphi
    # Beam patterns, when the sonar has them: the transmit beam and the element pattern
    # shape the reverberation in azimuth and select paths by elevation.
    rotation = np.asarray(core.rotation_ned(*record.geometry.rx_pose.rpy))
    tx_rotation = np.asarray(core.rotation_ned(*record.geometry.tx_pose.rpy))
    tx_gain = element_power_pattern(record.geometry.tx, tx_rotation)
    rx_gain = element_power_pattern(record.geometry.rx[0], rotation)
    if len({_pattern_key(e) for e in record.geometry.rx}) > 1:
        # One sequence per direction is shared by all channels, with a delay each: a gain per
        # channel and direction would have to enter the steering kernel (host and CUDA).
        warnings.warn("reverberation: the receive elements have different beam patterns; the first element's "
                      "pattern is used for all of them", RuntimeWarning, stacklevel=2)
    rx_velocity = np.asarray(record.geometry.rx_pose.velocity, dtype=float)
    tx_velocity = np.asarray(record.geometry.tx_pose.velocity, dtype=float)
    components = sector_of_row = None
    if bistatic:
        tx_position = np.asarray(record.tx_position(), dtype=float)
        separation = float(np.hypot(*(tx_position[:2] - centre[:2])))
        if coarse[0] > 0.2:
            min_range = max(0.6 * 0.5 * c * coarse[0] - 0.5 * separation, 0.0)
        common = dict(frequency=fc, mu_db=mu_db, strength=strength, tx_velocity=tx_velocity, rx_velocity=rx_velocity,
                      tx_gain=tx_gain, rx_gain=rx_gain, absorption_db_per_m=absorption_db_per_m, max_range=max_range,
                      n_elevation=9 if n_elevation is None else int(n_elevation), strength_map=strength_map,
                      subsamples=int(azimuth_subsamples), doppler_resolution=0.5 / wf.duration,
                      max_doppler_bins=int(max_doppler_bins))
        if any(v is not None and np.any(v) for v in motion_kw.values()):
            common.update(motion_kw)
        bed = None if bathymetry is None else make_bathymetry(bathymetry)
        if bed is not None and bed.flat_depth is not None:
            boundary_z, bed = bed.flat_depth, None                         # exactly level: the flat code
        if bed is not None:
            found = reverb_relief.directional_power(provider, tx_position, centre, bed, coarse, azimuths,
                                                    min_range=0.0, shadowing=shadowing,
                                                    bottom_bounces=bottom_bounces, **common)
        else:
            found = reverb_bistatic.directional_power(provider, tx_position, centre, boundary_z, coarse, azimuths,
                                                      min_range=min_range, **common)
        by_direction = found["power"] / step                              # [sectors, elevations, coarse]
        mean_density = by_direction.sum(axis=1).mean(axis=0)
        # The components with power, in order of sector then elevation.
        flat = by_direction.reshape(-1, len(coarse))
        rows = np.flatnonzero(flat.any(axis=1))
        if len(rows) == 0:
            rows = np.array([0])
        density = flat[rows]
        components = (found["direction"].reshape(-1, 3)[rows], found["doppler"].reshape(-1)[rows])
        sector_of_row = rows // by_direction.shape[1]
        spread = found["spread"].reshape(-1)[rows]
        doppler_extent = 0.0
        if spread.any():
            # Random scatterer motion: each component becomes Doppler lines about its mean
            # shift, with the component's power shared among them.
            lines = [reverb_bistatic.doppler_lines(float(sp), wf.duration) for sp in spread]
            count = np.array([len(w) for _, w in lines])
            index = np.repeat(np.arange(len(rows)), count)
            offsets, weights = np.concatenate([o for o, _ in lines]), np.concatenate([w for _, w in lines])
            density = density[index] * weights[:, None]
            components = (components[0][index], components[1][index] + offsets)
            sector_of_row = sector_of_row[index]
            doppler_extent = float(np.abs(offsets).max())
        if motion_kw["scatterer_velocity"] is not None:
            doppler_extent += 2.0 * float(np.linalg.norm(motion_kw["scatterer_velocity"])) * fc / c
    elif tx_gain is None and rx_gain is None:
        density = reverberation_power(provider, centre, boundary_z, coarse, mu_db, absorption_db_per_m,
                                      max_range, min_range=min_range, strength=strength) / step
        density = np.broadcast_to(density, (n_azimuth, len(coarse)))
    else:
        one = lambda d: np.ones(d.shape[:-1])                # noqa: E731
        density = reverberation_power(provider, centre, boundary_z, coarse, mu_db, absorption_db_per_m,
                                      max_range, min_range=min_range, strength=strength,
                                      pattern=(azimuths, tx_gain or one, rx_gain or one)) / step
    if not bistatic:
        mean_density = density.mean(axis=0)
    power = np.interp(times, coarse, mean_density, left=0.0, right=0.0) / fs
    # Reverberation occupies the pulse's band only, so it is generated at a reduced rate
    # (at least four times the bandwidth) and interpolated to the record's rate.
    # A moving sonar shifts each azimuth's reverberation by up to 2 v f / c either way.
    speed = float(np.linalg.norm(record.geometry.rx_pose.velocity))
    if bistatic:
        speed = 0.5 * (speed + float(np.linalg.norm(tx_velocity)))       # each platform shifts by v f / c
    band = max(abs(wf.f_stop - wf.f_start), 4.0 / wf.duration) + 4.0 * speed * fc / c
    if bistatic and doppler_extent > 0.0:
        band += 2.0 * doppler_extent                                       # the scatterers' own shifts, either way
    decim = max(int(fs // (4.0 * band)), 1)
    fs_low = fs / decim
    n_low = int(np.ceil(len(times) / decim))
    if guard < 0.0:
        raise ValueError("reverberation guard must not be negative")
    if band_taper:
        guard = max(guard, 2.0 * wf.duration)
    if guard > 0.0:
        n_low += int(np.ceil(guard * fs_low))                              # no power there: beyond the coarse grid
    pulse_low = baseband_pulse(wf, fs_low, fc) * wf.amplitude
    if band_taper:
        pulse_low = taper_pulse(pulse_low, n_low)
    start = int(round((s.start_time - times[0]) * fs))
    texture = None
    model = (statistics or {}).get("model", "rayleigh")
    if model not in ("rayleigh", "gaussian", "k"):
        raise ValueError(f"unknown reverberation statistics {model!r}: use 'rayleigh' or 'k'")
    if model == "k":
        if "shape" not in statistics:
            raise ValueError("K-distributed reverberation needs statistics.shape")
        shape = float(statistics["shape"])
        if np.isfinite(shape):                                             # infinite shape is Rayleigh
            cell_time = float(statistics.get("cell_time") or 1.0 / max(abs(wf.f_stop - wf.f_start), 1.0 / wf.duration))
            extent = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
            if statistics.get("cell_azimuth_deg") is not None:
                cell_azimuth = np.radians(float(statistics["cell_azimuth_deg"]))
            else:
                cell_azimuth = (c / fc) / extent if extent > 0.0 else dphi
            per_cell = max(int(round(cell_azimuth / dphi)), 1)             # sectors in one azimuth cell
            sectors = np.arange(n_azimuth) if sector_of_row is None else sector_of_row
            texture = CellTexture(rng, shape, n_low, int(round(cell_time * fs_low)), sectors // per_cell)
    chosen = _engine.resolve(engine)
    if chosen == "cpp" and components is not None and not cpp_has_components():
        warnings.warn("bistatic reverberation: this build of the C++ generator has azimuth sectors only; "
                      "falling back to the NumPy engine", RuntimeWarning, stacklevel=2)
        chosen = "numpy"
    if chosen == "cpp":
        out = _series_cpp(rng, coarse, density, not bistatic and tx_gain is None and rx_gain is None, pulse_low,
                          positions - centre, fs, n, s.start_time, times[0], fc, c, n_azimuth, decim, n_low,
                          record.geometry.rx_pose.velocity, carrier, real, backend, components, texture)
    else:
        out = _series_numpy(rng, coarse, density, pulse_low, positions, centre, fs, n, s.start_time, times, fc, c,
                            n_azimuth, decim, n_low, record.geometry.rx_pose.velocity, carrier, real,
                            components, texture)
    x = s.array + out.astype(s.array.dtype)
    kind = s.kind
    s.array = x
    s.kind = kind
    record.samples = s
    return power[start:start + n]
