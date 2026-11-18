"""Late reverberation of an enclosed basin (fjord, harbour): an opt-in diffuse tail [OURS].

What it is for. In a fjord the sound of a source keeps arriving for most of a second after the
arrivals a ray model over the charted section gives: scattering from shores, side slopes and
distant boundaries. Measured (SFI Smart Ocean dataset, 4-8 kHz, one fjord, one night; van Walree
et al., IEEE DataPort 2025, doi:10.21227/3aa6-4k33, CC BY 4.0; our analysis, PHYSICS_NOTES
Sec. 21.6 and 25.1): 99 % of the received energy took 465 ms at 274 m and 636 ms at 1.02 km,
where Bellhop puts 98 to 99 % in the first 50 ms. The simulator's ocean is 2-D and has no sides,
so it has nothing of the kind. This module adds it as what it looks like to a receiver: a
stochastic tail behind every modelled response.

The model (nothing here is from a paper; the form is the one room acoustics uses for a
reverberant enclosure, and it is fitted to the two measured curves by
tools/basin_reverberation_fit.py):

- One-way. Behind each modelled one-way response (a radiating source heard by the array) comes a
  tail  x * g : the modelled signal x convolved with a kernel g(tau) of independent zero-mean
  Gaussian samples whose variance follows  E p(tau - onset),  p(u) = exp(-u / T) / T  for u >= 0
  (`law: exponential`, the only law: see below), E = 10^(energy_db / 10) the tail's energy over
  the modelled arrivals' energy, T = `decay_s`, `onset_s` the delay at which it starts. Closed
  forms: the share of the tail's energy later than t is exp(-(t - onset) / T); its mean delay
  is onset + T; its rms delay spread is T.
- Two-way (echoes of the sonar's own ping). Each leg has the response delta + tail, so an echo
  becomes echo * (delta + g_out) * (delta + g_back): the echo, plus (1) the echo convolved with
  g_out: the sound that reached the target late and came straight back; it arrives from the
  target's direction with the echo's own wavefront (coherent across the array, smeared in range),
  energy E; (2) the echo convolved with g_back: the sound that came back by way of the basin,
  diffuse at the array, energy E; (3) both, diffuse, energy E^2 with the variance envelope
  p * p (for the exponential (u / T^2) exp(-u / T) from twice the onset). The tail of an echo
  therefore carries 2 E + E^2 times the echo's energy, E of it at the target's bearing.
- The sonar's own ping (`own_ping_db`, optional). The basin's answer to the transmission itself,
  with no target: the transmitted pulse convolved with a diffuse kernel of the same law whose
  energy is 10^(own_ping_db / 10) times that of the pulse at 1 m. THE MEASUREMENT DOES NOT GIVE
  THIS NUMBER: it was made between a source and hydrophones 274 m and 1 km apart, not at the
  source. In those terms the late field was -43 dB (274 m) and -60 dB (1020 m) re the source's
  energy at 1 m (tools/basin_reverberation_fit.py); a sonar listening to its own ping is at
  range zero. Give a value only as a what-if.
- Diffuse means: a sum of `directions` plane waves, each with its own independent kernel, from
  azimuths spread evenly round the horizon (`coherence_m: isotropic`; the coherence between two
  hydrophones d apart in the horizontal is then J0(k d), zero first at 0.38 wavelengths), or
  from a sector about `bearing_deg` with Gaussian rms width sqrt(2) / (k L), k at the centre
  frequency (`coherence_m: L`; the coherence across the sector's direction then falls to 1/e
  at d = L). Elevations are Gaussian about the horizontal with rms `elevation_spread_deg`
  (default 10: an ASSUMPTION; nothing measured). Each element gets every plane wave with its
  own delay, so the tail is reverberation to a beamformer: no one bearing, array gain against
  it as against noise of that directionality.
- Ping to ping. `repeatable` is the share of the tail's energy that is the same on every ping
  (frozen geometry); the rest is drawn anew each ping. Measured over five chirps 3 s apart, late
  part of the response: 0.5 to 0.7 at 274 m, 0.35 to 0.65 at 1 km (the source hung from a
  floating platform, which accounts for some of what did not repeat). The frozen part belongs
  to places in the water: a sonar that has moved sees each plane wave with the phase of its
  new position (k u . x at the centre frequency), so a moving array's tail decorrelates over
  about a third of a wavelength of travel whatever `repeatable` says. Within a ping the kernel
  does not change.

Why exponential and nothing else. The two measured energy-time curves (energy still to come
against time, 80 soundings each) are straight lines in dB from 10 ms down to -30 and -40 dB:
an exponential (two numbers: energy, decay) fits them to 0.5-0.8 dB rms, with T = 0.145-0.158 s
at 274 m and 0.161-0.162 s at 1020 m. A power law (three numbers) fits no better at 1 km (it
converges to its own exponential limit, exponent > 400) and by 0.4 dB better at 274 m with an
exponent of 11 to 14, which is an exponential with a slight curvature. So the decay is that of
the basin and nearly the same at both ranges, as a room's reverberation time is.

Limits (stated, not hidden):
- Added after synthesis, in NumPy, on either engine (it is noise-like: a kernel of ~1e5
  independent samples per plane wave; nothing in it needs the kernel's sub-sample accuracy).
- A tail is made from what is in the record's window: an arrival before the window opens
  leaves no tail in it (the own-ping term is the exception: it is made from the transmit time).
- The diffuse terms take their level from one hydrophone (the one nearest the array's centre),
  so level differences of the modelled arrivals across the array are not carried into them.
  With directional elements the plane waves are weighted by the element's pattern.
- No Doppler: neither the platform's motion nor moving scatterers shift the tail.
- The boundary reverberation of sim/reverb.py (the 2-D ocean's own) and the projector's direct
  sound (`transmit.direct_path`) get no tail.
- Truncated where the variance has fallen by 60 dB (13.8 decay times).

Scenario key (no default; absent: nothing is added and the record is bit-identical):

    basin_reverberation:
      energy_db: -4.3        # one-way tail energy re the modelled arrivals, dB
      onset_s: 0.010
      decay_s: 0.145         # law: exponential (the only one)
      coherence_m: isotropic # or a length in m, with bearing_deg
      repeatable: 0.7        # share of the energy that repeats from ping to ping (true: 1, false: 0)
      own_ping_db: -43.0     # optional what-if (see above)
      elevation_spread_deg: 10.0, directions: null (from the array's size), seed: null (the scenario's),
      echoes: true, sources: true

`MEASURED` holds the two measured sets, for documentation and tests: one fjord, one night, 4-8 kHz.
They are not presets.
"""
from __future__ import annotations

import numpy as np

from .. import _gpusonar as core

# One fjord, one night, 4-8 kHz (tools/basin_reverberation_fit.py, docs/figures/basin_reverberation_fit.md):
# exponential fitted to the energy still to come, from 10 ms to -30 dB. `own_ping_equivalent_db`
# is the tail's energy re the source's at 1 m AT THAT RANGE (stated source level 183 dB), not at the source.
MEASURED = {
    "smartocean_274m": {"energy_db": -4.3, "onset_s": 0.010, "decay_s": 0.145, "repeatable": 0.7, "own_ping_equivalent_db": -43.0},
    "smartocean_1020m": {"energy_db": 0.2, "onset_s": 0.010, "decay_s": 0.160, "repeatable": 0.45, "own_ping_equivalent_db": -60.0},
}
MEASURED_LABEL = "one fjord, one night, 4-8 kHz (SFI Smart Ocean dataset, doi:10.21227/3aa6-4k33, CC BY 4.0)"
_REQUIRED = ("energy_db", "onset_s", "decay_s", "coherence_m", "repeatable")
_KNOWN = set(_REQUIRED) | {"law", "bearing_deg", "elevation_spread_deg", "directions", "seed", "own_ping_db", "echoes", "sources"}
_TRUNCATE = 13.8          # decay times kept: -60 dB


# -- the law: closed forms ------------------------------------------------------------------------------

def energy_later_than(t, onset: float, decay: float):
    """Share of the one-way tail's energy that arrives later than delay t (s)."""
    return np.exp(-np.maximum(np.asarray(t, dtype=float) - onset, 0.0) / decay)


def delay_of_share(share, onset: float, decay: float, energy_db: float):
    """Delay by which all but `share` of the WHOLE response's energy (modelled arrivals, taken to be
    at delay 0, plus tail) has arrived: what "99 % within" reads for share = 0.01. 0 when the
    tail holds less than that."""
    e = 10.0 ** (energy_db / 10.0)
    f = e / (1.0 + e)                                  # the tail's share of the whole
    share = np.asarray(share, dtype=float)
    return np.where(share < f, onset + decay * np.log(f / np.maximum(share, 1e-300)), 0.0)


def rms_delay_spread(onset: float, decay: float, energy_db: float | None = None) -> float:
    """Rms delay spread: of the tail alone (`energy_db` None; = decay), or of the whole response
    with the modelled arrivals as one arrival at delay 0."""
    if energy_db is None:
        return float(decay)
    e = 10.0 ** (energy_db / 10.0)
    f = e / (1.0 + e)
    m1 = f * (onset + decay)
    m2 = f * ((onset + decay) ** 2 + decay**2)
    return float(np.sqrt(m2 - m1**2))


def two_way_density(u, onset: float, decay: float, energy: float):
    """Variance density (1/s) of the DIFFUSE part of a two-way tail at delay u after the echo, per
    unit echo energy: E p(u - onset) + E^2 (p * p)(u - 2 onset)."""
    u = np.asarray(u, dtype=float)
    a, b = u - onset, u - 2.0 * onset
    with np.errstate(over="ignore", under="ignore"):
        one = np.where(a >= 0.0, np.exp(-np.maximum(a, 0.0) / decay) / decay, 0.0)
        two = np.where(b >= 0.0, np.maximum(b, 0.0) / decay**2 * np.exp(-np.maximum(b, 0.0) / decay), 0.0)
    return energy * one + energy**2 * two


# -- directions -----------------------------------------------------------------------------------------

def plane_waves(n: int, coherence, wavenumber: float, *, bearing_deg: float | None = None,
                elevation_spread_deg: float = 10.0, rng: np.random.Generator) -> np.ndarray:
    """Unit vectors [n, 3] (NED: x, y horizontal, z down) pointing to where each plane wave comes
    from. Azimuths: one in each of n equal sectors round the horizon, at a random place in it
    (`isotropic`); or Gaussian about `bearing_deg` (from +x towards +y) with rms sqrt(2) / (k L),
    by equal steps of probability, for a coherence length L. Elevations: Gaussian, rms
    `elevation_spread_deg`, cut at 60 degrees."""
    from scipy.special import ndtri
    u = (np.arange(n) + rng.random(n)) / n
    if isinstance(coherence, str):
        if coherence != "isotropic":
            raise ValueError("basin_reverberation: coherence_m is a length in metres or 'isotropic'")
        azimuth = 2.0 * np.pi * u
    else:
        if bearing_deg is None:
            raise ValueError("basin_reverberation: a coherence length needs bearing_deg (where the reverberation comes from)")
        if float(coherence) <= 0.0:
            raise ValueError("basin_reverberation: coherence_m must be positive")
        sigma = np.sqrt(2.0) / (wavenumber * float(coherence))
        azimuth = np.radians(float(bearing_deg)) + sigma * ndtri(np.clip(u, 1e-9, 1.0 - 1e-9))
    elevation = np.clip(np.radians(float(elevation_spread_deg)) * rng.standard_normal(n), -np.pi / 3, np.pi / 3)
    rng.shuffle(elevation)
    return np.stack([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)], axis=1)


def direction_count(positions: np.ndarray, wavenumber_max: float) -> int:
    """Plane waves needed for a field that is diffuse over the whole array: a field in a disc of
    diameter D has about k D independent angular components; 1.3 times that and 16 more."""
    p = np.asarray(positions, dtype=float)
    extent = float(np.linalg.norm(p.max(axis=0) - p.min(axis=0))) if len(p) > 1 else 0.0
    return int(np.clip(np.ceil(1.3 * wavenumber_max * extent + 16.0), 24, 512))


# -- the generator --------------------------------------------------------------------------------------

def _kernel(rng: np.random.Generator, n: int, complex_: bool) -> np.ndarray:
    """n independent samples of unit variance (real, or complex with E |g|^2 = 1)."""
    if complex_:
        return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
    return rng.standard_normal(n)


class BasinReverberation:
    """The tail generator of one scenario (module docstring). `attach` puts it on a provider."""

    def __init__(self, cfg: dict, seed: int = 0):
        cfg = dict(cfg)
        unknown, missing = set(cfg) - _KNOWN, [k for k in _REQUIRED if k not in cfg]
        if unknown:
            raise ValueError(f"basin_reverberation: unknown keys {sorted(unknown)}")
        if missing:
            raise ValueError(f"basin_reverberation: {missing} must be given (there are no defaults: basin.MEASURED has two measured sets, {MEASURED_LABEL})")
        if str(cfg.get("law", "exponential")) != "exponential":
            raise ValueError("basin_reverberation: law is 'exponential' (the measured curves support no other: sim/basin.py)")
        self.energy = 10.0 ** (float(cfg["energy_db"]) / 10.0)
        self.energy_db = float(cfg["energy_db"])
        self.onset, self.decay = float(cfg["onset_s"]), float(cfg["decay_s"])
        if self.onset < 0.0 or self.decay <= 0.0:
            raise ValueError("basin_reverberation: onset_s must not be negative and decay_s must be positive")
        self.coherence = cfg["coherence_m"] if isinstance(cfg["coherence_m"], str) else float(cfg["coherence_m"])
        self.bearing = None if cfg.get("bearing_deg") is None else float(cfg["bearing_deg"])
        if not isinstance(self.coherence, str) and self.bearing is None:
            raise ValueError("basin_reverberation: a coherence length needs bearing_deg")
        if isinstance(self.coherence, str) and self.coherence != "isotropic":
            raise ValueError("basin_reverberation: coherence_m is a length in metres or 'isotropic'")
        self.elevation_spread = float(cfg.get("elevation_spread_deg", 10.0))
        r = cfg["repeatable"]
        self.repeatable = (1.0 if r else 0.0) if isinstance(r, bool) else float(r)
        if not 0.0 <= self.repeatable <= 1.0:
            raise ValueError("basin_reverberation: repeatable is a share of the energy, 0 to 1 (or true / false)")
        self.own_ping = None if cfg.get("own_ping_db") is None else 10.0 ** (float(cfg["own_ping_db"]) / 10.0)
        self.directions = None if cfg.get("directions") is None else int(cfg["directions"])
        self.seed = int(cfg.get("seed", seed) if cfg.get("seed") is not None else seed)
        self.echoes, self.sources = bool(cfg.get("echoes", True)), bool(cfg.get("sources", True))
        self.last = {}                 # what the last call added (energies, counts): for tests and reports

    # .. what one record needs ..........................................................................

    def _array(self, record):
        from .sources import _channel_clocks
        s = record.samples
        positions = np.asarray(record.rx_positions(0.0), dtype=float)
        c = float(record.environment.sound_speed)
        offset, _ = _channel_clocks(record)
        centre = positions.mean(axis=0)
        ref = int(np.argmin(np.linalg.norm(positions - centre, axis=1)))
        complex_ = s.kind == core.SampleKind.ComplexBaseband
        fc = float(s.center_frequency)
        return {"positions": positions, "c": c, "offset": offset, "ref": ref, "complex": complex_, "fc": fc,
                "fs": float(s.sample_rate), "start": float(s.start_time), "n": int(s.n_samples), "centre": centre}

    def _waves(self, record, arr: dict, f_top: float) -> tuple[np.ndarray, np.ndarray]:
        """Directions [M, 3] and power weights [M] (summing to 1 for omnidirectional elements)."""
        k_c = 2.0 * np.pi * arr["fc"] / arr["c"]
        if self.directions is None:
            # Once per scenario: the directions are part of the frozen field.
            self.directions = direction_count(arr["positions"], 2.0 * np.pi * f_top / arr["c"])
        n = self.directions
        rng = np.random.default_rng([self.seed, 0xBA51, 1])
        u = plane_waves(n, self.coherence, k_c, bearing_deg=self.bearing, elevation_spread_deg=self.elevation_spread, rng=rng)
        from . import reverb
        pattern = reverb.element_power_pattern(record.geometry.rx[0], np.asarray(core.rotation_ned(*record.geometry.rx_pose.rpy)))
        w = np.full(n, 1.0 / n) if pattern is None else np.asarray(pattern(u), dtype=float) / n
        return u, w

    def _kernels(self, tag: int, m: int, n: int, first: int, complex_: bool, ping_key: int) -> np.ndarray:
        """Kernel samples [first, first + n) of plane wave m of term `tag`: sqrt(r) frozen + sqrt(1 - r) this ping's."""
        r = self.repeatable
        out = 0.0
        if r > 0.0:
            out = np.sqrt(r) * _kernel(np.random.default_rng([self.seed, 0xBA51, 2, tag, m]), first + n, complex_)[first:]
        if r < 1.0:
            out = out + np.sqrt(1.0 - r) * _kernel(np.random.default_rng([self.seed, 0xBA51, 3, tag, m, ping_key]), first + n, complex_)[first:]
        return out

    @staticmethod
    def _ping_key(scenario_time: float) -> int:
        return int(round(float(scenario_time) * 1e6)) & 0x7FFFFFFFFFFF

    def _spectrum(self, x: np.ndarray, length: int, arr: dict):
        """Transform of x at `length` points, the bins that carry it, and their absolute frequencies."""
        if arr["complex"]:
            xf = np.fft.fft(x, length)
            f = np.fft.fftfreq(length, 1.0 / arr["fs"]) + arr["fc"]
        else:
            xf = np.fft.rfft(x, length)
            f = np.fft.rfftfreq(length, 1.0 / arr["fs"])
        power = np.abs(xf) ** 2
        keep = np.flatnonzero(power > 1e-12 * power.max()) if power.max() > 0.0 else np.zeros(0, dtype=int)
        return xf, keep, f

    def _diffuse(self, record, arr: dict, x: np.ndarray, x_time: float, density, tag: int, ping_key: int, weights_scale: float = 1.0):
        """[C, N] for the record's window: x (the reference hydrophone's samples from time `x_time`)
        convolved with one kernel per plane wave of variance density(delay) (1/s), each plane wave
        delayed to each element. None when nothing of it reaches the window."""
        fs, n_out = arr["fs"], arr["n"]
        # Kernel delays that put anything of x into the window, on the sample grid from the onset.
        shift = arr["start"] - x_time - self.onset                    # s: output sample 0 is x sample 0 delayed by onset + shift
        first = max(int(np.floor((shift - len(x) / fs) * fs)) - 2, 0)
        last = min(int(np.ceil((shift + n_out / fs) * fs)) + 2, int(np.ceil(_TRUNCATE * 2.0 * self.decay * fs)))
        if last <= first or not np.any(x):
            return None
        n_k = last - first
        tau = self.onset + (first + np.arange(n_k)) / fs
        sigma = np.sqrt(np.maximum(density(tau), 0.0) / fs)
        if not np.any(sigma > 0.0):
            return None
        from scipy.fft import next_fast_len
        rel = arr["positions"] - arr["positions"][arr["ref"]]
        # Room for each element's own delay (the transform is circular): the array's size and the boards' clock offsets.
        margin = int(np.ceil((np.linalg.norm(rel, axis=1).max() / arr["c"] + np.abs(arr["offset"] - arr["offset"][arr["ref"]]).max()) * fs)) + 8
        length = next_fast_len(len(x) + n_k + 2 * margin, real=not arr["complex"])
        xf, keep, f = self._spectrum(x, length, arr)
        if keep.size == 0:
            return None
        power = np.abs(xf[keep]) ** 2
        u, w = self._waves(record, arr, float(np.abs(f[keep][power > 1e-4 * power.max()]).max()))
        w = w * weights_scale
        a = np.empty((len(u), keep.size), dtype=complex)
        for m in range(len(u)):
            g = sigma * self._kernels(tag, m, n_k, first, arr["complex"], ping_key)
            gf = np.fft.fft(g, length) if arr["complex"] else np.fft.rfft(g, length)
            a[m] = np.sqrt(w[m]) * xf[keep] * gf[keep]
        # Each plane wave at the reference point has the phase of where the array now is (frozen field).
        k_c = 2.0 * np.pi * arr["fc"] / arr["c"]
        a *= np.exp(1j * k_c * (u @ arr["positions"][arr["ref"]]))[:, None]
        # Output sample j of the transform is at delay first / fs + j / fs after (x sample 0 + onset); the window's
        # sample 0 is at `shift`: an advance of q0 samples, its fraction applied as a phase.
        q0 = shift * fs - first
        qi = int(np.floor(q0))
        frac = (q0 - qi) / fs
        delay = -(rel @ u.T) / arr["c"] - (arr["offset"] - arr["offset"][arr["ref"]])[:, None] - frac        # [C, M]
        out = np.zeros((len(rel), n_out), dtype=complex if arr["complex"] else float)
        spec = np.zeros(length if arr["complex"] else length // 2 + 1, dtype=complex)
        fk = f[keep]
        for ch in range(len(rel)):
            spec[:] = 0.0
            spec[keep] = (a * np.exp(-2j * np.pi * delay[ch][:, None] * fk[None, :])).sum(axis=0)
            y = np.fft.ifft(spec) if arr["complex"] else np.fft.irfft(spec, length)
            lo, hi = max(qi, 0), min(qi + n_out, length)
            if hi > lo:
                out[ch, lo - qi:hi - qi] = y[lo:hi]
        return out

    def _common(self, x: np.ndarray, arr: dict, tag: int, ping_key: int):
        """[C, N]: every channel's own samples convolved with ONE kernel (variance E p): the part of
        an echo's tail that keeps the echo's wavefront."""
        fs, n_out = arr["fs"], arr["n"]
        n_k = min(n_out, int(np.ceil(_TRUNCATE * self.decay * fs)))
        tau = np.arange(n_k) / fs
        g = np.sqrt(self.energy * np.exp(-tau / self.decay) / self.decay / fs) * self._kernels(tag, 0, n_k, 0, arr["complex"], ping_key)
        from scipy.fft import next_fast_len
        k0 = int(round(self.onset * fs))
        length = next_fast_len(n_out + n_k + 8, real=not arr["complex"])
        if arr["complex"]:
            y = np.fft.ifft(np.fft.fft(x, length, axis=1) * np.fft.fft(g, length)[None, :], axis=1)
        else:
            y = np.fft.irfft(np.fft.rfft(x, length, axis=1) * np.fft.rfft(g, length)[None, :], length, axis=1)
        out = np.zeros_like(y[:, :n_out])
        if k0 < n_out:
            out[:, k0:] = y[:, :n_out - k0]
        return out

    # .. the hooks ......................................................................................

    def _silence_reference_inputs(self, record, sonar, tail: np.ndarray) -> np.ndarray:
        ref = sonar.config["receive"].get("reference") if sonar is not None else None
        if ref and ref.get("dedicated"):
            for channel in ref["channels"]:                   # a reference input has no hydrophone on it
                if channel in record.metadata.channel_element:
                    tail[record.metadata.channel_element.index(channel)] = 0.0
        return tail

    @staticmethod
    def _add(record, tail: np.ndarray) -> None:
        s = record.samples
        kind = s.kind
        s.array = s.array + tail.astype(s.array.dtype)
        s.kind = kind
        record.samples = s

    def after_echoes(self, record, sonar, provider=None, scenario_time: float = 0.0, *, environment: dict | None = None,
                     absorption_model: str = "francois_garrison", backend=None, engine: str | None = None) -> None:
        """Called by the simulator once the echoes of the scene are in the record (and nothing else):
        adds their two-way tails and, with `own_ping_db`, the basin's answer to the ping itself."""
        arr = self._array(record)
        ping = self._ping_key(scenario_time)
        x = np.array(record.samples.array)
        self.last = {"echo_energy": float((np.abs(x) ** 2).sum()), "common_energy": 0.0, "diffuse_energy": 0.0, "own_energy": 0.0}
        total = None
        if self.echoes and np.any(x):
            common = self._common(x, arr, 1, ping)
            self.last["common_energy"] = float((np.abs(common) ** 2).sum())
            total = common
            diffuse = self._diffuse(record, arr, x[arr["ref"]], arr["start"],
                                    lambda t: two_way_density(t, self.onset, self.decay, self.energy), 2, ping)
            if diffuse is not None:
                self.last["diffuse_energy"] = float((np.abs(diffuse) ** 2).sum())
                total = total + diffuse
        if self.own_ping is not None:
            pulse, t0 = transmitted_pulse(record, sonar, environment=environment or {}, absorption_model=absorption_model,
                                          backend=backend, engine=engine, channel=arr["ref"])
            self.last["pulse_energy"] = float((np.abs(pulse) ** 2).sum())
            own = self._diffuse(record, arr, pulse, t0, lambda t: self.own_ping * np.where(t >= self.onset, np.exp(-(t - self.onset) / self.decay) / self.decay, 0.0),
                                3, ping)
            if own is not None:
                self.last["own_energy"] = float((np.abs(own) ** 2).sum())
                total = own if total is None else total + own
        if total is not None:
            self._add(record, self._silence_reference_inputs(record, sonar, total))

    def one_way(self, record, added: np.ndarray, scenario_time: float = 0.0) -> np.ndarray:
        """Called by sim/sources.py with the one-way sound [C, N] of the radiating sources: returns
        it with its tail (diffuse, energy E times the modelled sound's)."""
        if not self.sources or added is None or not np.any(added):
            return added
        arr = self._array(record)
        diffuse = self._diffuse(record, arr, np.asarray(added)[arr["ref"]], arr["start"],
                                lambda t: self.energy * np.where(t >= self.onset, np.exp(-(t - self.onset) / self.decay) / self.decay, 0.0),
                                4, self._ping_key(scenario_time))
        self.last = dict(self.last, source_energy=float((np.abs(added) ** 2).sum()),
                         source_tail_energy=0.0 if diffuse is None else float((np.abs(diffuse) ** 2).sum()))
        return added if diffuse is None else added + diffuse.astype(np.asarray(added).dtype)


def transmitted_pulse(record, sonar, *, environment: dict, absorption_model: str = "francois_garrison", backend=None,
                      engine: str | None = None, channel: int = 0, distance: float = 30.0) -> tuple[np.ndarray, float]:
    """The sonar's own pulse as hydrophone `channel` would record it from 1 m, free field, in the
    record's sample units, and the time of its first sample (s after the transmit): made by the
    one-way pulse path of sim/sources.py (the engines' own kernel, so every waveform type and
    level convention is the simulator's) from a point `distance` m away on the element's
    boresight, scaled back to 1 m."""
    from . import sources as source_models
    from .propagation import make_provider
    c = float(record.environment.sound_speed)
    positions = np.asarray(record.rx_positions(0.0), dtype=float)
    rot = np.asarray(core.rotation_ned(*record.geometry.rx_pose.rpy))
    b = rot @ np.asarray(record.geometry.rx[channel].boresight, dtype=float)
    where = positions[channel] + distance * b / np.linalg.norm(b)
    velocity = np.asarray(record.geometry.rx_pose.velocity, dtype=float)
    duration = float(sonar.waveform_config["duration"])
    s = record.samples
    lead = 0.002
    n = int(np.ceil((duration + 2.0 * lead) * s.sample_rate))
    work = record.copy()
    smp = work.samples
    kind = smp.kind
    smp.array = np.zeros((smp.n_channels, n), dtype=smp.array.dtype)
    smp.kind = kind
    smp.start_time = distance / c - lead
    work.samples = smp
    projector = source_models._Projector(where, velocity, 0.0)
    part = source_models._pulse(work, sonar, projector, projector.signals[0], make_provider("free_field", c), environment, absorption_model, 0.0, c,
                                core.Backend.Host if backend is None else backend, engine, True, (0.0, 0.0, 0.0))
    if part is None:
        return np.zeros(n, dtype=record.samples.array.dtype), -lead
    return np.asarray(part)[channel] * distance, -lead


def attach(provider, cfg: dict | None, seed: int = 0):
    """Put the scenario's `basin_reverberation` on its eigenray provider (where the simulator and
    sim/sources.py look for it) and return the provider. None or empty: nothing is attached."""
    if cfg:
        provider.basin = BasinReverberation(cfg, seed)
    return provider
