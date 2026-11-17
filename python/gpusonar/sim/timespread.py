"""Time spreading of boundary-bounce paths, after SST (APL-UW TR 0702, Sec. 12.4).

A path that reflects from a rough boundary does not arrive as one copy of the pulse. Part of
its energy arrives coherently at the specular delay; the rest arrives over the following
milliseconds with random amplitude and phase that change slowly from ping to ping. Without
this, a bounce path is either fully coherent (flat boundary) or simply weaker (coherent loss
only, `acoustics.surface_coherent_reflection`), and processing gain on bounce paths is too good.

`TimeSpreadProvider` wraps any eigenray provider. For each ray with bounces it

1. splits the total (flat-boundary) intensity into a coherent part, the product of
   exp(-Gamma^2), Gamma = 2 k h sin(grazing), over the bounces, and an incoherent rest;
2. keeps the ray with the coherent amplitude;
3. adds `taps` rays later than the specular delay, whose mean intensities follow an
   exponential intensity impulse response and sum to the incoherent part, with complex
   Gaussian amplitudes;
4. lets those amplitudes change in time with an exponential correlation (what SST's AR(1)
   filter gives), at a rate set by the Doppler spread of the bounce.

What comes from TR 0702 and what does not:

- The structure (1-4), the three boundary parameters (rms height, mean-square slope, rms
  vertical speed) and the decay time are TR 0702's. Its rule of thumb for the decay time is
  "mean-square slope x half the one-way travel time for a bounce in mid-path; mean-square
  slope x twice the travel time of the shorter leg for a bounce near one end". Both are
  the two ends of  T = 2 mss t1 t2 / (t1 + t2)  (t1, t2 the travel times before and after
  the bounce), which is what is used.
- TR 0702 takes the coherent/incoherent split, the shape of the intensity impulse response
  and the frequency spread from APL-UW TR 9407, which was not available. Substituted here:
  the Rayleigh-parameter coherent reflection; an exponential shape; and a Doppler width
  2 f v sin(grazing) / c for a boundary moving vertically at rms speed v (the Doppler of a
  mirror), with v the root-sum-square of the surface's own speed and (speed of the
  reflection point) x sqrt(mss) as TR 0702 states.
- Several bounces on one path: the mean excess delays add and one exponential with that
  mean is used. TR 0702 says its own multi-bounce shape is wrong; so is this one.
- Angle spreading is not modelled (nor is it in SST).
- The same realisation is used for scatterers within `coherence_length` of each other, so
  the highlights of one target fade together and distant targets fade independently.

The random amplitudes are a deterministic function of (seed, place, ray, tap, time): no
state is kept, a scenario's seed reproduces a ping, and pings can be simulated in any order.
"""
from __future__ import annotations

import numpy as np

from .. import acoustics
from .propagation import EigenrayProvider, EigenraySet

_M1, _M2, _M3 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser: a well-mixed 64-bit hash of each element."""
    with np.errstate(over="ignore"):
        x = (x + _M1).astype(np.uint64)
        x = (x ^ (x >> np.uint64(30))) * _M2
        x = (x ^ (x >> np.uint64(27))) * _M3
        return x ^ (x >> np.uint64(31))


def _uniform(key: np.ndarray, stream: int) -> np.ndarray:
    """Uniform numbers in (0, 1), one per key, independent between streams."""
    with np.errstate(over="ignore"):
        h = _mix(key ^ (np.uint64(stream) * _M2))
    return ((h >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)


def boundary_parameters(cfg: dict | None, *, surface: bool) -> dict | None:
    """{rms_height, mean_square_slope, rms_vertical_speed} from a config section: given
    directly, or for the sea surface from `wind_speed` (Pierson-Moskowitz height and vertical
    speed, Cox-Munk slope). None or all zero: no spreading at that boundary."""
    if not cfg:
        return None
    out = {"rms_height": 0.0, "mean_square_slope": 0.0, "rms_vertical_speed": 0.0}
    if "wind_speed" in cfg:
        if not surface:
            raise ValueError("time_spread: wind_speed describes the surface, not the bottom")
        u = float(cfg["wind_speed"])
        out = {"rms_height": acoustics.rms_wave_height(u), "mean_square_slope": acoustics.mean_square_slope(u),
               "rms_vertical_speed": acoustics.rms_surface_vertical_speed(u)}
    for key in out:
        if key in cfg:
            out[key] = float(cfg[key])
    # Energy absorbed by near-surface bubbles at each bounce: a number in dB, or a table
    # {grazing_deg: [...], loss_db: [...]}. No wind-speed law is built in (the APL-UW one is in
    # a handbook that was not available): supply measured or published values.
    loss = cfg.get("bubble_loss_db", 0.0)
    if isinstance(loss, dict):
        out["bubble_loss"] = (np.radians(np.asarray(loss["grazing_deg"], dtype=float)), np.asarray(loss["loss_db"], dtype=float))
    elif float(loss) != 0.0:
        out["bubble_loss"] = (np.array([0.0, np.pi / 2]), np.array([float(loss), float(loss)]))
    return out if out["rms_height"] > 0.0 or "bubble_loss" in out else None


class TimeSpreadProvider(EigenrayProvider):
    """An eigenray provider with time-spread bounce paths (see the module docstring).

    inner: the provider of flat-boundary ("total") eigenrays.
    surface, bottom: boundary parameters (`boundary_parameters`), or None.
    surface_z, bottom_z: boundary depths, m, for the position of each bounce along its path.
    frequency: Hz, for the coherent split and the Doppler width.
    taps: incoherent arrivals added per bounce ray. spacing: s between them (None: so that
    they span three decay times, but never closer than `min_spacing`).
    """

    def __init__(self, inner: EigenrayProvider, *, frequency: float, surface: dict | None = None,
                 bottom: dict | None = None, surface_z: float = 0.0, bottom_z: float | None = None,
                 taps: int = 6, min_spacing: float = 0.0, coherence_length: float = 10.0, seed: int = 0,
                 surface_sound_speed: float | None = None, bottom_sound_speed: float | None = None,
                 incoherent: bool = True):
        self.inner = inner
        self.frequency = float(frequency)
        self.surface, self.bottom = surface, bottom
        self.surface_z, self.bottom_z = float(surface_z), (None if bottom_z is None else float(bottom_z))
        self.taps = int(taps)
        self.min_spacing = float(min_spacing)
        self.coherence_length = float(coherence_length)
        self.seed = int(seed)
        self.c_surface, self.c_bottom = surface_sound_speed, bottom_sound_speed
        self.incoherent = bool(incoherent)   # False: coherent loss only (the scattered part is dropped)
        self.time = 0.0                # s: when the current ping is
        self.platform_speed = 0.0      # m/s: horizontal speed of the sonar
        if self.taps < 1:
            raise ValueError("time_spread: taps must be at least 1")

    # Reverberation is already incoherent and uses mean intensities: it takes the total paths.
    @property
    def reverberation_provider(self) -> EigenrayProvider:
        return self.inner

    def set_ping(self, time: float, platform_speed: float = 0.0) -> None:
        self.time, self.platform_speed = float(time), float(platform_speed)

    # -- geometry of the bounces ---------------------------------------------------------------

    def _bounce_fractions(self, source_z, point_z, up_first, n_surface, n_bottom):
        """Fraction of the path (0..1) at which each bounce happens and which boundary it is,
        from unfolding the path vertically: [..., n_max] fractions (NaN where there is no such
        bounce) and a matching array that is True for surface bounces. Straight-ray geometry;
        with refraction it is the same sequence of boundaries with approximate positions."""
        n = (n_surface + n_bottom).astype(int)
        n_max = int(n.max()) if n.size else 0
        frac = np.full(n.shape + (n_max,), np.nan)
        is_surface = np.zeros(n.shape + (n_max,), dtype=bool)
        if n_max == 0:
            return frac, is_surface
        top = self.surface_z
        bot = self.bottom_z if self.bottom_z is not None else np.inf
        depth = bot - top
        # With one kind of boundary only, every bounce is from it, once.
        both = (n_surface > 0) & (n_bottom > 0) | (n > 1)
        first_up = np.where(n_bottom == 0, True, np.where(n_surface == 0, False, up_first))
        d_first = np.where(first_up, source_z - top, bot - source_z)
        last_up = first_up ^ ((n - 1) % 2 == 1)                     # boundary of the last bounce
        d_last = np.where(last_up, point_z - top, bot - point_z)
        d_first = np.abs(np.where(np.isfinite(d_first), d_first, 0.0))
        d_last = np.abs(np.where(np.isfinite(d_last), d_last, 0.0))
        between = np.where(both & np.isfinite(depth), depth, 0.0)
        total = d_first + d_last + np.maximum(n - 1, 0) * between
        total = np.where(total > 0.0, total, 1.0)
        for i in range(n_max):
            here = n > i
            frac[..., i] = np.where(here, (d_first + i * between) / total, np.nan)
            is_surface[..., i] = np.where(here, first_up ^ (i % 2 == 1), False)
        return np.clip(frac, 1e-6, 1.0 - 1e-6), is_surface

    # -- the provider ----------------------------------------------------------------------------

    def eigenrays(self, source, points) -> EigenraySet:
        base = self.inner.eigenrays(source, points)
        if self.surface is None and self.bottom is None:
            return base
        source = np.asarray(source, dtype=float).reshape(3)
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        s, k = base.delay.shape
        c = float(base.sound_speed)
        bounces = base.valid & ((base.n_surface + base.n_bottom) > 0)
        if not bounces.any():
            return base
        frac, is_surface = self._bounce_fractions(source[2], points[:, 2][:, None], base.launch[..., 2] < 0.0,
                                                  base.n_surface, base.n_bottom)
        cos_launch = np.clip(np.hypot(base.launch[..., 0], base.launch[..., 1]), 0.0, 1.0)
        coherent2 = np.ones((s, k))            # product of exp(-Gamma^2) over the bounces
        absorbed = np.ones((s, k))             # intensity factor left after absorption (bubbles) at the bounces
        mean_excess = np.zeros((s, k))         # s: mean delay of the incoherent part after the specular one
        speed2 = np.zeros((s, k))              # (m/s)^2 x sin^2(grazing), summed over bounces
        for i in range(frac.shape[-1]):
            for surface, par, c_b in ((True, self.surface, self.c_surface), (False, self.bottom, self.c_bottom)):
                if par is None:
                    continue
                here = np.isfinite(frac[..., i]) & (is_surface[..., i] == surface) & bounces
                if not here.any():
                    continue
                # Grazing angle at the boundary from the launch angle by Snell's law.
                cos_g = np.clip(cos_launch * ((c_b or c) / c), 0.0, 1.0)
                grazing = np.arccos(cos_g)
                r = acoustics.rough_coherent_reflection(grazing, par["rms_height"], self.frequency, c_b or c)
                coherent2 = np.where(here, coherent2 * r**2, coherent2)
                if "bubble_loss" in par:
                    loss_db = np.interp(grazing, *par["bubble_loss"])
                    absorbed = np.where(here, absorbed * 10.0 ** (-loss_db / 10.0), absorbed)
                f = np.where(here, frac[..., i], 0.5)
                t1, t2 = f * base.delay, (1.0 - f) * base.delay
                mean_excess += np.where(here, 2.0 * par["mean_square_slope"] * t1 * t2 / np.maximum(t1 + t2, 1e-12), 0.0)
                v2 = par["rms_vertical_speed"] ** 2 + self.platform_speed**2 * par["mean_square_slope"]
                speed2 += np.where(here, v2 * np.sin(grazing) ** 2, 0.0)
        total = np.where(bounces, base.amplitude * np.sqrt(absorbed), base.amplitude)   # what the boundary does not absorb
        incoherent = np.where(bounces & self.incoherent, np.abs(total) ** 2 * (1.0 - coherent2), 0.0)
        # Taps: equally spaced from one spacing after the specular delay, exponential mean intensity.
        j = np.arange(1, self.taps + 1)
        spacing = np.maximum(3.0 * mean_excess / self.taps, self.min_spacing)             # [S, K]
        tau = spacing[..., None] * j                                                       # [S, K, J]
        decay = np.maximum(mean_excess, 1e-12)[..., None]
        weight = np.exp(-tau / decay)
        weight = np.where(mean_excess[..., None] > 0.0, weight, 1.0)                       # no slope given: no delay spread
        weight = weight / weight.sum(axis=-1, keepdims=True)
        # A handful of taps cannot follow the exponential exactly: scale their delays so that the
        # mean excess delay of the intensity is the model's (unless the spacing floor forbids it).
        mean_tau = (weight * tau).sum(axis=-1, keepdims=True)
        scale = np.where(spacing[..., None] > self.min_spacing, decay / np.maximum(mean_tau, 1e-30), 1.0)
        tau = np.where(mean_excess[..., None] > 0.0, tau * scale, 0.0)
        g = self._process(source, points, base, np.sqrt(speed2))                           # [S, K, J] complex, unit variance
        tap_amp = np.sqrt(incoherent[..., None] * weight) * g

        def widen(a, extra):
            return np.concatenate([a, extra.reshape(s, k * self.taps, *a.shape[2:])], axis=1)

        rep = lambda a: np.repeat(a[:, :, None], self.taps, axis=2)                        # noqa: E731
        return EigenraySet(
            delay=widen(base.delay, base.delay[..., None] + tau),
            amplitude=widen(np.where(bounces, total * np.sqrt(coherent2), base.amplitude), tap_amp),
            launch=widen(base.launch, rep(base.launch)), arrival=widen(base.arrival, rep(base.arrival)),
            n_surface=widen(base.n_surface, rep(base.n_surface)), n_bottom=widen(base.n_bottom, rep(base.n_bottom)),
            valid=widen(base.valid, rep(bounces) & (incoherent[..., None] > 0.0)), sound_speed=c)

    def _process(self, source, points, base: EigenraySet, speed_sin: np.ndarray) -> np.ndarray:
        """Unit-variance complex amplitudes [S, K, J] at the current ping time. Each is a sum of
        sinusoids in time with Cauchy-distributed frequencies of scale B, so its autocorrelation
        is exp(-2 pi B |t|): the statistics of white noise through an AR(1) filter."""
        s, k = base.delay.shape
        n_sin = 8
        cell = np.floor(points / max(self.coherence_length, 1e-6)).astype(np.int64)
        src = np.floor(source / max(self.coherence_length, 1e-6)).astype(np.int64)
        with np.errstate(over="ignore"):
            key = np.uint64(self.seed) * _M3
            for axis in range(3):
                key = _mix(key ^ cell[:, axis].astype(np.uint64) * _M1 ^ np.uint64(src[axis]) * _M2)
            # A ray is known by its bounce counts and the sign of its launch: stable from ping to ping.
            ray = (base.n_surface.astype(np.uint64) * np.uint64(64) + base.n_bottom.astype(np.uint64)) * np.uint64(2) \
                + (base.launch[..., 2] < 0.0).astype(np.uint64)
            key = _mix(key[:, None] ^ ray * _M2)                                           # [S, K]
            key = _mix(key[..., None] ^ (np.arange(self.taps, dtype=np.uint64) + np.uint64(1)) * _M3)     # [S, K, J]
            key = _mix(key[..., None] ^ (np.arange(n_sin, dtype=np.uint64) + np.uint64(1)) * _M1)         # [S, K, J, M]
        width = 2.0 * self.frequency * speed_sin / float(base.sound_speed)                 # Hz, [S, K]
        nu = width[..., None, None] * np.tan(np.pi * (_uniform(key, 1) - 0.5))
        phase = 2.0 * np.pi * (_uniform(key, 2) + nu * self.time)
        return np.exp(1j * phase).sum(axis=-1) / np.sqrt(n_sin)


def wrap(provider: EigenrayProvider, cfg: dict, *, frequency: float) -> EigenrayProvider:
    """Wrap a provider according to a scenario's `time_spread` section:
    {surface: {wind_speed | rms_height, mean_square_slope, rms_vertical_speed}, bottom: {...},
     taps, min_spacing, coherence_length, seed}."""
    surface = boundary_parameters(cfg.get("surface"), surface=True)
    bottom = boundary_parameters(cfg.get("bottom"), surface=False)
    surface_z = getattr(provider, "surface_z", 0.0)
    bottom_z = getattr(provider, "bottom_z", None)
    c_surface = c_bottom = None
    env = getattr(provider, "env", None)
    if env is not None:                       # a ray-traced ocean: its depth and the speeds at its boundaries
        bottom_z = float(env.depth)
        surface_z = 0.0
        c_surface, c_bottom = float(env.speed_at(0.0)), float(env.speed_at(env.depth))
        if getattr(provider, "wind_speed", 0.0) > 0.0:
            # The provider was taking the coherent loss itself; the wrapper does the split instead.
            if surface is None:
                surface = boundary_parameters({"wind_speed": provider.wind_speed}, surface=True)
            provider.wind_speed = 0.0
    return TimeSpreadProvider(provider, frequency=float(cfg.get("frequency", frequency)), surface=surface, bottom=bottom,
                              surface_z=0.0 if surface_z is None else surface_z, bottom_z=bottom_z,
                              taps=int(cfg.get("taps", 6)), min_spacing=float(cfg.get("min_spacing", 0.0)),
                              coherence_length=float(cfg.get("coherence_length", 10.0)), seed=int(cfg.get("seed", 0)),
                              surface_sound_speed=c_surface, bottom_sound_speed=c_bottom,
                              incoherent=bool(cfg.get("incoherent", True)))
