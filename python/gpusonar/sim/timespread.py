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
- Several bounces on one path: by default the bounces' responses are convolved (4 below),
  which is what a cascade of independent bounces gives. `multi_bounce="sum"` is the earlier
  rule: the mean excess delays add and one exponential with that mean is used. TR 0702 says
  its own multi-bounce shape is wrong ("too high immediately after the pulse and too low
  later"); `sum` is wrong in that way for two rough bounces, and the other way round when
  one bounce is mostly coherent with the longer decay (tools/sst_validation.py, ASIAEX).
- The same realisation is used for scatterers within `coherence_length` of each other, so
  the highlights of one target fade together and distant targets fade independently.

The random amplitudes are a deterministic function of (seed, place, ray, tap, time): no
state is kept, a scenario's seed reproduces a ping, and pings can be simulated in any order.

Options beyond SST (all off by default; with them off the output is what it was)
---------------------------------------------------------------------------------
The model behind all of them is the one behind the decay time: a boundary of tilted mirrors
(facets) with Gaussian slopes. Nothing here is taken from a paper; it is derived below and
tested against exact geometry. Notation for one bounce: r1, r2 the path lengths before and
after it, g its grazing angle, q = 1/r1 + 1/r2, and (x, y) the position of a facet on the
boundary relative to the specular point, x along the horizontal direction of travel.

Geometry. To second order the path through the facet at (x, y) is longer than the specular
one by  c tau = (x^2 sin^2 g + y^2) q / 2 : the curves of equal delay are ellipses on the
boundary, a factor 1/sin g longer along the path than across it. Seen along the ray the
ellipse is a circle: with (x sin g, y) = rho (cos psi, sin psi),  rho^2 = 2 c tau / q.
The facet reflects from the source to the receiver only if its slopes are
s_x = x sin(g) q / 2  and  s_y = y q / (2 sin g)  (the mirror law; for a far source they are
"tilt the mirror by s, turn the ray by 2 s" in the plane of incidence and "by 2 s sin g"
across it).

1. Angle spreading (`angle_spread`). The tap of excess delay tau comes from the circle of
   radius rho: it leaves the source at an angle  rho / r1 = sqrt(2 c tau r2 / (r1 (r1 + r2)))
   from the specular ray and reaches the far end at  rho / r2 . Where on the circle (psi) is
   drawn once per tap from the density that Gaussian slopes of variance mss/2 per axis give
   on it,  exp(kappa cos 2 psi),  kappa = (c tau q / 4 mss) (1 / sin^2 g - 1): at low grazing
   angles the late facets lie up and down the path rather than beside it. Summed over the
   tail (mean excess delay T) the mean-square angle at the source is 2 c T r2 / (r1 (r1 + r2));
   with the facet decay time below that is  2 mss (1 + sin^2 g) (r2 / (r1 + r2))^2, of which
   the part in the vertical plane is 2 mss (r2 / (r1 + r2))^2 and the part across it is
   sin^2 g times that. The coherent arrival keeps the specular direction. The directions are
   carried in the eigenrays' `launch` and `arrival`, so both engines use them, and a moving
   sonar then sees each tap at its own Doppler shift (which is what TR 0702's "speed of the
   reflection point x sqrt(mss)" stands for).
   On a path with several rough bounces the direction at the source comes from the first of
   them and that at the far end from the last, each with its expected share of the tap's
   excess delay; the cross terms between bounces are neglected.

2. Decay time and shape from the same facets (`decay_time="facet"`). The slopes above put
   tau = 2 (s_x^2 + s_y^2 sin^2 g) / (c q), so the mean excess delay is
   T = mss (1 + sin^2 g) t1 t2 / (t1 + t2): TR 0702's rule of thumb at normal incidence and
   half of it at low grazing angles (there only the slopes along the path delay the sound).
   The shape is that of a sum of two squared Gaussians with means a = mss t1 t2 / (t1 + t2)
   and b = a sin^2 g:  exp(-tau (a + b) / (4 a b)) I0(tau (a - b) / (4 a b)) / (2 sqrt(a b)).
   With this option the angles of (1) are Gaussian in each plane, with the variances given.

3. Frequency spreading within a ping (`doppler`). The process of step 4 above is a sum of
   `doppler_lines` sinusoids in time. By default it is sampled once per ping: the tail fades
   from ping to ping but is frozen during a ping, so a tone comes back as a tone. With
   `doppler` each sinusoid is its own ray with a rate of change of path length
   (`EigenraySet.range_rate`) that gives its frequency, and the same process then runs on
   through the ping. Width: a boundary displaced by zeta(t) changes the path by
   -2 zeta sin g, so the reflected field is exp(i chi zeta), chi = 2 k sin g; for Gaussian
   zeta with rms speed v the second moment of the spectrum of the scattered (non-coherent)
   part is exactly
       sigma_nu^2 = (chi v / 2 pi)^2 / (1 - exp(-Gamma^2)),   chi v / 2 pi = 2 f v sin(g) / c.
   For a rough boundary (Gamma >> 1) the spectrum is Gaussian with that rms width; for a
   nearly flat one it is the wave spectrum itself and the width tends to v / (2 pi h), the
   rms wave frequency. Over several bounces the numerators add and the denominator is one
   minus the product of the coherent factors. `doppler_spectrum="gaussian"` (the default)
   draws the sinusoids' frequencies from a Gaussian of that rms width; "lorentzian" is the
   AR(1) spectrum SST uses (half-width 2 f v sin(g) / c; it has no rms width). Measured
   channels support the Gaussian (PHYSICS_NOTES Sec. 21: 15 of 18 surface arrivals at one
   site against 7 of 18 for the Lorentzian), which is why it is the default. A bottom has
   v = 0: its only spread is that of the sonar's motion. With `angle_spread` the sonar's
   motion is left to the taps' directions and only the boundary's own speed sets the
   frequencies within a ping.

4. Several bounces (`multi_bounce="convolve"`). Each bounce's intensity response is
   c_i delta(tau) + (1 - c_i) h_i(tau)  (c_i its coherent fraction, h_i its tail of mean
   T_i), and a path's response is their convolution. The term with no scattering is the
   coherent arrival, as before; the rest is the tail, of energy 1 - prod c_i (as before) and
   mean  sum (1 - c_i) T_i / (1 - prod c_i)  (before: sum T_i). Two fully rough bounces give
   tau exp(-tau / T): nothing right after the specular delay, most at T, a longer tail. The
   convolution is done on the tap grid.
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


def boundary_parameters(cfg: dict | None, *, surface: bool, frequency: float | None = None) -> dict | None:
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
    # Energy absorbed by near-surface bubbles at each bounce: a number in dB, a table
    # {grazing_deg: [...], loss_db: [...]}, or a wind-speed law {model: hall_novarini} built from
    # open sources (sim/surface.py; the APL-UW law is in a handbook that was not available).
    loss = cfg.get("bubble_loss_db", 0.0)
    if isinstance(loss, dict) and "model" in loss:
        # {model: hall_novarini, wind_speed}: a wind-speed law (sim/surface.py). The wind defaults
        # to the section's `wind_speed`; the frequency is the wrapper's.
        from . import surface as surface_model
        if not surface or frequency is None or "wind_speed" not in {**cfg, **loss}:
            raise ValueError("time_spread: a bubble loss model needs the surface section, a wind_speed and the frequency")
        out["bubble_loss"] = surface_model.bubble_loss_table(str(loss["model"]), float(loss.get("wind_speed", cfg.get("wind_speed"))),
                                                             float(loss.get("frequency", frequency)))
    elif isinstance(loss, dict):
        out["bubble_loss"] = (np.radians(np.asarray(loss["grazing_deg"], dtype=float)), np.asarray(loss["loss_db"], dtype=float))
    elif float(loss) != 0.0:
        out["bubble_loss"] = (np.array([0.0, np.pi / 2]), np.array([float(loss), float(loss)]))
    return out if out["rms_height"] > 0.0 or "bubble_loss" in out else None


def _i0e(x) -> np.ndarray:
    """exp(-x) I0(x) for x >= 0, without overflow (large x: the asymptotic series)."""
    x = np.asarray(x, dtype=float)
    small = np.minimum(x, 500.0)
    big = np.maximum(x, 500.0)
    series = (1.0 + 1.0 / (8.0 * big) + 9.0 / (128.0 * big**2)) / np.sqrt(2.0 * np.pi * big)
    return np.where(x < 500.0, np.i0(small) * np.exp(-small), series)


def facet_delay_density(tau, a, b):
    """Density of the excess delay tau of one bounce from Gaussian facets: tau is a sum of two
    squared Gaussians with means a (slopes along the path) and b <= a (across it),
    exp(-tau (a + b) / (4 a b)) I0(tau (a - b) / (4 a b)) / (2 sqrt(a b)). For a = b it is
    the exponential of mean 2 a."""
    tau = np.asarray(tau, dtype=float)
    a = np.maximum(np.asarray(a, dtype=float), 1e-300)
    b = np.maximum(np.asarray(b, dtype=float), 1e-300)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.exp(-tau / (2.0 * a)) * _i0e(tau * (a - b) / (4.0 * a * b)) / (2.0 * np.sqrt(a * b))


def iso_delay_angle(kappa, u, v, n_grid: int = 129):
    """Where on its circle of equal delay a tap's facet lies: the angle psi (0: along the path,
    towards the far end; pi/2: across it) drawn from the density exp(kappa cos 2 psi) by
    inverting its distribution at the uniform numbers u; v (uniform) picks the quadrant."""
    kappa = np.asarray(kappa, dtype=float)
    shape = kappa.shape
    kappa = kappa.reshape(-1)
    u, v = np.asarray(u, dtype=float).reshape(-1), np.asarray(v, dtype=float).reshape(-1)
    psi0 = np.empty_like(kappa)
    grid = np.linspace(0.0, 0.5 * np.pi, n_grid)
    sin2 = np.sin(grid) ** 2
    step = 32768                                                       # rows at a time: bounded memory
    for lo in range(0, kappa.size, step):
        k = np.clip(kappa[lo:lo + step, None], 0.0, 1e6)
        pdf = np.exp(-2.0 * k * sin2)                                  # exp(kappa (cos 2 psi - 1))
        cdf = np.concatenate([np.zeros((k.shape[0], 1)), np.cumsum(0.5 * (pdf[:, 1:] + pdf[:, :-1]), axis=1)], axis=1)
        target = u[lo:lo + step] * cdf[:, -1]
        i = np.clip((cdf < target[:, None]).sum(axis=1), 1, n_grid - 1)
        rows = np.arange(k.shape[0])
        c0, c1 = cdf[rows, i - 1], cdf[rows, i]
        psi0[lo:lo + step] = grid[i - 1] + (grid[i] - grid[i - 1]) * (target - c0) / np.maximum(c1 - c0, 1e-300)
    quadrant = np.minimum((4.0 * v).astype(int), 3)
    psi = np.where(quadrant == 0, psi0, np.where(quadrant == 1, np.pi - psi0, np.where(quadrant == 2, np.pi + psi0, -psi0)))
    return psi.reshape(shape)


def _turn(direction, angle, psi, sign: float):
    """Unit vectors `direction` [..., 3] turned by `angle` (rad) towards the facet at angle psi
    on the circle of equal delay. sign +1: the direction leaving the source; -1: the direction
    of travel on arrival. A facet farther along the path is seen at a flatter angle from the
    source and at a steeper one from the far end."""
    hn = np.hypot(direction[..., 0], direction[..., 1])
    safe = np.maximum(hn, 1e-12)
    hx = np.where(hn > 1e-9, direction[..., 0] / safe, 1.0)
    hy = np.where(hn > 1e-9, direction[..., 1] / safe, 0.0)
    dz = direction[..., 2]
    across = np.stack([-hy, hx, np.zeros_like(hx)], axis=-1)           # horizontal, normal to the ray
    in_plane = np.stack([-dz * hx, -dz * hy, hn], axis=-1)             # normal to the ray, in its vertical plane
    steeper = np.where(dz >= 0.0, 1.0, -1.0)
    # The second-order geometry holds for turns well inside the grazing angle. Beyond it a
    # ray would be turned through the horizontal and miss the boundary: the turn towards
    # the horizontal stops at 0.9 of the ray's own elevation.
    flatter = np.minimum(sign * angle * np.cos(psi), 0.9 * np.arcsin(np.clip(np.abs(dz), 0.0, 1.0)))
    out = direction + (sign * angle * np.sin(psi))[..., None] * across - (flatter * steeper)[..., None] * in_plane
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


class TimeSpreadProvider(EigenrayProvider):
    """An eigenray provider with time-spread bounce paths (see the module docstring).

    inner: the provider of flat-boundary ("total") eigenrays.
    surface, bottom: boundary parameters (`boundary_parameters`), or None.
    surface_z, bottom_z: boundary depths, m, for the position of each bounce along its path.
    frequency: Hz, for the coherent split and the Doppler width.
    taps: incoherent arrivals added per bounce ray. spacing: s between them (None: so that
    they span three decay times, but never closer than `min_spacing`).
    angle_spread: each tap leaves and arrives in its own direction (module docstring, 1).
    decay_time: "rule_of_thumb" (TR 0702's) or "facet" (2).
    doppler: the tail varies during a ping; each tap is then `doppler_lines` rays, each with a
    `range_rate` (3). doppler_spectrum: "gaussian" (default) or "lorentzian".
    multi_bounce: "convolve" (default) or "sum" (4).
    """

    def __init__(self, inner: EigenrayProvider, *, frequency: float, surface: dict | None = None,
                 bottom: dict | None = None, surface_z: float = 0.0, bottom_z: float | None = None,
                 taps: int = 6, min_spacing: float = 0.0, coherence_length: float = 10.0, seed: int = 0,
                 surface_sound_speed: float | None = None, bottom_sound_speed: float | None = None,
                 incoherent: bool = True, angle_spread: bool = False, decay_time: str = "rule_of_thumb",
                 doppler: bool = False, doppler_spectrum: str = "gaussian", doppler_lines: int = 8,
                 multi_bounce: str = "convolve"):
        self.angle_spread, self.doppler = bool(angle_spread), bool(doppler)
        self.decay_time, self.doppler_spectrum, self.multi_bounce = str(decay_time), str(doppler_spectrum), str(multi_bounce)
        self.doppler_lines = int(doppler_lines)
        if self.decay_time not in ("rule_of_thumb", "facet"):
            raise ValueError("time_spread: decay_time is 'rule_of_thumb' or 'facet'")
        if self.doppler_spectrum not in ("lorentzian", "gaussian"):
            raise ValueError("time_spread: doppler_spectrum is 'lorentzian' or 'gaussian'")
        if self.multi_bounce not in ("sum", "convolve"):
            raise ValueError("time_spread: multi_bounce is 'sum' or 'convolve'")
        if self.doppler_lines < 1:
            raise ValueError("time_spread: doppler_lines must be at least 1")
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
        own2 = np.zeros((s, k))                # the same without the sonar's motion: the boundary's own speed
        facet = self.decay_time == "facet"
        slots = []                             # the rough bounces of each ray, in path order
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
                # Mean excess delays from the slopes along the path and across it (facet model).
                along = par["mean_square_slope"] * t1 * t2 / np.maximum(t1 + t2, 1e-12)
                across = along * np.maximum(np.sin(grazing), 1e-3) ** 2
                if facet:
                    mean_i = np.where(here, along + across, 0.0)
                else:
                    mean_i = np.where(here, 2.0 * par["mean_square_slope"] * t1 * t2 / np.maximum(t1 + t2, 1e-12), 0.0)
                mean_excess += mean_i
                v2 = par["rms_vertical_speed"] ** 2 + self.platform_speed**2 * par["mean_square_slope"]
                speed2 += np.where(here, v2 * np.sin(grazing) ** 2, 0.0)
                own2 += np.where(here, par["rms_vertical_speed"] ** 2 * np.sin(grazing) ** 2, 0.0)
                slots.append({"here": here, "mean": mean_i, "coherent": np.where(here, r**2, 1.0), "frac": f,
                              "along": np.where(here, along, 0.0), "across": np.where(here, across, 0.0)})
        total = np.where(bounces, base.amplitude * np.sqrt(absorbed), base.amplitude)   # what the boundary does not absorb
        incoherent = np.where(bounces & self.incoherent, np.abs(total) ** 2 * (1.0 - coherent2), 0.0)
        n_rough = np.zeros((s, k), dtype=int)
        for slot in slots:
            n_rough += slot["here"]
        many = (n_rough >= 2) & (self.multi_bounce == "convolve")
        if many.any():
            # Convolved bounces: a bounce delays the tail only when it scatters.
            scattered = sum((1.0 - slot["coherent"]) * slot["mean"] for slot in slots)
            mean_excess = np.where(many, scattered / np.maximum(1.0 - coherent2, 1e-300), mean_excess)
        # Taps: equally spaced from one spacing after the specular delay, exponential mean intensity.
        j = np.arange(1, self.taps + 1)
        spacing = np.maximum(3.0 * mean_excess / self.taps, self.min_spacing)             # [S, K]
        tau = spacing[..., None] * j                                                       # [S, K, J]
        decay = np.maximum(mean_excess, 1e-12)[..., None]
        weight = np.exp(-tau / decay)
        if facet:
            # One rough bounce: the facets' own shape (several, unless convolved: the exponential).
            single = (n_rough == 1) & (mean_excess > 0.0)
            a_sum, b_sum = sum(slot["along"] for slot in slots), sum(slot["across"] for slot in slots)
            weight = np.where(single[..., None], facet_delay_density(tau, a_sum[..., None], b_sum[..., None]), weight)
        if many.any():
            weight = np.where(many[..., None], self._convolved(slots, spacing, facet), weight)
        weight = np.where(mean_excess[..., None] > 0.0, weight, 1.0)                       # no slope given: no delay spread
        weight = weight / weight.sum(axis=-1, keepdims=True)
        # A handful of taps cannot follow the exponential exactly: scale their delays so that the
        # mean excess delay of the intensity is the model's (unless the spacing floor forbids it).
        mean_tau = (weight * tau).sum(axis=-1, keepdims=True)
        scale = np.where(spacing[..., None] > self.min_spacing, decay / np.maximum(mean_tau, 1e-30), 1.0)
        tau = np.where(mean_excess[..., None] > 0.0, tau * scale, 0.0)
        gaussian = self.doppler_spectrum == "gaussian"
        # The rms width of the scattered part's spectrum, as a speed (module docstring, 3).
        speed_sin = np.sqrt(speed2 / np.maximum(1.0 - coherent2, 1e-12)) if gaussian else np.sqrt(speed2)
        g, nu, phase, tap_key = self._process(source, points, base, speed_sin)             # g: [S, K, J] complex, unit variance
        tap_amp = np.sqrt(incoherent[..., None] * weight) * g
        rep = lambda a: np.repeat(a[:, :, None], self.taps, axis=2)                        # noqa: E731
        tap_delay = base.delay[..., None] + tau
        tap_launch, tap_arrival = rep(base.launch), rep(base.arrival)
        tap_valid = rep(bounces) & (incoherent[..., None] > 0.0)
        if self.angle_spread:
            tap_launch, tap_arrival = self._directions(base, slots, tau, tap_key, tap_valid)
        tap_ns, tap_nb = rep(base.n_surface), rep(base.n_bottom)
        range_rate = None
        if self.doppler:
            # Each sinusoid of the process is a ray of its own with the path-length rate of its frequency.
            m = self.doppler_lines
            line = lambda a: np.repeat(a[:, :, :, None], m, axis=3)                        # noqa: E731
            tap_amp = np.sqrt(incoherent[..., None] * weight)[..., None] * np.exp(1j * phase) / np.sqrt(m)
            # With angle spreading the sonar's motion acts through each tap's direction instead.
            own = np.sqrt(own2 / np.where(speed2 > 0.0, speed2, 1.0)) if self.angle_spread else np.ones((s, k))
            tap_rate = -c * nu * own[..., None, None] / self.frequency
            tap_delay, tap_launch, tap_arrival = line(tap_delay), line(tap_launch), line(tap_arrival)
            tap_valid, tap_ns, tap_nb = line(tap_valid), line(tap_ns), line(tap_nb)
            range_rate = np.concatenate([np.zeros((s, k)), tap_rate.reshape(s, -1)], axis=1)

        def widen(a, extra):
            return np.concatenate([a, extra.reshape(s, -1, *a.shape[2:])], axis=1)

        return EigenraySet(
            delay=widen(base.delay, tap_delay),
            amplitude=widen(np.where(bounces, total * np.sqrt(coherent2), base.amplitude), tap_amp),
            launch=widen(base.launch, tap_launch), arrival=widen(base.arrival, tap_arrival),
            n_surface=widen(base.n_surface, tap_ns), n_bottom=widen(base.n_bottom, tap_nb),
            valid=widen(base.valid, tap_valid), sound_speed=c, range_rate=range_rate)

    def _convolved(self, slots, spacing, facet: bool) -> np.ndarray:
        """Mean tap intensities [S, K, J] (not normalised) of the convolution of the bounces'
        responses c_i delta + (1 - c_i) h_i on the tap grid, without its coherent term."""
        jn = self.taps
        tau = spacing[..., None] * np.arange(1, jn + 1)
        # Index m of a sequence: m spacings of delay. One bounce's tail is sampled at the taps
        # (as for a single bounce) and stored one index down, so that the tail of a path that
        # scattered once sits on the taps and each further scattering adds its delay less one
        # spacing: two equal exponentials then give tau exp(-tau / T) at the taps exactly.
        res = np.zeros(spacing.shape + (jn,))
        res[..., 0] = 1.0
        coherent = np.ones(spacing.shape)
        first = np.zeros(jn, dtype=bool)
        first[0] = True
        for slot in slots:
            here, c_i = slot["here"], slot["coherent"]
            spread = here & (slot["mean"] > 0.0) & (spacing > 0.0)
            if facet:
                shape = facet_delay_density(tau, slot["along"][..., None], slot["across"][..., None])
            else:
                with np.errstate(under="ignore"):
                    shape = np.exp(-tau / np.maximum(slot["mean"], 1e-300)[..., None])
            shape = np.where(spread[..., None], shape, 0.0)
            norm = shape.sum(axis=-1, keepdims=True)
            # A tail shorter than the grid, or a bounce with no slope given: no delay of its own.
            shape = np.where(norm > 0.0, shape / np.maximum(norm, 1e-300), first)
            seq = np.where(here[..., None], (1.0 - c_i)[..., None] * shape, 0.0)
            seq[..., 0] += np.where(here, c_i, 1.0)
            new = np.zeros_like(res)
            for a in range(jn):
                new[..., a:] += res[..., :jn - a] * seq[..., a:a + 1]
            res = new
            coherent = coherent * np.where(here, c_i, 1.0)
        res[..., 0] -= coherent                                                           # the path that never scattered
        return np.maximum(res, 0.0)

    def _directions(self, base: EigenraySet, slots, tau, tap_key, tap_valid):
        """Launch and arrival directions [S, K, J, 3] of the taps (module docstring, 1)."""
        s, k = base.delay.shape
        convolve = self.multi_bounce == "convolve"
        zero = np.zeros((s, k))
        first = {"set": np.zeros((s, k), dtype=bool), "frac": zero + 0.5, "part": zero, "kappa": zero}
        last = {"frac": zero + 0.5, "part": zero, "kappa": zero}
        parts = [slot["mean"] * ((1.0 - slot["coherent"]) if convolve else 1.0) for slot in slots]
        whole = sum(parts)
        for slot, part in zip(slots, parts):
            rough = slot["here"] & (slot["mean"] > 0.0)
            share = part / np.where(whole > 0.0, whole, 1.0)
            # kappa / tau = (1/b - 1/a) / 4 with a, b the mean delays from the two slopes.
            a, b = np.where(rough, slot["along"], 1.0), np.where(rough, slot["across"], 1.0)
            kappa = 0.25 * (1.0 / b - 1.0 / a)
            take = rough & ~first["set"]
            for name, value in (("frac", slot["frac"]), ("part", share), ("kappa", kappa)):
                first[name] = np.where(take, value, first[name])
                last[name] = np.where(rough, value, last[name])
            first["set"] |= rough
        u, v = _uniform(tap_key, 4), _uniform(tap_key, 5)
        delay = np.maximum(base.delay, 1e-12)[..., None]
        out = []
        for end, sign, direction in ((first, 1.0, base.launch), (last, -1.0, base.arrival)):
            f = end["frac"][..., None]
            near = f if sign > 0 else 1.0 - f                               # fraction of the path between this end and the bounce
            t = np.where(tap_valid, tau * end["part"][..., None], 0.0)
            angle = np.sqrt(2.0 * t * (1.0 - near) / (near * delay))
            psi = iso_delay_angle(end["kappa"][..., None] * t, u, v)
            out.append(_turn(np.repeat(direction[:, :, None], self.taps, axis=2), angle, psi, sign))
        return out[0], out[1]

    def _process(self, source, points, base: EigenraySet, speed_sin: np.ndarray):
        """Unit-variance complex amplitudes [S, K, J] at the current ping time, and their parts:
        each is a sum of sinusoids in time of frequencies nu [S, K, J, M] (Hz) and phases
        `phase` (rad, at the ping time); also the taps' hash keys. Lorentzian: Cauchy-distributed
        frequencies of scale B, so the autocorrelation is exp(-2 pi B |t|), the statistics of
        white noise through an AR(1) filter. Gaussian: frequencies of rms B, autocorrelation
        exp(-(2 pi B t)^2 / 2). B = 2 f speed_sin / c."""
        s, k = base.delay.shape
        n_sin = self.doppler_lines
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
            tap_key = _mix(key[..., None] ^ (np.arange(self.taps, dtype=np.uint64) + np.uint64(1)) * _M3)     # [S, K, J]
            key = _mix(tap_key[..., None] ^ (np.arange(n_sin, dtype=np.uint64) + np.uint64(1)) * _M1)     # [S, K, J, M]
        width = 2.0 * self.frequency * speed_sin / float(base.sound_speed)                 # Hz, [S, K]
        if self.doppler_spectrum == "gaussian":
            unit = np.sqrt(-2.0 * np.log(_uniform(key, 1))) * np.cos(2.0 * np.pi * _uniform(key, 3))   # Box-Muller
            nu = width[..., None, None] * unit
        else:
            nu = width[..., None, None] * np.tan(np.pi * (_uniform(key, 1) - 0.5))
        phase = 2.0 * np.pi * (_uniform(key, 2) + nu * self.time)
        return np.exp(1j * phase).sum(axis=-1) / np.sqrt(n_sin), nu, phase, tap_key


def wrap(provider: EigenrayProvider, cfg: dict, *, frequency: float) -> EigenrayProvider:
    """Wrap a provider according to a scenario's `time_spread` section:
    {surface: {wind_speed | rms_height, mean_square_slope, rms_vertical_speed}, bottom: {...},
     taps, min_spacing, coherence_length, seed, incoherent,
     angle_spread, decay_time, doppler, doppler_spectrum, doppler_lines, multi_bounce}."""
    surface = boundary_parameters(cfg.get("surface"), surface=True, frequency=float(cfg.get("frequency", frequency)))
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
                              incoherent=bool(cfg.get("incoherent", True)),
                              angle_spread=bool(cfg.get("angle_spread", False)),
                              decay_time=str(cfg.get("decay_time", "rule_of_thumb")),
                              doppler=bool(cfg.get("doppler", False)),
                              doppler_spectrum=str(cfg.get("doppler_spectrum", "gaussian")),
                              doppler_lines=int(cfg.get("doppler_lines", 8)),
                              multi_bounce=str(cfg.get("multi_bounce", "convolve")))
