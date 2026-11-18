"""Surface-image (Lloyd-mirror) interference and the layer's image series, as a factor on a link budget.

A link budget with spherical spreading takes the direct path only. A source and a target both below a sea
surface are also joined by the path reflected at the surface, which arrives almost as strong and with the
opposite sign; the two interfere. This module gives the factor that puts that into a budget, one way and two
way (an active sonar: projector and array at one depth, target at another), with

  the surface's roughness     coherent reflection exp(-Gamma^2 / 2), Gamma = 2 k h sin(grazing)
                              (`acoustics.rough_coherent_reflection`); the rms height h is an input of its own, the
                              Pierson-Moskowitz wind law only its default (docs/PHYSICS_NOTES.md Sec. 21: measured
                              heights did not follow the wind law)
  bubble loss                 dB per surface bounce from a named wind law of `sim/surface.py` (`apl_1994`)
  the band                    the factor is evaluated across the transmitted band: energy over the band, and the
                              peak of the compressed echo, against the single-frequency value
  the incoherent limit        what a rough surface scatters out of the coherent path still arrives, with no fixed
                              phase: it is kept as power
  a bottom, optionally        the image series of an isovelocity layer over a fluid half-space (Rayleigh
                              reflection at each image's grazing angle): the ducted case, without Bellhop

Derivation (isovelocity water, exp(+i w t), a unit source). Horizontal range R, own depth z1, target depth z2.

    direct path      R_d = sqrt(R^2 + (z1 - z2)^2)           amplitude 1 / R_d
    surface image    R_s = sqrt(R^2 + (z1 + z2)^2)           amplitude mu / R_s,  sin(grazing) = (z1 + z2) / R_s
    mu = -rho_c 10^(-L_b / 20),  rho_c = exp(-Gamma^2 / 2)   (flat, no bubbles: mu = -1)

    H(f) = R_d (exp(-i k R_d) / R_d + mu exp(-i k R_s) / R_s)        one-way field re the direct path alone
    |H|^2 = 1 + m^2 - 2 m cos(k (R_s - R_d)),  m = |mu| R_d / R_s    -> 4 sin^2(2 pi z1 z2 / (lambda R)) for mu = -1, R >> z

which is the textbook Lloyd-mirror pattern: nulls at z1 z2 = n lambda R / 2 (n = 0: the surface itself), maxima
(+6 dB one way) at z1 z2 = (2 n + 1) lambda R / 4. The part of the reflected energy that is not coherent,
S = (1 - rho_c^2) 10^(-L_b / 10) (R_d / R_s)^2, has no fixed phase; the mean one-way intensity is |H|^2 + S.

Two way, monostatic: the echo travels the same set of paths out and back (reciprocity), so its transfer function is
H(f)^2. Over the two-way travel time (16 s at 12 km) the surface has changed, so the scattered parts of the two
legs are independent and the mean two-way intensity is (|H|^2 + S)^2. Reported per geometry:

    narrowband_db   10 log10 (|H(fc)|^2 + S(fc))^2                      the single-frequency textbook figure
    energy_db       10 log10 mean_f (|H(f)|^2 + S(f))^2                 all the echo's energy, whenever it arrives
    peak_db         10 log10 max_t |mean_f H(f)^2 exp(i 2 pi f t)|^2     the peak of the compressed echo's coherent part:
                                                                        what a matched filter and a peak detector see

`peak_db` is the one a detection budget wants (the budget's SNR is peak signal power over mean noise power); it
leaves the scattered energy out (that energy is spread in angle and delay), so it is the cautious one of the two.
With one path all three are 0 dB. With a bottom the sum runs over the layer's images (2 per bounce count).

Not in the model: refraction (a sound-speed profile bends the paths: use Bellhop, `bellhop_transfer`), a target
inside the bubble layer (the law removes energy from the reflected path only), the scattered energy's spread in
time and angle, a rough bottom, a target's vertical extent other than through `depth_average`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from . import acoustics


@dataclass
class Surface:
    """The sea surface as the budget needs it. `rms_height` (m) wins over the wind law when given."""

    wind_speed: float = 0.0                 # m/s at 10 m: the wind law's height when rms_height is None, and the bubble law
    rms_height: float | None = None         # m, rms displacement of the surface
    bubble_model: str | None = None         # a key of sim.surface.BUBBLE_LOSS_MODELS (`apl_1994`), or None: no bubble loss
    coefficient: float = -1.0               # flat-surface pressure reflection coefficient (pressure release)

    def height(self) -> float:
        return float(acoustics.rms_wave_height(self.wind_speed) if self.rms_height is None else self.rms_height)

    def coherent(self, grazing, frequency):
        """Amplitude factor of the coherent reflection relative to a flat surface, at grazing angles (rad) and
        frequencies (Hz) that broadcast together."""
        f = np.asarray(frequency, dtype=float)
        f_ref = float(np.mean(f))
        # exp(-Gamma^2 / 2) with Gamma proportional to frequency: the repository's function at one frequency, raised to (f / f_ref)^2
        return acoustics.rough_coherent_reflection(grazing, self.height(), f_ref, self._c) ** ((f / f_ref) ** 2)

    def bubble_loss_db(self, grazing, frequency: float):
        """dB of energy lost per surface bounce (0 without a bubble model or without wind)."""
        g = np.asarray(grazing, dtype=float)
        if self.bubble_model is None or self.wind_speed <= 0.0:
            return np.zeros_like(g)
        from .sim import surface as surface_model
        try:
            fn = surface_model.BUBBLE_LOSS_MODELS[self.bubble_model]
        except KeyError as e:
            raise ValueError(f"unknown bubble loss model {self.bubble_model!r}; one of {sorted(surface_model.BUBBLE_LOSS_MODELS)}") from e
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(fn(g, self.wind_speed, float(frequency), self._c), dtype=float)

    _c: float = field(default=acoustics.NOMINAL_SOUND_SPEED, repr=False)

    @classmethod
    def from_config(cls, cfg: dict | None, sound_speed: float = acoustics.NOMINAL_SOUND_SPEED) -> "Surface":
        """{wind_speed | sea_state, rms_height, bubble_model} -> Surface. `sea_state` (0 to 9) means its median wind speed."""
        cfg = dict(cfg or {})
        if "sea_state" in cfg:
            wind = acoustics.sea_state_wind_speed(cfg.pop("sea_state"))
            if "wind_speed" in cfg and abs(float(cfg["wind_speed"]) - wind) > 1e-9:
                raise ValueError("surface: give sea_state or wind_speed, not both")
            cfg["wind_speed"] = wind
        unknown = set(cfg) - {"wind_speed", "rms_height", "bubble_model", "coefficient"}
        if unknown:
            raise ValueError(f"surface: unknown keys {sorted(unknown)}; keys are wind_speed, sea_state, rms_height, bubble_model")
        out = cls(wind_speed=float(cfg.get("wind_speed", 0.0)), rms_height=None if cfg.get("rms_height") is None else float(cfg["rms_height"]),
                  bubble_model=cfg.get("bubble_model"), coefficient=float(cfg.get("coefficient", -1.0)))
        out._c = float(sound_speed)
        return out


FLAT = Surface()


@dataclass
class Layer:
    """Isovelocity water. `depth` None: no bottom (or one that returns nothing): direct path and surface image only,
    the non-ducting case. With `depth` and `bottom` ({speed m/s, density re water, attenuation dB/wavelength} or
    {sediment: name in configs/sediments.yaml}): the image series to `max_order` bounces."""

    depth: float | None = None
    bottom: dict | None = None
    sound_speed: float = acoustics.NOMINAL_SOUND_SPEED
    max_order: int = 0                       # 0: chosen from the geometry (every image inside `max_grazing_deg`)
    max_grazing_deg: float = 45.0
    absorption_db_per_m: float = 0.0         # applied to each image's extra path length only (the budget carries the direct path's)

    def halfspace(self) -> dict | None:
        if self.depth is None or self.bottom is None:
            return None
        if "sediment" in self.bottom:
            from .sim import bottom as bottom_model
            rest = {k: v for k, v in self.bottom.items() if k not in ("sediment", "sediment_file")}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return dict(bottom_model.fluid_halfspace(self.bottom["sediment"], self.sound_speed, self.bottom.get("sediment_file")), **rest)
        return {k: float(self.bottom[k]) for k in ("speed", "density", "attenuation") if k in self.bottom}


def _images(z, depth, order: int):
    """(image depth, surface bounces, bottom bounces) of a point at depth z in a layer 0..depth (None: surface only)."""
    yield z, 0, 0
    for first in ("s", "b"):
        if first == "b" and depth is None:
            continue
        img, ns, nb, nxt = z, 0, 0, first
        for _ in range(order):
            if nxt == "s":
                img, ns, nxt = -img, ns + 1, "b"
            else:
                img, nb, nxt = 2.0 * depth - img, nb + 1, "s"
            yield img, ns, nb
            if depth is None:
                break


def band(f_lo: float, f_hi: float, n: int = 129) -> np.ndarray:
    """`n` frequencies across a band, centre included (n odd)."""
    n = int(n) | 1
    return np.linspace(float(f_lo), float(f_hi), n)


def one_way(frequencies, own_depth, target_depth, range_m, surface: Surface = FLAT, layer: Layer | None = None):
    """(H, S): the coherent one-way field and the scattered power, both relative to the direct path alone, at
    `frequencies` (last axis). own_depth, target_depth and range_m (horizontal) broadcast together."""
    layer = layer or Layer()
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    z1, z2, r = np.broadcast_arrays(np.asarray(own_depth, dtype=float), np.asarray(target_depth, dtype=float), np.asarray(range_m, dtype=float))
    if np.any(z1 < 0) or np.any(z2 < 0) or np.any(r <= 0):
        raise ValueError("depths must be >= 0 (m below the surface) and ranges > 0")
    c = float(layer.sound_speed)
    if layer.depth is not None and (np.any(z1 > layer.depth) or np.any(z2 > layer.depth)):
        raise ValueError(f"a depth is below the bottom of the layer ({layer.depth:g} m)")
    half = layer.halfspace()
    if layer.depth is None or half is None:
        order, depth = 1, None
    else:
        depth = float(layer.depth)
        order = int(layer.max_order) or int(np.ceil(float(np.max(r)) * np.tan(np.radians(layer.max_grazing_deg)) / depth)) + 2
    k = 2.0 * np.pi * f / c
    r_d = np.sqrt(r**2 + (z1 - z2) ** 2)[..., None]
    h = np.zeros(r_d.shape[:-1] + f.shape, dtype=complex)
    s = np.zeros(h.shape)
    fc = float(f[f.size // 2])
    alpha = layer.absorption_db_per_m * np.log(10.0) / 20.0                       # nepers of amplitude per metre
    if half is not None:
        from .sim.propagation import rayleigh_reflection
    for img, ns, nb in _images(z2, depth, order):
        r_j = np.sqrt(r**2 + (z1 - img) ** 2)[..., None]
        amp = (r_d / r_j) * np.exp(-alpha * (r_j - r_d))
        total = amp**2
        coh = amp.astype(complex)
        if ns or nb:
            grazing = np.arcsin(np.clip(np.abs(z1 - img)[..., None] / r_j, 0.0, 1.0))
            if ns:
                loss = 10.0 ** (-surface.bubble_loss_db(grazing, fc) / 20.0)      # amplitude per bounce
                rho = surface.coherent(grazing, f)
                coh = coh * (surface.coefficient * loss * rho) ** ns
                total = total * (surface.coefficient**2 * loss**2) ** ns
            if nb:
                rb = rayleigh_reflection(grazing, c, **half)
                coh = coh * rb**nb
                total = total * np.abs(rb) ** (2 * nb)
        h += coh * np.exp(-1j * k * (r_j - r_d))
        s += np.maximum(total - np.abs(coh) ** 2, 0.0) if ns else 0.0
    return h, s


def summarise(h, s, h_tx=None, s_tx=None, pad: int = 8) -> dict:
    """The three two-way figures (module docstring), in dB re the free-field echo, from one-way (H, S) with frequency
    on the last axis, plus the one-way ones. `h_tx`, `s_tx`: the outbound leg when the projector is not at the array's
    depth (the echo's transfer function is then H_tx H, and the mean intensity (|H_tx|^2 + S_tx)(|H|^2 + S))."""
    nf = h.shape[-1]
    e1 = np.abs(h) ** 2 + s
    if h_tx is None:
        h_tx, e_tx = h, e1
    else:
        e_tx = np.abs(h_tx) ** 2 + (0.0 if s_tx is None else s_tx)
    mid = nf // 2
    spectrum = np.fft.fft(h * h_tx, n=int(pad) * nf, axis=-1) / nf
    tiny = 1e-30
    return {"narrowband_db": 10.0 * np.log10(np.maximum(e1[..., mid] * e_tx[..., mid], tiny)),
            "energy_db": 10.0 * np.log10(np.maximum(np.mean(e1 * e_tx, axis=-1), tiny)),
            "peak_db": 10.0 * np.log10(np.maximum(np.max(np.abs(spectrum) ** 2, axis=-1), tiny)),
            "one_way_narrowband_db": 10.0 * np.log10(np.maximum(e1[..., mid], tiny)),
            "one_way_energy_db": 10.0 * np.log10(np.maximum(np.mean(e1, axis=-1), tiny)),
            "coherent_share": np.mean(np.abs(h) ** 2, axis=-1) / np.maximum(np.mean(e1, axis=-1), tiny)}


def two_way(f_lo: float, f_hi: float, own_depth, target_depth, range_m, surface: Surface = FLAT, layer: Layer | None = None, n_f: int = 129,
            projector_below_m: float = 0.0) -> dict:
    """The two-way factor on a monostatic echo, dB re the free-field budget, for a band [f_lo, f_hi]: see `summarise`.
    `own_depth` is the array's; the projector is `projector_below_m` deeper (0: at the array's depth). In a deep null the
    difference matters: the factor there goes as (projector depth x array depth x target depth^2)."""
    f = band(f_lo, f_hi, n_f)
    h, s = one_way(f, own_depth, target_depth, range_m, surface, layer)
    if not projector_below_m:
        return summarise(h, s)
    h_tx, s_tx = one_way(f, np.asarray(own_depth, dtype=float) + float(projector_below_m), target_depth, range_m, surface, layer)
    return summarise(h, s, h_tx, s_tx)


def depth_average(values_db, axis: int = -1):
    """Power mean, in dB, along an axis: the factor for a target whose scattering is spread evenly over the depths on that axis."""
    return 10.0 * np.log10(np.mean(10.0 ** (np.asarray(values_db, dtype=float) / 10.0), axis=axis))


def lloyd_two_way_db(frequency: float, own_depth, target_depth, range_m, sound_speed: float = acoustics.NOMINAL_SOUND_SPEED):
    """The textbook single-frequency figure for a flat surface and R >> depths: 2 x 20 log10 |2 sin(2 pi z1 z2 / (lambda R))|.
    An independent closed form the tests hold `two_way` to."""
    lam = float(sound_speed) / float(frequency)
    x = 2.0 * np.pi * np.asarray(own_depth, dtype=float) * np.asarray(target_depth, dtype=float) / (lam * np.asarray(range_m, dtype=float))
    return 40.0 * np.log10(np.maximum(np.abs(2.0 * np.sin(x)), 1e-15))


def first_lobe_own_depth(frequency: float, target_depth: float, range_m: float, sound_speed: float = acoustics.NOMINAL_SOUND_SPEED) -> dict:
    """Own depths of the first Lloyd lobe for a target at `target_depth` and horizontal range `range_m`, flat surface, one
    frequency: the peak (z1 z2 = lambda R / 4: +12 dB two way), where the factor crosses 0 dB on the way up
    (lambda R / 12) and the first null below the surface (lambda R / 2)."""
    lam_r = float(sound_speed) / float(frequency) * float(range_m)
    return {"peak_m": lam_r / (4.0 * target_depth), "zero_db_m": lam_r / (12.0 * target_depth), "first_null_m": lam_r / (2.0 * target_depth),
            "depth_product_m2": lam_r / 4.0}


# --- Bellhop: the same figures from its arrivals, for a sound-speed profile -----------------------------------------------

def bellhop_transfer(arrivals, i_depth: int, i_range: int, frequencies, surface: Surface = FLAT, speed_at_source: float = acoustics.NOMINAL_SOUND_SPEED,
                     speed_at_surface: float = acoustics.NOMINAL_SOUND_SPEED):
    """(H, S) as `one_way` returns them, from one receiver of a Bellhop arrivals set (sim.bellhop.run_bellhop): each
    arrival's amplitude (computed at the run's frequency) with its delay's phase at every frequency of the band,
    relative to spherical spreading over the slant range. Bellhop's surface is flat; the rough surface's coherent
    factor and the bubble loss are applied per surface bounce at the grazing angle Snell's law gives at the surface
    for the arrival's launch angle. By reciprocity one run from the target's depth serves every own depth in its grid."""
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    a = np.asarray(arrivals.amplitude[i_depth][i_range])
    t = np.asarray(arrivals.delay[i_depth][i_range])
    ns = np.asarray(arrivals.n_surface[i_depth][i_range])
    slant = float(np.hypot(arrivals.ranges[i_range], 0.0))                         # reference: spherical spreading over the horizontal range
    if a.size == 0:
        return np.zeros(f.shape, dtype=complex), np.zeros(f.shape)
    launch = np.asarray(arrivals.launch_angle[i_depth][i_range])
    cos_s = np.clip(np.cos(launch) * speed_at_surface / speed_at_source, -1.0, 1.0)
    grazing = np.arccos(cos_s)[:, None]
    fc = float(f[f.size // 2])
    loss = 10.0 ** (-surface.bubble_loss_db(grazing, fc) / 20.0)
    rho = surface.coherent(grazing, f[None, :])
    coh = (a[:, None] * slant) * (loss * rho) ** ns[:, None]
    total = (np.abs(a[:, None]) * slant) ** 2 * loss ** (2 * ns[:, None]) * np.ones_like(rho)
    h = np.sum(coh * np.exp(-2j * np.pi * f[None, :] * (t[:, None] - float(np.min(t)))), axis=0)    # the common delay taken out
    s = np.sum(np.where(ns[:, None] > 0, np.maximum(total - np.abs(coh) ** 2, 0.0), 0.0), axis=0)
    return h, s
